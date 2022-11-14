import argparse
import torch
import torch.nn.functional as F


device = torch.device('cuda')


def get_args(is_train=True):
    parser = argparse.ArgumentParser(description="XViTSTR")

    # 训练相关参数
    parser.add_argument("--exp_name", help='保存日志和模型的目录名')
    parser.add_argument("--exp", type=str, help='实验名')
    choices = ['xvitstr_tiny_patch16_224', 'xvitstr_small_patch16_224', 'xvitstr_base_patch16_224']
    parser.add_argument('--TransformerModel',
                        default=choices[0], help='哪一个vit模型', choices=choices)
    parser.add_argument('--manualSeed', type=int, default=1111, help='设置随机种子')
    parser.add_argument('--sensitive', action='store_true', help='使用字符敏感模型')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='字符类别')
    parser.add_argument('--workers', type=int, default=4,
                        help='加载数据的线程数。-1表示使用全部的核')
    parser.add_argument('--batch_size', type=int,
                        default=192, help='输入的batch size')
    parser.add_argument('--data_filtering_off',
                        action='store_true', help='使用数据过滤关闭')
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='选择训练数据(默认是MJ-ST,表示使用MJ和ST两个数据集作为训练数据)')
    parser.add_argument('--batch_ratio', type=str,
                        default='0.5-0.5', help='在一个batch中为被选择的数据的分配比例')
    parser.add_argument('--imgH', type=int, default=224, help='输入图片的高')
    parser.add_argument('--imgW', type=int, default=224, help='输入图片的宽')
    parser.add_argument('--PAD', action='store_true',
                        help='是否保留比例，然后填充以调整图像大小')
    parser.add_argument('--rgb', action='store_true', help='使用RGB输入')
    parser.add_argument('--train_data', required=is_train, help='训练数据集的路径')
    parser.add_argument('--valid_data', required=is_train, help='验证数据集的路径')
    parser.add_argument('--total_data_usage_ratio', type=str,
                        default='1.0', help='总数据使用率，此比率乘以数据总数')
    parser.add_argument('--batch_max_length', type=int,
                        default=25, help='标签长度的最大值')
    parser.add_argument('--input_channel', type=int,
                        default=1, help='特征提取器的输入通道数')
    parser.add_argument('--saved_model', default='', help="继续训练的模型路径")
    parser.add_argument('--val_interval', type=int,
                        default=2000, help='每次验证之间的间隔')
    parser.add_argument('--adam', action='store_true',
                        help='是否使用Adam（默认使用Adadelta）')
    parser.add_argument('--scheduler', action='store_true', help='使用学习率周期')
    parser.add_argument('--lr', type=float, default=1,
                        help='学习率，Adadelta默认为1.0')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='adam的beta1. 默认是0.9')
    parser.add_argument('--rho', type=float, default=0.95,
                        help='Adadelta的衰变率rho。默认值=0.95')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Adadelta的eps。默认值=1e-8')
    parser.add_argument('--num_iter', type=int, default=300000,
                        help='要训练的迭代次数。默认值=300000')
    parser.add_argument('--grad_clip', type=float, default=5,
                        help='梯度剪裁值。默认值=5')
    parser.add_argument('--isTrain', type=bool,
                        default=is_train, help='是否是训练模式。默认为训练模式')

    # 测试相关参数
    parser.add_argument('--benchmark_all_eval', action='store_true',
                        help='评估10个基准评估数据集')
    # 原论文将其用于快速基准测试
    parser.add_argument('--fast_acc', action='store_true',
                        help='快速平均精度计算')
    parser.add_argument('--calculate_infer_time',
                        action='store_true', help='计算推理时间')
    parser.add_argument('--eval_data', required=not is_train,
                        help='评估数据集的路径')
    parser.add_argument('--img_path', type=str, default='', help='测试单张图像的路径')
    args = parser.parse_args()
    return args


class TokenLabelConverter():
    def __init__(self, opt):
        """
        初始化后可以得到94个字符和一个开始标记'[GO]'，一个结束标记'[s]'，总共96个字符的字典，字符为键，0~95为值。
        """
        # [GO]获取注意力解码器的开始标记。[s] 用于句子结尾标记。
        self.GO = '[GO]'
        self.SPACE = '[s]'
        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(opt.character)
        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = opt.batch_max_length + len(self.list_token)

    def encode(self, text):
        """ 在文本标签和文本索引之间转换.
        输入:
            text: 一个长度为batch_size=192的标签列表
        输出：
            batch_text: 一个batch_size*self.batch_max_length=192*27的张量，张量的每一行是一个标签的每个字母的索引，每一行都是用0开头，每一行的索引不足25个的，后面都是填充0。
        """
        batch_text = torch.LongTensor(
            len(text), self.batch_max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)
        return batch_text.to(device)

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager():
    """ 计算torch.Tensor的平均值，用于损失平均值 """

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def Loss_f(feature):
    heads = feature.reshape(feature.shape[0], feature.shape[1], -1)
    heads_norm = F.normalize(heads, dim=2)
    P = torch.bmm(heads_norm, heads_norm.transpose(1, 2))
    M = torch.mean(P, dim=0)
    loss_f = torch.linalg.norm(M - torch.eye(M.shape[0], device = M.device))
    return loss_f


def Loss_l(weights, h=3):
    loss_lq = 0
    loss_lk = 0
    loss_lv = 0
    # print(len(weights)) # 12
    _, D = weights[0].shape
    for A in weights:
        A_Q = A.T[:, :D]
        loss_lq += Loss_lw(A_Q, D, h)
        # print(loss_lq)
        A_K = A.T[:, D:2*D]
        loss_lk += Loss_lw(A_K, D, h)
        # print(loss_lk)
        A_V = A.T[:, 2*D:3*D]
        loss_lv += Loss_lw(A_V, D, h)
        # print(loss_lv)
    return loss_lq, loss_lk, loss_lv

def Loss_lw(A, D, h):
    head_list = [a.T.flatten().unsqueeze(0) for a in A.split(D//h, dim=1)]
    H = torch.cat(head_list)
    H_norm = F.normalize(H)
    P = torch.mm(H_norm, H_norm.T) - torch.eye(h, device=A.device)
    return torch.linalg.norm(P)
