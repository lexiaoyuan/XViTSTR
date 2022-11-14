import os
import random
import re
import string
import sys
import time
from test import validation

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter


from dataset import AlignCollate, Batch_Balanced_Dataset, hierarchical_dataset
from model import Model
from utils import (Averager, TokenLabelConverter, get_args, Loss_l)

device = torch.device('cuda')


def train(opt):
    # 配置TensorBoard保存目录
    writer = SummaryWriter(log_dir=f'./runs/{opt.exp_name}')
    """ 加载数据 """
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    opt.eval = False
    # 获取训练集对象
    train_dataset = Batch_Balanced_Dataset(opt)

    # 加载验证集
    log = open(f'./saved_models/{opt.exp_name}/dataset_log.txt', 'a')
    opt.eval = True
    if opt.sensitive:
        opt.data_filtering_off = True
    AlignCollate_valid = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(
        opt.workers), collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    """ 模型配置 """
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3

    # 创建XViTSTR模型
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
  
    # 将模块设置为训练模式（将本层及子层的training属性同时设为true）
    model.train()
    if opt.saved_model != '':
        model.load_state_dict(torch.load(opt.saved_model))

    # 设置损失
    # 使用交叉熵损失
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=0).to(device)  # 忽略[GO]标记=忽略索引0
    loss_avg = Averager()
    # filtered_parameters中只包含那些能够进行梯度下降的参数
    filtered_parameters = []
    params_num = []
    # filter()函数将model.parameters()中requires_grad=True的返回
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    # 设置优化器
    scheduler = None
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters,
                               lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        # 使用Adadelta优化器
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    if opt.scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_iter)

    """ 最终的参数 """
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ 参数 -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'可训练网络参数数量 : {total_params:,}'
        print(total_params)
        opt_file.write(total_params)

    """ 开始训练 """
    start_iter = 0
    # 配置saved_model可以进行断点训练
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0]) - 1
        except:
            pass
    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter
    
    """ 超参数设置 """
    # Foc的惩罚因子
    alptha = 0.5
    Foc = 0
    # Loc的惩罚因子
    beta = 0.5
    Loc = 0

    # loss_fq的惩罚因子
    lambd1 = 1
    loss_fq = 0
    # loss_fk的惩罚因子
    lambd2 = 2
    loss_fk = 0
    # loss_fv的惩罚因子
    lambd3 = 1
    loss_fv = 0

    # loss_lq的惩罚因子
    mu1 = 10
    loss_lq = 0
    # loss_lk的惩罚因子
    mu2 = 20
    loss_lk = 0
    # loss_lv的惩罚因子
    mu3 = 10
    loss_lv = 0

    # 达到循环次数opt.num_iter=300000后，退出循环
    while(True):
        # 训练部分
        image_tensors, labels = train_dataset.get_batch()
        # 将一个带有固定内存的CPU张量转换为CUDA张量
        image = image_tensors.to(device)
        target = converter.encode(labels)
        # 调用Model类的forward方法
        preds = model(image, seqlen=converter.batch_max_length)
        cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        
        """ 论文实验 """
        #################### Foc实验 ####################
        loss_fq = model.module.get_loss_fq()
        with open(f"./figures/data/{opt.exp_name}_loss_fq.txt", "a") as f:
            f.write(f"{iteration} {loss_fq}\n")
        writer.add_scalar('Train/Loss_FQ', loss_fq, iteration)
        loss_fk = model.module.get_loss_fk()
        with open(f"./figures/data/{opt.exp_name}_loss_fk.txt", "a") as f:
            f.write(f"{iteration} {loss_fk}\n")
        writer.add_scalar('Train/Loss_FK', loss_fk, iteration)
        loss_fv = model.module.get_loss_fv()
        with open(f"./figures/data/{opt.exp_name}_loss_fv.txt", "a") as f:
            f.write(f"{iteration} {loss_fv}\n")
        writer.add_scalar('Train/Loss_FV', loss_fv, iteration)
        
        #################### Loc实验 ####################
        loss_lq, loss_lk, loss_lv = Loss_l(model.module.get_qkv_weights())
        with open(f"./figures/data/{opt.exp_name}_loss_lq.txt", "a") as f:
            f.write(f"{iteration} {loss_lq}\n")
        with open(f"./figures/data/{opt.exp_name}_loss_lk.txt", "a") as f:
            f.write(f"{iteration} {loss_lk}\n")
        with open(f"./figures/data/{opt.exp_name}_loss_lv.txt", "a") as f:
            f.write(f"{iteration} {loss_lv}\n")
        writer.add_scalar('Train/Loss_LQ', loss_lq, iteration)
        writer.add_scalar('Train/Loss_LK', loss_lk, iteration)
        writer.add_scalar('Train/Loss_LV', loss_lv, iteration)
        

        Foc = lambd1 * loss_fq + lambd2 * loss_fk + lambd3 * loss_fv
        Loc = mu1 * loss_lq + mu2 * loss_lk + mu3 * loss_lv
        cost = cost + alptha * Foc + beta * Loc
        
        model.zero_grad()
        cost.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        loss_avg.add(cost)
        
        with open(f"./figures/data/{opt.exp_name}_loss.txt", "a") as f:
            f.write(f"{iteration} {loss_avg.val()}\n")
        writer.add_scalar('Train/Loss', loss_avg.val(), iteration)
        
        # 验证部分
        if (iteration + 1) % opt.val_interval == 0 or iteration == 0:
            elapsed_time = time.time() - start_time
            # 保存训练日志
            with open(f'./saved_models/{opt.exp_name}/train_log.txt', 'a') as log:
                # 将模块设置为评估模式
                model.eval()
                with torch.no_grad():
                    """ 
                        valid_loss: 验证集的平均损失
                        current_accuracy：验证集的预测精度
                        current_norm_ED： 验证集的归一化编辑距离
                        preds：对验证集预测的标签列表
                        confidence_score：对验证集预测的置信度分数列表
                        labels：验证集的最后一个batch的真实标签列表
                    """
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                # 将模块设置为训练模式
                model.train()
                # 记录相关信息到TensorBoard
                writer.add_scalar('Vaild/Loss', valid_loss, iteration)
                writer.add_scalar(
                    'Learning Rate', optimizer.param_groups[0]['lr'], iteration)
                writer.add_scalar('Current Accuracy',
                                  current_accuracy, iteration)
                writer.add_scalar('Current Norm ED',
                                  current_norm_ED, iteration)
                # 训练损失和验证损失
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}, Learning Rate: {optimizer.param_groups[0]["lr"]:0.5f}'
                loss_avg.reset()
                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'
                # 保持最佳精度模型（在验证数据集上）
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(
                        model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')
                # 显示一些预测结果
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    pred = pred[:pred.find('[s]')]
                    # 使用字母数字和不区分大小写设置评估“区分大小写模型”
                    if opt.sensitive and opt.data_filtering_off:
                        pred = pred.lower()
                        gt = gt.lower()
                        alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                        out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                        pred = re.sub(
                            out_of_alphanumeric_case_insensitve, '', pred)
                        gt = re.sub(
                            out_of_alphanumeric_case_insensitve, '', gt)
                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # 每迭代1e+4次保存模型.
        if (iteration + 1) % 1e+4 == 0:
            torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            writer.close()
            sys.exit()
        iteration += 1
        if scheduler is not None:
            scheduler.step()


if __name__ == "__main__":
    # 获取参数
    opt = get_args()
    if not opt.exp_name:
        opt.exp_name = f'{opt.TransformerModel}'
    opt.exp_name += f'-Seed{opt.manualSeed}'
    if opt.exp:
        opt.exp_name += f'-{opt.exp}'
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    if opt.sensitive:
        opt.character = string.printable[:-6]

    # 随机种子和GPU设置
    print(f"随机种子：{opt.manualSeed}")
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    # torch.cuda.empty_cache()
    opt.num_gpu = torch.cuda.device_count()
    if opt.workers <= 0:
        opt.workers = (os.cpu_count() // 2) // opt.num_gpu
    if opt.num_gpu > 1:
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

    # 训练
    train(opt)
