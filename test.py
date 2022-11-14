import os
import re
import string
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import validators
from nltk.metrics.distance import edit_distance
from PIL import Image
from torchvision import transforms

from dataset import AlignCollate, hierarchical_dataset
from model import Model
from utils import (Averager, TokenLabelConverter, get_args)


device = torch.device('cuda')


def validation(model, criterion, evaluation_loader, converter, opt):
    """ 验证或评估
    参数：
        - model：XViTSTR模型对象
        - criterion：损失函数对象
        - evaluation_loader：验证集的Dataloader对象
        - converter：转换标签和索引的对象
        - opt：配置参数对象

    返回：
        - valid_loss_avd.val()：验证集的平均损失
        - accuracy：验证集的预测精度，预测正确的图片个数/验证集的总图片个数 * 100
        - norm_ED：归一化编辑距离
        - preds_str：对验证集预测的标签列表
        - confidence_score_list：对验证集预测的置信度分数列表
        - labels：验证集的最后一个batch的真实标签列表
        - infer_time：推理时间，预测完所有验证集数据的总时间
        - length_of_data：验证集的总图片个数
    """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # 最大长度预测
        target = converter.encode(labels)
        start_time = time.time()
        preds = model(image, seqlen=converter.batch_max_length)
        _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, converter.batch_max_length)
        forward_time = time.time() - start_time
        cost = criterion(preds.contiguous().view(-1,
                         preds.shape[-1]), target.contiguous().view(-1))
        # converter.batch_max_length - 1是减掉了开始标记（第一个索引都是0）
        length_for_pred = torch.IntTensor(
            [converter.batch_max_length - 1] * batch_size).to(device)

        # 进入解码的预测值的索引不包括开始标记（第一个索引都是0）
        preds_str = converter.decode(preds_index[:, 1:], length_for_pred)
        infer_time += forward_time
        valid_loss_avg.add(cost)

        # 计算精度和置信度分数
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            """
                labels是长度为192的列表，列表中的每个值是一张图片的真实标签
                preds_str是长度为192的列表，列表中的每个值是一张图片的预测标签（含有结束标记附'[s]')
                preds_max_prob是一个192*27的张量，张量的每一行是一张图片每个字符是某个字符的概率
                gt是取一张图片的真实标签
                pred是取一张图片的预测标签
                pred_max_prob是预测标签中每个字符是某个字符的概率
            """
            pred_EOS = pred.find('[s]')
            # 裁剪调结束标记'[s]'后面的预测字符
            pred = pred[:pred_EOS]
            pred_max_prob = pred_max_prob[:pred_EOS]
            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred == gt:
                n_correct += 1
            # ICDAR2019 归一化编辑距离
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)
            # 计算置信度得分=pred_max_prob的乘积
            try:
                # cumprod计算累计乘积
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                # 对于空的pred大小写，当在“句末”标记（[s]）后进行删减时
                confidence_score = 0
            confidence_score_list.append(confidence_score)
    accuracy = n_correct / float(length_of_data) * 100
    # ICDAR2019 归一化编辑距离
    norm_ED = norm_ED / float(length_of_data)

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data


def benchmark_all_eval(model, criterion, converter, opt):
    """ 使用10个基准评估数据集进行评估 """
    if opt.fast_acc:
        # 以便于计算我们论文的总准确度。
        eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867',
                          'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
    else:
        # 评价数据集、数据集顺序与本文表1相同。
        eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867',
                          'IC13_857', 'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
    if opt.calculate_infer_time:
        # batch_size大小应为1，以计算每个图像的GPU推断时间。
        evaluation_batch_size = 1
    else:
        evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/all_evaluation_log.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(
            imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        eval_data, eval_data_log = hierarchical_dataset(
            root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy_by_best_model, norm_ED_by_best_model, _, _, _, infer_time, length_of_data = validation(
            model, criterion, evaluation_loader, converter, opt)
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        log.write(eval_data_log)
        print(
            f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}')
        log.write(
            f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}\n')
        print(dashed_line)
        log.write(dashed_line + '\n')
    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.3f}'
    print(evaluation_log)
    log.write(evaluation_log + '\n')
    log.close()


def test(opt):
    """ 模型配置 """
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    # 加载模型
    if validators.url(opt.saved_model):
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            opt.saved_model, progress=True, map_location=device))
    else:
        # 指定saved_model后，直接从本地加载权重
        print(f"加载本地训练权重{opt.saved_model}")
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # 保存评估模型和结果日志
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ 设置损失 """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(
        device)  # ignore [GO] token = ignore index 0

    """ 评估 """
    model.eval()
    opt.eval = True
    with torch.no_grad():
        if opt.benchmark_all_eval:
            benchmark_all_eval(model, criterion, converter, opt)
        else:
            AlignCollate_evaluation = AlignCollate(
                imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
            eval_data = hierarchical_dataset(root=opt.eval_data, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=True)
            _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
                model, criterion, evaluation_loader, converter, opt)
            print(f'{accuracy_by_best_model:0.3f}')


def valid_img(model, converter, opt):
    # 打开图片
    img = Image.open(opt.img_path).convert("L")
    # 调整大小
    image = img.resize((opt.imgW, opt.imgH), Image.Resampling.BICUBIC)
    # 转为张量
    image_tensor = transforms.ToTensor()(image)
    # 调整维度
    image_tensor = image_tensor.unsqueeze(0)

    infer_time = 0
    batch_size = image_tensor.size(0)
    image = image_tensor.to(device)
    start_time = time.time()
    # 模型预测
    preds = model(image, seqlen=converter.batch_max_length)

    forward_time = time.time() - start_time
    # 预测结果后处理
    _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
    preds_index = preds_index.view(-1, converter.batch_max_length)
    # 解码预测结果
    length_for_pred = torch.IntTensor(
        [converter.batch_max_length - 1] * batch_size).to(device)
    preds_str = converter.decode(preds_index[:, 1:], length_for_pred)[0]
    pred_EOS = preds_str.find('[s]')
    pred_str = preds_str[:pred_EOS]
    infer_time += forward_time
    # 计算置信度
    preds_prob = F.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)
    preds_max_prob = preds_max_prob.squeeze(0)
    pred_max_prob = preds_max_prob[:pred_EOS]
    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
    return pred_str, confidence_score, forward_time, infer_time


def test_img(opt):
    """ 模型配置 """
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    """ 加载权重 """
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    """ 推理图片 """
    model.eval()
    with torch.no_grad():
        pred_str, confidence_score, forward_time, infer_time = valid_img(
            model, converter, opt)
        print(
            f'预测结果：{pred_str} \t 置信度：{float(confidence_score)} \t 前向传播时间：{forward_time:0.4f}s \t 推理时间：{infer_time:0.4f}s')


if __name__ == "__main__":
    # 获取参数
    opt = get_args(is_train=False)
    if opt.sensitive:
        opt.character = string.printable[:-6]
    cudnn.benchmark = True
    cudnn.deterministic = True

    if opt.benchmark_all_eval:
        # 基准数据集测试
        test(opt)
    if opt.img_path != "" and os.path.isfile(opt.img_path):
        # 单张图片测试
        test_img(opt)
