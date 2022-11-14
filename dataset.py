import os
import re
import sys

import lmdb
import numpy as np
import six
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch._utils import _accumulate
from torch.utils.data import ConcatDataset, Dataset, Subset


class Batch_Balanced_Dataset():
    def __init__(self, opt):
        """ 平衡batch中的数据集

        初始化之后，会得到一个data_loader_list和一个dataloader_iter_list，
        data_loader_list中是MJ数据集和ST数据集的数据加载器对象，每个加载器对象会按照batch_size=96来加载对应数据集中的数据。
        """
        log = open(f'./saved_models/{opt.exp_name}/dataset_log.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(
            f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(
            f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)
        # _AlignCollate是一个实例，会根据配置的参数，初始化一些属性，这个实例会在DataLoader()中使用到
        _AlignCollate = AlignCollate(
            imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        # 总共的batch size是192，batch_ratio是0.5，表示在一个batch中，有96张图片来自MJ，96张图片来自ST
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            # _dataset是root目录下指定的selected_d数据集拼接后的数据集对象，这里是训练集对象
            _dataset, _dataset_log = hierarchical_dataset(
                root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)
            number_dataset = int(total_number_dataset *
                                 float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset,
                             total_number_dataset - number_dataset]

            indices = range(total_number_dataset)
            # 由于opt.total_data_usage_ratio=1.0，这里的子集的操作其实就没有意义了，还是用的全部拼接后的数据集
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size
            # 分别得到MJ和ST数据集下的所有数据，加载数据时，使用batch_size=96来加载数据
            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size
        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []
        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                # 每个image中有96张图片, 通道数是1
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                # 开始训练后，这里还没有执行。但是不确定这里究竟会不会执行
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/'):
    """ 返回root目录下的数据集拼接后的数据集对象，和数据集日志 """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                # dataset是一个可以获取lmdb格式中图片和标签的对象
                dataset = LmdbDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    # 将数据集列表中的数据集对象拼接成一个数据集对象
    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):
    """ 
    加载和获取lmdb类型的数据。
    实例化后，会初始化一些属性，包括数据集的目录(root)、配置的参数(opt)、
    数据库环境结构对象(env)、样本数(nSamples)、样本的索引列表(filtered_index_list)。
    使用`dataset = LmdbDataset(dirpath, opt)`实例化对象后，可以通过`dataset[index]`的方式获取图片和对应的标签。
    """

    def __init__(self, root, opt):
        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            if self.opt.data_filtering_off:
                # 用于快速检查或基准评估，无需过滤
                self.filtered_index_list = [
                    index + 1 for index in range(self.nSamples)]
            else:
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.opt.batch_max_length:
                        continue

                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    # 在这里读取图片的时候，将图片转为灰度图
                    img = Image.open(buf).convert('L')
                    # 这里图片还没有被reshape，依然是原图大小
            except IOError:
                print(f'Corrupted image for {index}')
                # 为损坏的图像制作虚拟图像和虚拟标签。
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            # 这里是区分大小写的，没有把标签都转换为小写
            if not self.opt.sensitive:
                label = label.lower()

            # 我们只对字母数字（或train.py中的预定义字符集）进行训练和评估
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


def isless(prob=0.5):
    return np.random.uniform(0, 1) < prob


class DataAugment():
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, img):
        # 在这个地方调整图片大小的！
        img = img.resize((self.opt.imgW, self.opt.imgH), Image.BICUBIC)
        img = transforms.ToTensor()(img)
        return img


class PAD():
    def __init__(self, max_w, max_h) -> None:
        self.max_w = max_w
        self.max_h = max_h

    def __call__(self, img):
        w, h = img.size
        c = 3 if img.mode == 'RGB' else 1
        pad_img = torch.FloatTensor(c, self.max_h, self.max_w).fill_(0)
        if w <= self.max_w and h <= self.max_h:
            # 直接填充
            resized_w = w
            resized_h = h
        elif w > self.max_w and h <= self.max_h:
            # w调整到max_w，h按原图等比例调整
            resized_w = self.max_w
            resized_h = round(self.max_w * h / w)  # 四舍五入
        elif w <= self.max_w and h > self.max_h:
            # h调整到max_h, w按原图等比例调整
            resized_h = self.max_h
            resized_w = round(self.max_h * w / h)  # 四舍五入
        elif w > self.max_w and h > self.max_h:
            # w, h都调整到max_w, max_h
            resized_w = self.max_w
            resized_h = self.max_h
        img = img.resize((resized_w, resized_h), Image.BICUBIC)
        img = transforms.ToTensor()(img)
        pad_img[:, :resized_h, :resized_w] = img
        return pad_img


class AlignCollate():
    """ 整理图片

    统一图片的大小，并将图片转化为张量。
    返回图片张量和对应的标签
    """

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, opt=None):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.opt = opt

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        # zip() 与 * 运算符相结合可以用来拆解一个列表
        # 这里的images还是原图大小，没有被reshape
        images, labels = zip(*batch)
        if self.keep_ratio_with_pad:
            transform = PAD(self.imgW, self.imgH)
            pad_images = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in pad_images], 0)
        else:
            transform = DataAugment(self.opt)
            # 在执行transform(image)的时候，执行了DataAugment类中的call方法，在该call方法中调整了图片的大小为224*224
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0)
                                      for t in image_tensors], 0)
        return image_tensors, labels
