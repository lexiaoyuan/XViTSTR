# 正交约束多头自注意力用于场景文本识别

[中文](./README.md) | [English](./README.en.md)

## 环境安装

1. 创建conda环境并激活
```bash
conda create --name xvitstr python=3.8 -y
conda activate xvitstr
```

2. 下载源码
```bash
git clone https://github.com/lexiaoyuan/XViTSTR.git
cd XViTSTR
```

3. 安装相关python库
```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 数据集准备
下载lmdb数据集：
- 参考[CLOVA AI Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

- 百度网盘：[https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ](https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ) 提取码：rryk

将数据集解压后保存到一个可访问的路径。数据集目录是这样的：
```
data_lmdb_release/
├── evaluation
│   ├── CUTE80
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC03_860
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC03_867
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC13_1015
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC13_857
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC15_1811
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IC15_2077
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── IIIT5k_3000
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── SVT
│   │   ├── data.mdb
│   │   └── lock.mdb
│   └── SVTP
│       ├── data.mdb
│       └── lock.mdb
├── training
│   ├── MJ
│   │   ├── MJ_test
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   ├── MJ_train
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   └── MJ_valid
│   │       ├── data.mdb
│   │       └── lock.mdb
│   └── ST
│       ├── data.mdb
│       └── lock.mdb
└── validation
    ├── data.mdb
    └── lock.mdb
```
## 快速验证
- 基准数据集测试

```bash
export CUDA_VISIBLE_DEVICES=0
python3 test.py --eval_data="data_lmdb_release/evaluation" --benchmark_all_eval --sensitive --data_filtering_off --saved_model="./saved_models/xvitstr_tiny_exp4.pth"
```
- 单张图片测试
```bash
export CUDA_VISIBLE_DEVICES=0
python3 test.py --saved_model="./saved_models/xvitstr_tiny_exp4.pth" --img_path="demo.jpg" --eval_data="" --sensitive --data_filtering_off
```

## 快速训练

- 默认训练论文中的XViTSTR-Tiny + *Foc* + *Loc*：
```bash
RANDOM=$$
export CUDA_VISIBLE_DEVICES=0
python3 train.py --train_data="data_lmdb_release/training" --valid_data="data_lmdb_release/validation" --manualSeed=$RANDOM --sensitive --adam --lr=0.001 --scheduler --exp="Name of experiment"
```
- 训练其它模型可参考源码和注释。可配置参数在[utils.py](./utils.py)中。

## 注意力可视化

> 参考：[https://github.com/jacobgil/vit-explain](https://github.com/jacobgil/vit-explain)

```bash
export CUDA_VISIBLE_DEVICES=0
python3 vit_explain.py --saved_model="./saved_models/xvitstr_tiny_exp4.pth" --img_path="demo.jpg" --head_fusion="max" --use_cuda
```

## 参考
- [Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
- [ViTSTR](https://github.com/roatienza/deep-text-recognition-benchmark)
- [vit-explain](https://github.com/jacobgil/vit-explain)
