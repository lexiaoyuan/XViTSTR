# Orthogonality constrained multihead self-attention for scene text recognition

[中文](./README.md) | [English](./README.en.md)

## Installation

1. Create a conda virtual environment and activate it.
```bash
conda create --name xvitstr python=3.8 -y
conda activate xvitstr
```

2. Clone project from GitHub
```bash
git clone https://github.com/lexiaoyuan/XViTSTR.git
cd XViTSTR
```

3. Install requirements
```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Dataset
Download lmdb dataset：
- from [CLOVA AI Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

- Baidu online disk：[https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ](https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ) Extraction code：rryk

Extract the dataset and save it to an accessible path. The dataset directory is as follows:
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
## Quick validation using a pre-trained model
- Download the weights file (`*.pth`) from Releases to the `. /saved_models` directory.
  - [xvitstr_tiny_exp1.pth](https://github.com/lexiaoyuan/XViTSTR/releases/download/V1.0.0/xvitstr_tiny_exp1.pth)
  - [xvitstr_tiny_exp2.pth](https://github.com/lexiaoyuan/XViTSTR/releases/download/V1.0.0/xvitstr_tiny_exp2.pth)
  - [xvitstr_tiny_exp3.pth](https://github.com/lexiaoyuan/XViTSTR/releases/download/V1.0.0/xvitstr_tiny_exp3.pth)
  - [xvitstr_tiny_exp4.pth](https://github.com/lexiaoyuan/XViTSTR/releases/download/V1.0.0/xvitstr_tiny_exp4.pth)
  - [xvitstr_small_exp1.pth](https://github.com/lexiaoyuan/XViTSTR/releases/download/V1.0.0/xvitstr_small_exp1.pth)
  - [xvitstr_base_exp1.pth]()（Baidu online disk：https://pan.baidu.com/s/12ja5hyim3rt7laTsvwgaSg?pwd=v3sz 
    Extraction code：v3sz）
- Benchmarks

```bash
export CUDA_VISIBLE_DEVICES=0
python3 test.py --eval_data="data_lmdb_release/evaluation" --benchmark_all_eval --sensitive --data_filtering_off --saved_model="./saved_models/xvitstr_tiny_exp4.pth"
```
- One image
```bash
export CUDA_VISIBLE_DEVICES=0
python3 test.py --saved_model="./saved_models/xvitstr_tiny_exp4.pth" --img_path="demo.jpg" --eval_data="" --sensitive --data_filtering_off
```

- Calculate infer time
```bash
export CUDA_VISIBLE_DEVICES=0
python3 test.py --saved_model="./saved_models/xvitstr_tiny_exp4.pth" --img_path="./demo_image/" --eval_data="" --sensitive --data_filtering_off --calculate_infer_time

- Calculate FLOPs
​```bash
export CUDA_VISIBLE_DEVICES=0
python3 test.py --eval_data="" --sensitive --data_filtering_off --flops

## Train

- XViTSTR Tiny+ *Foc* + *Loc* in the default training paper:
​```bash
RANDOM=$$
export CUDA_VISIBLE_DEVICES=0
python3 train.py --train_data="data_lmdb_release/training" --valid_data="data_lmdb_release/validation" --manualSeed=$RANDOM --sensitive --adam --lr=0.001 --scheduler --exp="Name of experiment"
```
- To train other models, refer to the source code and comments.The configurable parameters are in [utils.py](./utils.py).

## Visualization of attention

> reference：[https://github.com/jacobgil/vit-explain](https://github.com/jacobgil/vit-explain)

```bash
export CUDA_VISIBLE_DEVICES=0
python3 vit_explain.py --saved_model="./saved_models/xvitstr_tiny_exp4.pth" --img_path="demo.jpg" --head_fusion="max" --use_cuda
```

## Reference
- [Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
- [ViTSTR](https://github.com/roatienza/deep-text-recognition-benchmark)
- [vit-explain](https://github.com/jacobgil/vit-explain)
