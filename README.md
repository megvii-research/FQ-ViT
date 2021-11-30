# FQ-ViT [[arXiv]](http://arxiv.org/abs/2111.13824)


This repo contains the official implementation of **["FQ-ViT: Fully Quantized Vision Transformer without Retraining"](http://arxiv.org/abs/2111.13824).**


## Table of Contents
- [Introduction](#introduction)
  - [Layernorm quantized with Powers-of-Two Scale (PTS)](#layernorm-quantized-with-powers-of-two-scale-pts)
  - [Softmax quantized with Log-Int-Softmax (LIS)](#softmax-quantized-with-log-int-softmax-lis)
- [Getting Started](#getting-started)
  - [Install](#install)
  - [Data preparation](#data-preparation)
  - [Run](#run)
- [Results on ImageNet](#results-on-imagenet)
- [Citation](#citation)


## Introduction

Transformer-based architectures have achieved competitive performance in various CV tasks. Compared to the CNNs, Transformers usually have more parameters and higher computational costs, presenting a challenge when deployed to resource-constrained hardware devices.

Most existing quantization approaches are designed and tested on CNNs and lack proper handling of Transformer-specific modules. Previous work found there would be significant accuracy degradation when quantizing LayerNorm and Softmax of Transformer-based architectures. As a result, they left **LayerNorm and Softmax unquantized with floating-point numbers**. We revisit these two exclusive modules of the Vision Transformers and discover the reasons for degradation. In this work, we propose the **FQ-ViT**, the first fully quantized Vision Transformer, which contains two specific modules: Powers-of-Two Scale (PTS) and Log-Int-Softmax (LIS). 

### Layernorm quantized with Powers-of-Two Scale (PTS)

  These two figures below show that there exists serious inter-channel variation in Vision Transformers than CNNs, which leads to unacceptable quantization errors with layer-wise quantization. 

  <div align=center>
  <img src="./figures/inter-channel_variation.png" width="850px" />
  </div>
  
  Taking the advantages of both layer-wise and channel-wise quantization, we propose PTS for LayerNorm's quantization. The core idea of PTS is to equip different channels with different Powers-of-Two Scale factors, rather than different quantization scales. 

### Softmax quantized with Log-Int-Softmax (LIS)

  The storage and computation of attention map is known as a bottleneck for transformer structures, so we want to quantize it to extreme lower bit-width (e.g. 4-bit). However, if directly implementing 4-bit uniform quantization, there will be severe accuracy degeneration. We observe a distribution centering at a fairly small value of the output of Softmax, while only few outliers have larger values close to 1. Based on the following visualization, Log2 preserves more quantization bins than uniform for the small value interval with dense distribution.
 
  <div align=center>
  <img src="./figures/distribution.png" width="400px" />
  </div>

  Combining Log2 quantization with i-exp, which is a polynomial approximation of exponential function presented by [I-BERT](https://arxiv.org/abs/2101.01321), we propose LIS, an integer-only, faster, low consuming Softmax.

  The whole process is visualized as follow.

  <div align=center>
  <img src="./figures/log-int-softmax.png" width="400px" />
  </div>

## Getting Started

### Install

- Clone this repo.

```bash
git clone https://github.com/linyang-zhh/FQ-ViT.git
cd FQ-ViT
```

- Create a conda virtual environment and activate it.

```bash
conda create -n fq-vit python=3.7 -y
conda activate fq-vit
```

- Install PyTorch and torchvision. e.g.,

```bash
conda install pytorch=1.7.1 torchvision cudatoolkit=10.1 -c pytorch
```

### Data preparation

You should download the standard ImageNet Dataset.

```
├── imagenet
│   ├── train
|
│   ├── val
```


### Run

Example: Evaluate quantized DeiT-S with MinMax quantizer and our proposed PTS and LIS

```bash
python test_quant.py deit_small <YOUR_DATA_DIR> --quant --pts --lis --quant-method minmax

```

- `deit_small`: model architecture, which can be replaced by `deit_tiny`, `deit_base`, `vit_base`, `vit_large`, `swin_tiny`, `swin_small` and `swin_base`.

- `--quant`: whether to quantize the model.

- `--pts`: whether to use **Power-of-Two Scale Integer Layernorm**. 

- `--lis`: whether to use **Log-Integer-Softmax**.

- `--quant-method`: quantization methods of activations, which can be chosen from `minmax`, `ema`, `percentile` and `omse`.

## Results on ImageNet

This paper employs several current post-training quantization strategies together with our methods, including MinMax, EMA , Percentile and OMSE.

- MinMax uses the minimum and maximum values of the total data as the clipping values; 

- [EMA](https://arxiv.org/abs/1712.05877) is based on MinMax and uses an average moving mechanism to smooth the minimum and maximum values of different mini-batch;

- [Percentile](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Fully_Quantized_Network_for_Object_Detection_CVPR_2019_paper.pdf) assumes that the distribution of values conforms to a normal distribution and uses the percentile to clip. In this paper, we use the 1e-5 percentile because the 1e-4 commonly used in CNNs has poor performance in Vision Transformers. 

- [OMSE](https://arxiv.org/abs/1902.06822) determines the clipping values by minimizing the quantization error. 


The following results are evaluated on ImageNet.

|         Method         | W/A/Attn Bits |   ViT-B   |   ViT-L   |  DeiT-T   |  DeiT-S   |  DeiT-B   |  Swin-T   |  Swin-S   |  Swin-B   |
| :--------------------: | :-----------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|     Full Precision     | 32/32/32  |   84.53   |   85.81   |   72.21   |   79.85   |   81.85   |   81.35   |   83.20   | 83.60 |
|         MinMax         |   8/8/8   |   23.64   |   3.37    |   70.94   |   75.05   |   78.02   |   64.38   |   74.37   | 25.58 |
|     MinMax w/ PTS      |   8/8/8   |   83.31   |   85.03   |   71.61   |   79.17   |   81.20   |   80.51   |   82.71   | 82.97 |
|   MinMax w/ PTS, LIS   | 8/8/**4** | **82.68** |   84.89   |   71.07   |   78.40   |   80.85   |   80.04   |   82.47   | 82.38 |
|          EMA           |   8/8/8   |   30.30   |   3.53    |   71.17   |   75.71   |   78.82   |   70.81   |   75.05   | 28.00 |
|       EMA w/ PTS       |   8/8/8   |   83.49   |   85.10   |   71.66   |   79.09   |   81.43   |   80.52   |   82.81   | 83.01 |
|    EMA w/ PTS, LIS     | 8/8/**4** |   82.57   |   85.08   |   70.91   | **78.53** | **80.90** |   80.02   |   82.56   | 82.43 |
|       Percentile       |   8/8/8   |   46.69   |   5.85    |   71.47   |   76.57   |   78.37   |   78.78   |   78.12   | 40.93 |
|   Percentile w/ PTS    |   8/8/8   |   80.86   |   85.24   |   71.74   |   78.99   |   80.30   |   80.80   |   82.85   | 83.10 |
| Percentile w/ PTS, LIS | 8/8/**4** |   80.22   | **85.17** | **71.23** |   78.30   |   80.02   | **80.46** | **82.67** | 82.79 |
|          OMSE          |   8/8/8   |   73.39   |   11.32   |   71.30   |   75.03   |   79.57   |   79.30   |   78.96   | 48.55 |
|      OMSE w/ PTS       |   8/8/8   |   82.73   |   85.27   |   71.64   |   78.96   |   81.25   |   80.64   |   82.87   | 83.07 |
|    OMSE w/ PTS, LIS    | 8/8/**4** |   82.37   |   85.16   |   70.87   |   78.42   | **80.90** |   80.41   |   82.57   | 82.45 |

## Citation

If you find this repo useful in your research, please consider citing the following paper:

```BibTex
@misc{
    lin2021fqvit,
    title={FQ-ViT: Fully Quantized Vision Transformer without Retraining}, 
    author={Yang Lin and Tianyu Zhang and Peiqin Sun and Zheng Li and Shuchang Zhou},
    year={2021},
    eprint={2111.13824},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
