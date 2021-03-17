# Mutual Information Neural Estimation

This repository contains a pytorch implementation of the Information Bottleneck (IB) using Mutual Information Neural Estimation (MINE). [[Belghazi et al., 2018]](#references)

A standard baseline MLP (as described in Deep VIB paper [[Alemi et al., 2017]](#references)) has been used for comparison.

## Setup

```
git clone https://github.com/mohith-sakthivel/mine-pytorch.git mine
cd mine

conda create -n mine -f environment.yml
conda activate mine
```

## Run
* To run the baseline model with default parameters
    ```
    python3 -m mine.mine --deter
    ```

    The baseline model is a standard MLP with 3 hidden layers and ReLU non-linearity. During training, an exponential weighted average of the parameters is maintained and these averaged parameters are used at test time.


* To run MINE+IB model
    ```
    python3 -m mine.mine --mine
    ```



## References
1. M I Belghazi, A Baratin, S Rajeswar, S Ozair, Y Bengio, A Courville, R D Hjelm - MINE: Mutual Information Neural Estimation, ICML, 2018. ([paper](https://arxiv.org/abs/1801.04062))

2. A A Alemi, I Fischer, J V Dillon, K Murphy - Deep Variational Information Bottleneck, ICLR, 2017. ([paper](https://arxiv.org/abs/1612.00410))