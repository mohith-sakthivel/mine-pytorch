# Mutual Information Neural Estimation

This repository contains an implementation of the Information Bottleneck (IB) using Mutual Information Neural Estimation (MINE). [Belghazi et al., 2018]

A standard baseline MLP (as in Deep VIB paper [Alemi et al., 2017]) has been used for comparison.

## To Check
* on_fit gets called on Trainer.test

## Changes
* Remove bias reducer

## References
1. M I Belghazi, A Baratin, S Rajeswar, S Ozair, Y Bengio, A Courville, R D Hjelm - MINE: Mutual Information Neural Estimation, ICML, 2018. ([paper](https://arxiv.org/abs/1801.04062))

2. A A Alemi, I Fischer, J V Dillon, K Murphy - Deep Variational Information Bottleneck, ICLR, 2017. ([paper](https://arxiv.org/abs/1612.00410))