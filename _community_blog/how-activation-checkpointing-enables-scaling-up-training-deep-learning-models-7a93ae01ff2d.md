---
title: 'How Activation Checkpointing enables scaling up training deep learning models'
author: PyTorch
ext_url: https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d
date: Nov 9, 2023
---

Activation checkpointing is a technique used for reducing the memory footprint at the cost of more compute. It utilizes the simple observation that we can avoid saving intermediate tensors necessary for backward computation if we just recompute them on demand instead.