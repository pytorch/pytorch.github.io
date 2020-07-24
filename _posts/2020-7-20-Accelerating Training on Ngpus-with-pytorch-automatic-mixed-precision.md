---
layout: blog_detail
title: 'Accelerating Training on NVIDIA GPUs with PyTorch Automatic Mixed Precision'
author: Michael Carilli, Mengdi Huang, Chetan Tekur
---

Most deep learning frameworks, including PyTorch, train with 32-bit floating point (FP32) arithmetic by default. However this is not essential to achieve full accuracy for many deep learning models. In 2017, NVIDIA researchers developed a methodology for [mixed-precision training](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/), which combined [single-precision](https://blogs.nvidia.com/blog/2019/11/15/whats-the-difference-between-single-double-multi-and-mixed-precision-computing/) (FP32) with half-precision (e.g. FP16) format when training a network, and achieved the same accuracy as FP32 training using the same hyperparameters, with additional performance benefits on NVIDIA GPUs:

* Shorter training time;
* Lower memory requirements, enabling larger batch sizes, larger models, or larger inputs.
