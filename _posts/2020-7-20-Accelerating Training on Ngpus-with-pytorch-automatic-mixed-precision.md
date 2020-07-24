---
layout: blog_detail
title: 'Accelerating Training on NVIDIA GPUs with PyTorch Automatic Mixed Precision'
author: Michael Carilli, Mengdi Huang, Chetan Tekur
---

Most deep learning frameworks, including PyTorch, train with 32-bit floating point (FP32) arithmetic by default. However this is not essential to achieve full accuracy for many deep learning models. In 2017, NVIDIA researchers developed a methodology for [mixed-precision training](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/), which combined [single-precision](https://blogs.nvidia.com/blog/2019/11/15/whats-the-difference-between-single-double-multi-and-mixed-precision-computing/) (FP32) with half-precision (e.g. FP16) format when training a network, and achieved the same accuracy as FP32 training using the same hyperparameters, with additional performance benefits on NVIDIA GPUs:

* Shorter training time;
* Lower memory requirements, enabling larger batch sizes, larger models, or larger inputs.

In order to streamline the user experience of training in mixed precision for researchers and practitioners, NVIDIA developed [Apex](https://developer.nvidia.com/blog/apex-pytorch-easy-mixed-precision-training/) in 2018, which is a lightweight PyTorch extension with [Automatic Mixed Precision](https://developer.nvidia.com/automatic-mixed-precision) (AMP) feature. This feature enables automatic conversion of certain GPU operations from FP32 precision to mixed precision, thus improving performance while maintaining accuracy. 

For the PyTorch 1.6 release, developers at NVIDIA and Facebook moved mixed precision functionality into PyTorch core as the AMP package, `torch.cuda.amp` < Missing link>. `torch.cuda.amp` is more flexible and intuitive compared to `apex.amp`. Some of `apex.amp`'s known pain points that torch.cuda.amp has been able to fix:

* Guaranteed PyTorch version compatibility, because it's part of PyTorch
* No need to build extensions
* Windows support
* Bitwise accurate [saving/restoring](https://pytorch.org/docs/master/amp.html#torch.cuda.amp.GradScaler.load_state_dict) of checkpoints
* [DataParallel](https://pytorch.org/docs/master/notes/amp_examples.html#dataparallel-in-a-single-process) and intra-process model parallelism (although we still recommend [torch.nn.DistributedDataParallel](https://pytorch.org/docs/master/notes/amp_examples.html#distributeddataparallel-one-gpu-per-process) with one GPU per process as the most performant approach)
* [Gradient penalty](https://pytorch.org/docs/master/notes/amp_examples.html#gradient-penalty) (double backward)
* torch.cuda.amp.autocast() has no effect outside regions where it's enabled, so it should serve cases that formerly struggled with multiple calls to [apex.amp.initialize()](https://github.com/NVIDIA/apex/issues/439) (including [cross-validation)](https://github.com/NVIDIA/apex/issues/392#issuecomment-610038073) without difficulty. Multiple convergence runs in the same script should each use a fresh [GradScaler instance](https://github.com/NVIDIA/apex/issues/439#issuecomment-610028282), but GradScalers are lightweight and self-contained so that's not a problem.
* Sparse gradient support
 

