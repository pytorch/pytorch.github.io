---
layout: blog_detail
title: 'Introducing native PyTorch automatic mixed precision for faster training on NVIDIA GPUs'
author: Mengdi Huang, Chetan Tekur, Michael Carilli
---

Most deep learning frameworks, including PyTorch, train with 32-bit floating point (FP32) arithmetic by default. However this is not essential to achieve full accuracy for many deep learning models. In 2017, NVIDIA researchers developed a methodology for [mixed-precision training](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/), which combined [single-precision](https://blogs.nvidia.com/blog/2019/11/15/whats-the-difference-between-single-double-multi-and-mixed-precision-computing/) (FP32) with half-precision (e.g. FP16) format when training a network, and achieved the same accuracy as FP32 training using the same hyperparameters, with additional performance benefits on NVIDIA GPUs:

* Shorter training time;
* Lower memory requirements, enabling larger batch sizes, larger models, or larger inputs.

In order to streamline the user experience of training in mixed precision for researchers and practitioners, NVIDIA developed [Apex](https://developer.nvidia.com/blog/apex-pytorch-easy-mixed-precision-training/) in 2018, which is a lightweight PyTorch extension with [Automatic Mixed Precision](https://developer.nvidia.com/automatic-mixed-precision) (AMP) feature. This feature enables automatic conversion of certain GPU operations from FP32 precision to mixed precision, thus improving performance while maintaining accuracy.

For the PyTorch 1.6 release, developers at NVIDIA and Facebook moved mixed precision functionality into PyTorch core as the AMP package, [torch.cuda.amp](https://pytorch.org/docs/stable/amp.html). `torch.cuda.amp` is more flexible and intuitive compared to `apex.amp`. Some of `apex.amp`'s known pain points that `torch.cuda.amp` has been able to fix:

* Guaranteed PyTorch version compatibility, because it's part of PyTorch
* No need to build extensions
* Windows support
* Bitwise accurate [saving/restoring](https://pytorch.org/docs/master/amp.html#torch.cuda.amp.GradScaler.load_state_dict) of checkpoints
* [DataParallel](https://pytorch.org/docs/master/notes/amp_examples.html#dataparallel-in-a-single-process) and intra-process model parallelism (although we still recommend [torch.nn.DistributedDataParallel](https://pytorch.org/docs/master/notes/amp_examples.html#distributeddataparallel-one-gpu-per-process) with one GPU per process as the most performant approach)
* [Gradient penalty](https://pytorch.org/docs/master/notes/amp_examples.html#gradient-penalty) (double backward)
* torch.cuda.amp.autocast() has no effect outside regions where it's enabled, so it should serve cases that formerly struggled with multiple calls to [apex.amp.initialize()](https://github.com/NVIDIA/apex/issues/439) (including [cross-validation)](https://github.com/NVIDIA/apex/issues/392#issuecomment-610038073) without difficulty. Multiple convergence runs in the same script should each use a fresh [GradScaler instance](https://github.com/NVIDIA/apex/issues/439#issuecomment-610028282), but GradScalers are lightweight and self-contained so that's not a problem.
* Sparse gradient support

With AMP being added to PyTorch core, we have started the process of deprecating `apex.amp.` We have moved `apex.amp` to maintenance mode and will support customers using `apex.amp.` However, we highly encourage `apex.amp` customers to transition to using `torch.cuda.amp` from PyTorch Core.

# Example Walkthrough
Please see official docs for usage:
* [https://pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html )
* [https://pytorch.org/docs/stable/notes/amp_examples.html](https://pytorch.org/docs/stable/notes/amp_examples.html)

Example:

```python
import torch
# Creates once at the beginning of training
scaler = torch.cuda.amp.GradScaler()

for data, label in data_iter:
   optimizer.zero_grad()
   # Casts operations to mixed precision
   with torch.cuda.amp.autocast():
      loss = model(data)

   # Scales the loss, and calls backward()
   # to create scaled gradients
   scaler.scale(loss).backward()

   # Unscales gradients and calls
   # or skips optimizer.step()
   scaler.step(optimizer)

   # Updates the scale for next iteration
   scaler.update()
```

# Performance Benchmarks
In this section, we discuss the accuracy and performance of mixed precision training with AMP on the latest NVIDIA GPU A100 and also previous generation V100 GPU. The mixed precision performance is compared to FP32 performance, when running Deep Learning workloads in the [NVIDIA pytorch:20.06-py3 container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch?ncid=partn-52193#cid=ngc01_partn_en-us) from NGC.

## Accuracy: AMP (FP16), FP32
The advantage of using AMP for Deep Learning training is that the models converge to the similar final accuracy while providing improved training performance. To illustrate this point, for [Resnet 50 v1.5 training](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5#training-accuracy-nvidia-dgx-a100-8x-a100-40gb), we see the following accuracy results where higher is better. Please note that the below accuracy numbers are sample numbers that are subject to run to run variance of up to 0.4%. Accuracy numbers for other models including BERT, Transformer, ResNeXt-101, Mask-RCNN, DLRM can be found at  [NVIDIA Deep Learning Examples Github](https://github.com/NVIDIA/DeepLearningExamples).

Training accuracy: NVIDIA DGX A100 (8x A100 40GB)

<table width="460" border="0" cellspacing="5" cellpadding="5">
  <tbody>
    <tr>
      <td><strong>&nbsp;epochs</strong></td>
      <td><strong>&nbsp;Mixed Precision Top 1(%)</strong></td>
      <td>&nbsp;<strong>TF32 Top1(%)</strong></td>
    </tr>
    <tr>
      <td>&nbsp;90</td>
      <td>&nbsp;76.93</td>
      <td>&nbsp;76.85</td>
    </tr>
  </tbody>
</table>

Training accuracy: NVIDIA DGX-1 (8x V100 16GB)

 <table width="460" border="0" cellspacing="5" cellpadding="5">
  <tbody>
    <tr>
     <td><strong>&nbsp;epochs</strong></td>
      <td><strong>&nbsp;Mixed Precision Top 1(%)</strong></td>
      <td>&nbsp;<strong>FP32 Top1(%)</strong></td>
    </tr>
    <tr>
      <td>50</td>
      <td>76.25</td>
      <td>76.26</td>
    </tr>
    <tr>
      <td>90</td>
      <td>77.09</td>
      <td>77.01</td>
    </tr>
	  <tr>
      <td>250</td>
      <td>78.42</td>
      <td>78.30</td>
    </tr>
  </tbody>
</table>

## Speedup Performance:

### FP16 on NVIDIA V100 vs. FP32 on V100
AMP with FP16 is the most performant option for DL training on the V100. In Table 1, we can observe that for various models, AMP on V100 provides a speedup of 1.5x to 5.5x over FP32 on V100 while converging to the same final accuracy.

<div class="text-center">
  <img src="{{ site.baseurl }}/assets/images/nvidiafp32onv100.jpg" width="100%">
</div>
*Figure 2. Performance of mixed precision training on NVIDIA 8xV100 vs. FP32 training on 8xV100 GPU. Bars represent the speedup factor of V100 AMP over V100 FP32. The higher the better.*

## FP16 on NVIDIA A100 vs. FP16 on V100

AMP with FP16 remains the most performant option for DL training on the A100. In Figure 3, we can observe that for various models, AMP on A100 provides a speedup of 1.3x to 2.5x over AMP on V100 while converging to the same final accuracy.

<div class="text-center">
  <img src="{{ site.baseurl }}/assets/images/nvidiafp16onv100.png" width="100%">
</div>
*Figure 3. Performance of mixed precision training on NVIDIA 8xA100 vs. 8xV100 GPU. Bars represent the speedup factor of A100 over V100. The higher the better.*

# Call to action
AMP provides a healthy speedup for Deep Learning training workloads on Nvidia Tensor Core GPUs, especially on the latest Ampere generation A100 GPUs.  You can start experimenting with AMP enabled models and model scripts for A100, V100, T4 and other GPUs available at NVIDIA deep learning [examples](https://github.com/NVIDIA/DeepLearningExamples). NVIDIA PyTorch with native AMP support is available from the [PyTorch NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch?ncid=partn-52193#cid=ngc01_partn_en-us) version 20.06. We highly encourage existing `apex.amp` customers to transition to using `torch.cuda.amp` from PyTorch Core available in the latest [PyTorch 1.6 release](https://pytorch.org/blog/pytorch-1.6-released/).
