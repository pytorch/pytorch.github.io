---
layout: blog_detail
title: "Scaling PyTorch models on Cloud TPUs with FSDP"
author: Ronghang Hu, Vaibhav Singh, Jack Cao, Milad Mohammadi, Yeounoh Chung, Shauheen Zahirazami, Ross Girshick
featured-img: "/assets/images/scaling-pytorch-models-on-cloud-tpus-with-fsdp.jpg"
---

## Introduction

The research community has witnessed a lot of successes with large models across NLP, computer vision, and other domains in recent years. Many of these successes were enabled by Cloud TPUs -- which are powerful hardware for distributed training. To support TPUs in PyTorch, the PyTorch/XLA library provides a backend for XLA devices (most notably TPUs) and lays the groundwork for scaling large PyTorch models on TPUs.

However, most existing modeling scaling tools in the PyTorch ecosystem assume GPU (or CPU) devices, often depend on specific features in CUDA, and do not work directly on TPUs. The lack of scaling tools makes it challenging to build large models that cannot fit into the memory of a single TPU chip.

To support model scaling on TPUs, we implemented the widely-adopted [Fully Sharded Data Parallel (FSDP)](https://engineering.fb.com/2021/07/15/open-source/fsdp/) algorithm for XLA devices as part of the PyTorch/XLA 1.12 release. We provide an FSDP interface with a similar high-level design to the CUDA-based PyTorch FSDP class while also handling several restrictions in XLA (see Design Notes below for more details). This FSDP interface allowed us to easily build models with e.g. 10B+ parameters on TPUs and has enabled many research explorations.

## Using Fully Sharded Data Parallel (FSDP) in PyTorch/XLA

We provide a wrapper class `XlaFullyShardedDataParallel` over a given PyTorch model to shard its parameters across data-parallel workers. An example usage is as follows:

```python
import torch
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP

model = FSDP(my_module)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
output = model(x, y)
loss = output.sum()
loss.backward()
optim.step()
```

Wrapping an `nn.Module` instance with `XlaFullyShardedDataParallel` enables the [ZeRO-2](https://arxiv.org/abs/1910.02054) algorithm on it, where its gradients and the optimizer states are sharded for the entire training process. During its forward and backward passes, the full parameters of the wrapped module are first reconstructed from their corresponding shards for computation.

**Nested FSDP** wrapping can be used to further save memory. This allows the model to store only the full parameters of one individual layer at any given time. For nested FSDP, one should first wrap its individual submodules with an inner FSDP before wrapping the base model with an outer FSDP. This allows the model to store only the full parameters of one individual layer at any given time. And having an outer wrapper ensures to handle any leftover parameters, corresponding to the [ZeRO-3](https://arxiv.org/abs/1910.02054) algorithm. Nested FSDP wrapping can be applied at any depth of submodules and there can be more than 2 layers of nesting.

**Model checkpoint saving and loading** for models and optimizers can be done like before by saving and loading their `.state_dict()`. Meanwhile, each training process should save its own checkpoint file of the sharded model parameters and optimizer states, and load the checkpoint file for the corresponding rank when resuming (regardless of ZeRO-2 or ZeRO-3, i.e. nested wrapping or not). A command line tool and a Python interface are provided to consolidate the sharded model checkpoint files together into a full/unshareded model checkpoint file.

**Gradient checkpointing** (also referred to as "activation checkpointing" or "rematerialization") is another common technique for model scaling and can be used in conjunction with FSDP. We provide `checkpoint_module`, a wrapper function over a given `nn.Module` instance for gradient checkpointing (based on `torch_xla.utils.checkpoint.checkpoint`).

The MNIST and ImageNet examples below provide illustrative usages of (plain or nested) FSDP, saving and consolidation of model checkpoints, as well as gradient checkpointing.

## Starting examples of FSDP in PyTorch/XLA

### Training MNIST and ImageNet with FSDP

MNIST and ImageNet classification can often be used as starting points to build more complicated deep learning models. We provide the following FSDP examples on these two datasets:

- MNIST: [test/test_train_mp_mnist_fsdp_with_ckpt.py](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist_fsdp_with_ckpt.py) (it also illustrates checkpoint saving and consolidation)
- ImageNet: [test/test_train_mp_imagenet_fsdp.py](https://github.com/pytorch/xla/blob/master/test/test_train_mp_imagenet_fsdp.py)

A comparison of them with the vanilla data-parallel examples of [MNIST](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist.py) and [ImageNet](https://github.com/pytorch/xla/blob/master/test/test_train_mp_imagenet.py) illustrates how to adapt a training script to use FSDP. A major distinction to keep in mind is that when stepping the optimizer on an FSDP-wrapped model, one should directly call `optimizer.step()` instead of `xm.optimizer_step(optimizer)`. The latter reduces the gradients across ranks, which is not what we need in FSDP, where the gradients are already reduced and sharded (from a reduce-scatter op in its backward pass).

#### Installation

FSDP is available from the PyTorch/XLA 1.12 and newer nightly releases. Please refer to [https://github.com/pytorch/xla#-available-images-and-wheels](https://github.com/pytorch/xla#-available-images-and-wheels) for a guide on installation as well as Cloud TPU allocation. Then clone PyTorch/XLA repo on a TPU VM as follows

```python
mkdir -p ~/pytorch && cd ~/pytorch
git clone --recursive https://github.com/pytorch/xla.git
cd ~/
```

#### Train MNIST on v3-8 TPU

It gets around 98.9 accuracy for 2 epochs:

```python
python3 ~/pytorch/xla/test/test_train_mp_mnist_fsdp_with_ckpt.py \
  --batch_size 16 --drop_last --num_epochs 2 \
  --use_nested_fsdp
```

The script above automatically tests consolidation of the sharded model checkpoints at the end. You can also manually consolidate the sharded checkpoint files via

```python
python3 -m torch_xla.distributed.fsdp.consolidate_sharded_ckpts \
  --ckpt_prefix /tmp/mnist-fsdp/final_ckpt \
  --ckpt_suffix "_rank-*-of-*.pth"
```

#### Train ImageNet with ResNet-50 on v3-8 TPU

It gets around 75.9 accuracy for 100 epochs, same as what one would get without using FSDP; download and preprocess the [ImageNet-1k](https://github.com/pytorch/examples/tree/master/imagenet#requirements) dataset to `/datasets/imagenet-1k`:

```python
python3 ~/pytorch/xla/test/test_train_mp_imagenet_fsdp.py \
  --datadir /datasets/imagenet-1k --drop_last \
  --model resnet50 --test_set_batch_size 64 --eval_interval 10 \
  --lr 0.4 --batch_size 128 --num_warmup_epochs 5 \
  --lr_scheduler_divide_every_n_epochs 30 --lr_scheduler_divisor 10 \
  --num_epochs 100 \
  --use_nested_fsdp
```

You can also explore other options in these two examples, such as `--use_gradient_checkpointing` to apply gradient checkpointing (i.e. activation checkpointing) on the ResNet blocks, or `--compute_dtype bfloat16` to perform forward and backward passes in bfloat16 precision.

### Examples on large-scale models

When building large models on TPUs, we often need to be aware of the memory constraints (e.g. 16 GB per core in TPU v3 and 32 GB per chip in TPU v4). For large models that cannot fit into a single TPU memory or the host CPU memory, one should use nested FSDP to implement the ZeRO-3 algorithm interleave submodule construction with inner FSDP wrapping, so that the full model never needs to be stored in memory during construction.

We illustrate these cases in [https://github.com/ronghanghu/ptxla_scaling_examples](https://github.com/ronghanghu/ptxla_scaling_examples), which provides examples of training a Vision Transformer (ViT) model with 10B+ parameters on a TPU v3 pod (with 128 cores) as well as other cases.

### Design Notes

One might wonder why we need to develop a separate FSDP class in PyTorch/XLA instead of directly reusing [PyTorch's FSDP class](https://pytorch.org/docs/stable/fsdp.html) or extending it to the XLA backend. The main motivation behind a separate FSDP class in PyTorch/XLA is that the native PyTorch's FSDP class heavily relies on CUDA features that are not supported by XLA devices, while XLA also has several unique characteristics that need special handling. These distinctions require a different implementation of FSDP that would be much easier to build in a separate class.

#### Changes in API calls
One prominent distinction is that the native PyTorch FSDP is built upon separate CUDA streams for asynchronous execution in eager mode, while PyTorch/XLA runs in lazy mode and also does not support streams. In addition, TPU requires that all devices homogeneously run the same program. As a result, in the PyTorch/XLA FSDP implementation, CUDA calls and per-process heterogeneity need to be replaced by XLA APIs and alternative homogeneous implementations.

#### Tensor Storage Handling

Another prominent distinction is how to free a tensor's storage, which is much harder in XLA than in CUDA. To implement ZeRO-3, one needs to free the storage of full parameters after a module's forward pass, so that the next module can reuse this memory buffer for subsequent computation. PyTorch's FSPD accomplishes this on CUDA by freeing the actual storage of a parameter `p` via `p.data.storage().resize_(0)`. However, XLA tensors do not have this `.storage()` handle given that the XLA HLO IRs are completely functional and do not provide any ops to deallocate a tensor or resize its storage. Below the PyTorch interface, only the XLA compiler can decide when to free a TPU device memory corresponding to an XLA tensor, and a prerequisite is that the memory can only be released when the tensor object gets deallocated in Python -- which cannot happen in FSDP because these parameter tensors are referenced as module attributes and also saved by PyTorch autograd for the backward pass.

Our solution to this issue is to split a tensor's value properties from its autograd Variable properties, and to free a `nn.Parameter` tensor by setting its `.data` attribute to a dummy scalar of size 1. This way the actual data tensor for the full parameter gets dereferenced in Python so that XLA can recycle its memory for other computation, while autograd can still trace the base `nn.Parameter` as a weak reference to the parameter data. To get this to work, one also needs to handle views over the parameters as views in PyTorch also hold references to its actual data (this required fixing a shape-related issue with views in PyTorch/XLA).

#### Working with XLA compiler

The solution above should be enough to free full parameters if the XLA compiler faithfully preserves the operations and their execution order in our PyTorch program. But there is another problem -- XLA attempts to optimize the program to speed up its execution by applying common subexpression elimination (CSE) to the HLO IRs. In a naive implementation of FSDP, the XLA compiler typically eliminates the 2nd all-gather in the backward pass to reconstruct the full parameters when it sees that it is a repeated computation from the forward pass, and directly holds and reuses the full parameters we want to free up after the forward pass. To guard against this undesired compiler behavior, we introduced the [optimization barrier op](https://www.tensorflow.org/xla/operation_semantics#optimizationbarrier) into PyTorch/XLA and used it to stop eliminating the 2nd all-gather. This optimization barrier is also applied to a similar case of gradient checkpointing to prevent CSE between forward and backward passes that could eliminate the rematerialization.

In the future, if the distinctions between CUDA and XLA become not as prominent as mentioned above, it could be worth considering a merge of the PyTorch/XLA FSDP with the native PyTorch FSDP to have a unified interface.

## Acknowledgments

Thanks to Junmin Hao from AWS for reviewing the PyTorch/XLA FSDP pull request. Thanks to Brian Hirsh from the Meta PyTorch team for support on the PyTorch core issues. Thanks to Isaack Karanja, Will Cromar, and Blake Hechtman from Google for support on GCP, XLA, and TPU issues.

Thanks to Piotr Dollar, Wan-Yen Lo, Alex Berg, Ryan Mark, Kaiming He, Xinlei Chen, Saining Xie, Shoubhik Debnath, Min Xu, and Vaibhav Aggarwal from Meta FAIR for various TPU-related discussions.