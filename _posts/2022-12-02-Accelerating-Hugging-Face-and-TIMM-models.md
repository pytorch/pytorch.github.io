---
layout: blog_detail
title: "Accelerating Hugging Face and TIMM models with PyTorch 2.0"
author: Mark Saroufim
featured-img: "assets/images/pytorch-2.0-feature-img.png"
---

`torch.compile()` makes it easy to experiment with different compiler backends to make PyTorch code faster with a single line decorator `torch.compile()`. It works either directly over an nn.Module as a drop-in replacement for `torch.jit.script()` but without requiring you to make any source code changes. We expect this one line code change to provide you with between 30%-2x training time speedups on the vast majority of models that you’re already running.

```python

opt_module = torch.compile(module)

```

torch.compile supports arbitrary PyTorch code, control flow, mutation and comes with experimental support for dynamic shapes. We’re so excited about this development that we call it PyTorch 2.0.

What makes this announcement different for us is we’ve already benchmarked some of the most popular open source PyTorch models and gotten substantial speedups ranging from 30% to 2x [https://github.com/pytorch/torchdynamo/issues/681](https://github.com/pytorch/torchdynamo/issues/681).

There are no tricks here, we’ve pip installed popular libraries like [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers), [https://github.com/huggingface/accelerate](https://github.com/huggingface/accelerate) and [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and then ran torch.compile() on them and that’s it.

It’s rare to get both performance and convenience, but this is why the core team finds PyTorch 2.0 so exciting. The Hugging Face team is also excited, in their words:

Ross Wightman the primary maintainer of TIMM: “PT 2.0 works out of the box with majority of timm models for inference and train workloads and no code changes”

Sylvain Gugger the primary maintainer of transformers and accelerate: "With just one line of code to add, PyTorch 2.0 gives a speedup between 1.5x and 2.x in training Transformers models. This is the most exciting thing since mixed precision training was introduced!"

This tutorial will show you exactly how to replicate those speedups so you can be as excited as to PyTorch 2.0 as we are.

## Requirements and Setup

For GPU (newer generation GPUs will see drastically better performance)

```
pip3 install numpy --pre torch --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117

```

For CPU

```
pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu

```

Optional: Verify Installation

```
git clone https://github.com/pytorch/pytorch
cd tools/dynamo
python verify_dynamo.py
```

Optional: Docker installation

We also provide all the required dependencies in the PyTorch nightly
binaries which you can download with

```
docker pull ghcr.io/pytorch/pytorch-nightly

```

And for ad hoc experiments just make sure that your container has access
to all your GPUs

```
docker run --gpus all -it ghcr.io/pytorch/pytorch-nightly:latest /bin/bash

```

## Getting started

### a toy exmaple

Let’s start with a simple example and make things more complicated step
by step. Please note that you’re likely to see more significant speedups the newer your GPU is.

```python
import torch
def fn(x, y):
    a = torch.sin(x).cuda()
    b = torch.sin(y).cuda()
    return a + b
new_fn = torch.compile(fn, backend="inductor")
input_tensor = torch.randn(10000).to(device="cuda:0")
a = new_fn(input_tensor, input_tensor)
```

This example won’t actually run faster but it’s educational.

example that features `torch.cos()` and `torch.sin()` which are examples of pointwise ops as in they operate element by element on a vector. A more famous pointwise op you might actually want to use would be something like `torch.relu()`.

Pointwise ops in eager mode are suboptimal because each one would need to read a tensor from memory, make some changes and then write back those changes.

The single most important optimization that PyTorch 2.0 does for you is fusion.

So back to our example we can turn 2 reads and 2 writes into 1 read and 1 write which is crucial especially for newer GPUs where the bottleneck is memory bandwidth (how quickly you can send data to a GPU) instead of compute (how quickly your GPU can crunch floating point operations)

The second most important optimization that PyTorch 2.0 does for you is CUDA graphs

CUDA graphs help eliminate the overhead from launching individual kernels from a python program.

torch.compile() supports many different backends but one that we’re particularly excited about is Inductor which generates Triton kernels [https://github.com/openai/triton](https://github.com/openai/triton) which are written in Python yet outperform the vast majority of handwritten CUDA kernels. Suppose our example above was called trig.py we can actually inspect the code generated triton kernels by running.

```
TORCH_COMPILE_DEBUG=1 python trig.py
```

```python

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sin(tmp0)
    tmp2 = tl.sin(tmp1)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)

```

And you can verify that fusing the two `sins` did actually occur because the two `sin` operations occur within a single Triton kernel and the temporary variables are held in registers with very fast access.

### a real model

As a next step let’s try a real model like resnet50 from the PyTorch hub.

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
opt_model = torch.compile(model, backend="inductor")
model(torch.randn(1,3,64,64))

```

If you actually run you may be surprised that the first run is slow and that’s because the model is being compiled. Subsequent runs will be faster so it's common practice to warm up your model before you start benchmarking it.

You may have noticed how we also passed in the name of a compiler explicitly here with “inductor” but it’s not the only available backend, you can run in a REPL `torch._dynamo.list_backends()` to see the full list of available backends. For fun you should try out `aot_cudagraphs` or `nvfuser`.

### Hugging Face models

Let’s do something a bit more interesting now, our community frequently
uses pretrained models from transformers [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers) or TIMM [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and one of our design goals for PyTorch 2.0 was that any new compiler stack needs to work out of the box with the vast majority of models people actually run.

So we’re going to directly download a pretrained model from the Hugging Face hub and optimize it

```python

import torch
from transformers import BertTokenizer, BertModel
# Copy pasted from here https://huggingface.co/bert-base-uncased
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to(device="cuda:0")
model = torch.compile(model) # This is the only line of code that we changed
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to(device="cuda:0")
output = model(**encoded_input)

```

If you remove the `to(device="cuda:0")` from the model and `encoded_input` then PyTorch 2.0 will generate C++ kernels that will be optimized for running on your CPU. You can inspect both Triton or C++ kernels for BERT, they’re obviously more complex than the trigonometry example we had above but you can similarly skim it and understand if you understand PyTorch.

The same code also works just fine if used with [https://github.com/huggingface/accelerate](https://github.com/huggingface/accelerate) and DDP

Similarly let’s try out a TIMM example

```python
import timm
import torch
model = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=2)
opt_model = torch.compile(model, backend="inductor")
opt_model(torch.randn(64,3,7,7))
```

Our goal with PyTorch was to build a breadth-first compiler that would speed up the vast majority of actual models people run in open source. The Hugging Face Hub ended up being an extremely valuable benchmarking tool for us, ensuring that any optimization we work on actually helps accelerate models people want to run.

So please try out PyTorch 2.0, enjoy the free perf and if you’re not seeing it then please open an issue and we will make sure your model is supported [https://github.com/pytorch/torchdynamo/issues](https://github.com/pytorch/torchdynamo/issues)

After all, we can’t claim we’re created a breadth-first unless YOUR models actually run faster.
