---
layout: blog_detail
title: 'Quantization in Practice'
author: Suraj Subramanian, Jerry Zhang
featured-img: ''
---

There are a few different ways to quantize your model with PyTorch. In this blog post, we'll take a look at how each technique looks like in practice. I will use a non-standard model that is not traceable, to paint an accurate picture of how much effort is really needed when quantizing your model.

<div class="text-center">
  <img src="/assets/images/quantization_gif.gif" width="60%">
</div>

## A quick introduction to quantization

> If someone asks you what time it is, you don't respond "10:14:34:430705", but you might say "a quarter past 10".

Quantization has roots in information compression; in deep networks it refers to reducing the numerical precision of its weights and/or activations. 

Overparameterized DNNs have more degrees of freedom and this makes them good candidates for information compression [[1](https://arxiv.org/pdf/2103.13630.pdf)]. When you quantize a model, two things generally happen - the model gets smaller and runs with better efficiency. Hardware vendors explicitly allow for faster processing of 8-bit data (than 32-bits). A smaller model has lower memory footprint and power consumption [[2](https://arxiv.org/pdf/1806.08342.pdf)], crucial for deployment at the edge.

At the heart of it all is a **mapping function**, a linear projection from floating-point to integer space: $Q(r) = round(r/S + Z)$

To reconvert to floating point space, the inverse function is given by $\tilde r = (Q(r) - Z) \cdot S$. $\tilde r \neq r$, and their difference constitutes the *quantization error*.

The scaling factor $S$ is simply the ratio of the input range to the output range: $S = \frac{\beta - \alpha}{\beta_q - \alpha_q}$
where [$\alpha, \beta$] is the clipping range of the input, i.e. the boundaries of permissible inputs. [$\alpha_q, \beta_q$] is the range in quantized output space that it is mapped to. For 8-bit quantization, the output range $\beta_q - \alpha_q <= (2^8 - 1) $.

The zero-point $Z$ acts as a bias to ensure that a 0 in the input space maps perfectly to a 0 in the quantized space. $Z = -(\frac{\alpha}{S} - \alpha_q)$ 

$S, Z$ can be calculated and used for quantizing an entire tensor ("per-tensor"), or individually for each channel ("per-channel").


### Calibration
The process of choosing the input range is known as **calibration**. The simplest technique (also the default in PyTorch) is to record the running mininmum and maximum values and assign them to $\alpha$ and $\beta$. In PyTorch, `Observer` modules ([docs](https://PyTorch.org/docs/stable/torch.quantization.html?highlight=observer#observers), [code](https://github.com/PyTorch/PyTorch/blob/748d9d24940cd17938df963456c90fa1a13f3932/torch/ao/quantization/observer.py#L88)) collect statistics on the input values and calculate the qparams $S, Z$. 

```python
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver
C, L = 3, 4
normal = torch.distributions.normal.Normal(0,1)
inputs = [normal.sample((C, L)), normal.sample((C, L))]
print(inputs)

# >>>>>
# [tensor([[-0.0590,  1.1674,  0.7119, -1.1270],
#          [-1.3974,  0.5077, -0.5601,  0.0683],
#          [-0.0929,  0.9473,  0.7159, -0.4574]]]),

# tensor([[-0.0236, -0.7599,  1.0290,  0.8914],
#          [-1.1727, -1.2556, -0.2271,  0.9568],
#          [-0.2500,  1.4579,  1.4707,  0.4043]])]

observers = [MinMaxObserver(), MovingAverageMinMaxObserver(), HistogramObserver()]
for obs in observers:
  for x in inputs: obs(x) 
  print(obs.__class__.__name__, obs.calculate_qparams())

# >>>>>
# MinMaxObserver (tensor([0.0112]), tensor([124], dtype=torch.int32))
# MovingAverageMinMaxObserver (tensor([0.0101]), tensor([139], dtype=torch.int32))
# HistogramObserver (tensor([0.0100]), tensor([106], dtype=torch.int32))
```

### Affine and Symmetric Quantization Schemes
Affine or asymmetric quantization schemes assign the input range to the min and max observed values. Affine schemes offer tighter clipping ranges and are useful for quantizing non-negative activations (you don't need the input range to contain negative values if your input tensors are never negative). The range is calculated as 
$\alpha = min(r), \beta = max(r)$. 

Symmetric quantization schemes center the input range around 0, eliminating the need to calculate a zero-point offset. The range is calculated as 
$-\alpha = \beta = max(|max(r)|,|min(r)|)$.

```python
for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
  obs = MovingAverageMinMaxObserver(qscheme=qscheme)
  for x in inputs: obs(x)
  print(f"Qscheme: {qscheme} | {obs.calculate_qparams()}")

# >>>>>
# Qscheme: torch.per_tensor_affine | (tensor([0.0101]), tensor([139], dtype=torch.int32))
# Qscheme: torch.per_tensor_symmetric | (tensor([0.0109]), tensor([128]))
```

### Per-Tensor and Per-Channel Quantization Schemes
Quantization parameters can be calculated for the layer's entire weight tensor as a whole, or separately for each channel

```python
from torch.quantization.observer import MovingAveragePerChannelMinMaxObserver
obs = MovingAveragePerChannelMinMaxObserver(ch_axis=0)  # calculate qparams for all `C` channels separately
for x in inputs: obs(x)
print(obs.calculate_qparams())

# >>>>>
# (tensor([0.0090, 0.0075, 0.0055]), tensor([125, 187,  82], dtype=torch.int32))
```
Per-channel quantization provides better accuracies in convolutional networks. Per-tensor performs poorly due to high variance in conv weights from batchnorm scaling. [[2](https://arxiv.org/pdf/1806.08342.pdf)]

### QConfig

The `QConfig` ([code](https://github.com/PyTorch/PyTorch/blob/d6b15bfcbdaff8eb73fa750ee47cef4ccee1cd92/torch/ao/quantization/qconfig.py#L165), [docs](https://pytorch.org/docs/stable/torch.quantization.html?highlight=qconfig#torch.quantization.QConfig)) NamedTuple stores the Observers and the quantization schemes used to quantize activations and weights.

Be sure to pass the Observer class (not the instance), or a callable that can return Observer instances. Use `with_args()` to override the default arguments.

```python
my_qconfig = torch.quantization.QConfig(
  activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine),
  weight=MovingAveragePerChannelMinMaxObserver.with_args(qscheme=torch.qint8)
)
# >>>>>
# QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.MovingAverageMinMaxObserver'>, qscheme=torch.per_tensor_affine){}, weight=functools.partial(<class 'torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver'>, qscheme=torch.qint8){})
```

### Backend
Currently, quantized operators run on x86 machines via the [FBGEMM backend](https://github.com/pytorch/FBGEMM), or use [QNNPACK](https://github.com/pytorch/QNNPACK) primitives on ARM machines. Backend support for server GPUs (via TensorRT and cuDNN) is coming soon. Learn more about extending quantization to custom backends: [RFC-0019](https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md).

```python
backend = 'fbgemm' if x86 else 'qnnpack'
qconfig = torch.quantization.get_default_qconfig(backend)  
torch.backends.quantized.engine = backend
```


## Techniques in PyTorch

PyTorch allows you a few different ways to quantize your model on the CPU, depending on
- if you prefer a flexible but manual, or a restricted automagic process (*Eager Mode* v/s *FX Graph Mode*)
- if $S, Z$ for quantizing activations (layer outputs) are precomputed for all inputs, or calculated afresh with each input (*static* v/s *dynamic*),
- if $S, Z$ are computed during, or after training (*quantization-aware training* v/s *post-training quantization*)

FX Graph Mode automatically fuses eligible modules, inserts Quant/DeQuant stubs, calibrates the model and throws out a quantized module - all in two method calls - but only for networks that are [symbolic traceable](https://PyTorch.org/docs/stable/fx.html#torch.fx.symbolic_trace). The examples below contain the calls using Eager Mode and FX Graph Mode for comparison.

In DNNs, eligible candidates for quantization are the FP32 weights (layer parameters) and activations (layer outputs). Quantizing weights reduces the model size. Quantized activations typically result in faster inference.

As an example, the 50-layer ResNet network has ~26 million weight parameters and computes ~16 million activations in the forward pass.

### Post-Training Dynamic/Weight-only Quantization 
Here the model's weights are pre-quantized; the activations are quantized on-the-fly ("dynamic") during inference. The simplest of all approaches, it has a one line API call in `torch.quantization.quantize_dynamic`. 

 **(+)** Can result in higher accuracies since the clipping range is exactly calibrated for each input [1].
 
 **(+)** Dynamic quantization is preferred for models like LSTMs and Transformers where writing/retrieving the model's weights from memory dominate bandwidths [4]. 
 
 **(-)** Calibrating and quantizing the activations at each layer during runtime can add to the compute overhead. 

```python
import torch
from torch import nn

# toy model
m = nn.Sequential(
  nn.Conv1d(2, 64, (8,)),
  nn.ReLU(),
  nn.Linear(16,10),
  nn.LSTM(10, 10))

m.eval()

## EAGER MODE
from torch.quantization import quantize_dynamic
model_quantized = quantize_dynamic(
    model=m, qconfig_spec={nn.LSTM, nn.Linear}, dtype=torch.qint8, inplace=False
)

## FX MODE
from torch.quantization import quantize_fx
qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}  # An empty key denotes the default applied to all modules
model_prepared = quantize_fx.prepare_fx(m, qconfig_dict)
model_quantized = quantize_fx.convert_fx(model_prepared)
```

### Post-Training Static Quantization
Model weights are pre-quantized; activations are pre-calibrated by observers using validation data and stay in quantized precision between operations. About 100 mini-batches of representative data are sufficient to calibrate the observers [2]. The examples below use random data in calibration for convenience - using that in your application will result in bad qparams.

**(+)** Static quantization has faster inference than dynamic quantization because it eliminates the float<->int conversion costs between layers. 

**(-)** Static quantized models may need regular re-calibration to stay robust against distribution-drift.


```python
# Static quantization of a model consists of the following steps:

#     Fuse modules
#     Insert Quant/DeQuant Stubs
#     Prepare the fused module (insert observers before and after)
#     Calibrate the prepared module (pass it representative data)
#     Convert the calibrated module (replace with quantized version)

import torch
from torch import nn

backend = "fbgemm"  # running on a x86 CPU. Use "qnnpack" if running on ARM.

m = nn.Sequential(
     nn.Conv1d(2,64,(8,)),
     nn.ReLU(),
     nn.Conv1d(64, 128, (1,)),
     nn.ReLU()
)

## EAGER MODE
"""Fuse
- Inplace fusion replaces the first module in the sequence with the fused module, and the rest with identity modules
"""
torch.quantization.fuse_modules(m, ['0','1'], inplace=True) # fuse first Conv-ReLU pair
torch.quantization.fuse_modules(m, ['2','3'], inplace=True) # fuse second Conv-ReLU pair

"""Insert stubs"""
m = nn.Sequential(torch.quantization.QuantStub(), 
                  *m, 
                  torch.quantization.DeQuantStub())

"""Prepare"""
m.qconfig = torch.quantization.get_default_qconfig(backend)
torch.quantization.prepare(m, inplace=True)

"""Calibrate
- This example uses random data for convenience. Use representative (validation) data instead.
"""
with torch.inference_mode():
  for _ in range(10):
    x = torch.rand(1,2,28)  
    m(x)
    
"""Convert"""
torch.quantization.convert(m, inplace=True)

"""Check"""
print(m[1].weight().element_size()) # 1 byte instead of 4 bytes for FP32


## FX GRAPH
from torch.quantization import quantize_fx
m.eval()
qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
# Prepare
model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
# Calibrate - Use representative (validation) data.
with torch.inference_mode():
  for _ in range(10):
    x = torch.rand(1,2,28)
    model_prepared(x)
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared)
```

### Quantization-aware Training (QAT)
The previous two methods are to quantize FP32 models *after* they have been trained. Although they work surprisingly well ([citation to some case study](google.com)), they're still subject to prediction errors arising from the drop in numerical precision.

<p align="center">
<img src="/assets/images/ptq_vs_qat.png" alt="Fig. 6: Comparison of PTQ and QAT" width="100%">
<br>
Wu, Hao, et al. "Integer quantization for deep learning inference: Principles and empirical evaluation." arXiv preprint arXiv:2004.09602 (2020)
</p>


Figure 6(a) shows why. In PTQ, the FP32 model's parameters are chosen by optimizing on the FP32 loss, and then projected to INT8 space. Depending on where the new INT8 weights lie on the loss curve, model accuracies can significantly change.

In QAT, the FP32 parameters are chosen by also optimizing on the INT8 loss. This allows the model to identify a wider region in the loss function (Figure 6(b)), and identify FP32 parameters such that quantizing them does not significantly affect accuracy.

It's likely that you can still use QAT by "fine-tuning" it on a sample of the training dataset, but I did not try it on demucs (yet).


## Quantizing "real-world" models

**Download the [notebook](https://gist.github.com/suraj813/735357e56321237950a0348b50f2f3b4) or run it on [Colab](https://colab.research.google.com/gist/suraj813/735357e56321237950a0348b50f2f3b4/fx-and-eager-mode-quantization-example.ipynb) (note that Colab runtimes may differ significantly from local machines).**

Traceable models can be easily quantized with FX Graph Mode, but it's possible the model you're using is not traceable end-to-end. Maybe it has loops or `if` statements on inputs (dynamic control flow), or relies on third-party libraries. The model we use in this example has [dynamic control flow and uses third-party libraries](https://github.com/facebookresearch/demucs/blob/v2/demucs/model.py). As a result, it cannot be symbolically traced directly. In this code walkthrough, I show how you can bypass this limitation by quantizing the child modules individually for FX Graph Mode, and how to patch Quant/DeQuant stubs in Eager Mode.



## What's next - Define-by-Run Quantization
At PyTorch Dev Day 2021, we got a sneak peek into the next cool feature in PyTorch quantization - [define-by-run](https://s3.amazonaws.com/assets.pytorch.org/ptdd2021/posters/C8.png). DBR attempts to improve usability by resolving the problem of model non-traceability.
For example, this dynamic control flow block would not be traceable:

```python
def forward(self, x):
# ....
    if x.shape[0] == 1:
         assert x.shape[1] == 1
# .....
```

As you might have seen in the real-world example above, refactoring the model can require effort. An early prototype of DBR aims to eliminate this cost. DBR dynamically traces the program, captures the subgraphs having quantizable ops, and performs the quantization transforms only on these subgraphs. The rest of the program is executed as-is. Although the if-block above operates on an input variable `x`, it does not perform any quantizable operation. DBR would not require this to be traced, and it can be executed as-is. 

DBR is an early prototype [code](https://github.com/PyTorch/PyTorch/tree/master/torch/ao/quantization/_dbr) but feel to play around and provide feedback via Github Issues.



## References
1. [A Survey of Quantization Methods for Efficient Neural Network Inference (arxiv)](https://arxiv.org/pdf/2103.13630.pdf)
2. [Quantizing Deep Convolutional Networks for Efficient Inference (arxiv)](https://arxiv.org/pdf/1806.08342.pdf)
2. [Integer quantization for deep learning inference: Principles and empirical evaluation (arxiv)](https://arxiv.org/abs/2004.09602)
3. [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization)
4. [Introduction to Quantization in PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)