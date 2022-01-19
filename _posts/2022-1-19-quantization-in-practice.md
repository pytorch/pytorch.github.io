---
layout: blog_detail
title: 'Quantization in Practice'
author: Suraj Subramanian
featured-img: ''
---

There are a few different ways to quantize your model with PyTorch. In this blog post, we'll take a look at how each technique looks like in practice. I will use a non-standard model that is not traceable, to paint an accurate picture of how much effort is really needed when quantizing your model.

<div class="text-center">
  <img src="/assets/images/quantization_gif.gif" width="60%">
</div>

## What happens when you "quantize" a model?

Two things, generally - the model gets smaller and runs faster. This is because adding and multiplying 8-bit numbers is faster than 32-bit numbers. Loading a smaller model from memory reduces I/O, making models more energy efficient.

> If someone asks you what time it is, you don't respond "10:14:34:430705", but you might say "a quarter past 10".

Quantizing a model means reducing the numerical precision of its weights and/or activations a.k.a information compression. Quantization of deep networks is especially interesting because overparameterized DNNs have more degrees of freedom and this makes them good candidates for information compression.

At the heart of it all is a **mapping function**, a linear projection from floating-point to integer space: $Q(r) = round(r/S + Z)$

To reconvert to floating point space, the inverse function is given by $\tilde r = (Q(r) - Z) \cdot S$. $\tilde r \neq r$, and their difference constitutes the **quantization error**.

The scaling factor $S$ is simply the ratio of the input range to the output range: $S = \frac{\beta - \alpha}{\beta_q - \alpha_q}$
where [$\alpha, \beta$] is the clipping range of the input, i.e. the boundaries of permissible inputs. [$\alpha_q, \beta_q$] is the range in quantized output space that it is mapped to. For 8-bit quantization, the output range $\beta_q - \alpha_q <= (2^8 - 1) $.

The process of choosing the appropriate input range is known as **calibration**; commonly used methods are MinMax and Entropy. 

The zero-point $Z$ acts as a bias to ensure that a 0 in the input space maps perfectly to a 0 in the quantized space. $Z = -(\frac{\alpha}{S} - \alpha_q)$


### Quantization Schemes
$S, Z$ can be calculated and used for quantizing an entire tensor ("per-tensor"), or individually for each channel ("per-channel").

When [$\alpha, \beta$] are centered around 0, it is called **symmetric quantization**. The range is calculated as $-\alpha = \beta = max(|max(r)|,|min(r)|)$. This removes the need of a zero-point offset in the mapping function. Asymmetric or **affine** schemes simply assign the boundaries to the minimum and maximum observed values. Asymmetric schemes have a tighter clipping range (for non-negative ReLU activations, for instance) but require a non-zero offset.


### PyTorch Classes
`Observer` modules ([docs](https://PyTorch.org/docs/stable/torch.quantization.html?highlight=observer#observers), [code](https://github.com/PyTorch/PyTorch/blob/748d9d24940cd17938df963456c90fa1a13f3932/torch/ao/quantization/observer.py#L88)) collect statistics on the input values and calculate the qparams $S, Z$. 

The `QConfig` ([code](https://github.com/PyTorch/PyTorch/blob/d6b15bfcbdaff8eb73fa750ee47cef4ccee1cd92/torch/ao/quantization/qconfig.py#L165)) NamedTuple specifies the observers and quantization schemes for the network's weights and activations. The default qconfig is at `torch.quantization.get_default_qconfig(backend)` where `backend='fbgemm'` for x86 CPU and `backend='qnnpack'` for ARM. 


## In PyTorch

PyTorch allows you a few different ways to quantize your model, depending on
- if you prefer a manual, or a more automatic process (*Eager Mode* v/s *FX Graph Mode*)
- if $S, Z$ for quantizing activations (layer outputs) are precomputed for all inputs, or calculated afresh with each input (*static* v/s *dynamic*),
- if $S, Z$ are computed during, or after training (*quantization-aware training* v/s *post-training quantization*)

Each approach has its unique tradeoffs; for instance FX Graph Mode can automagically figure out the right quantization configurations, but only for models that are [symbolically traceable](https://PyTorch.org/docs/stable/fx.html#torch.fx.symbolic_trace). Dynamic quantization can offer better precision at the cost of additional overhead in each inference. 

### Post-Training Dynamic Quantization 
The model's weights are pre-quantized before inference, but the activations are calibrated and quantized on the fly during inference. The simplest of all approaches, it has a one line API call in `torch.quantization.quantize_dynamic`

Because the calibrations are bespoke, clipping ranges can be tighter; dynamic quantization can therefore theoretically give higher accuracies, but calibrating and quantizing each layer's activations can add to the overhead. 

For this reason, it is best suited for models where most of the execution time is spent in loading weights from memory (think very large models with billions of parameters). 

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
Similar to dynamic quantization, weights are pre-quantized. In dynamic, the activations are calibrated and quantized at inference time. In static, activations are precalibrated by passing sample inputs to the model. 

This method has faster inference than dynamic quantization, but is less robust to accuracy drops from out-of-distribution inputs, i.e. if the model encounters data different from the sample inputs it has calibrated.

```python
# Static quantization of a model consists of the following steps:

#     Fuse modules
#     Insert Quant/DeQuant Stubs
#     Prepare the fused module (insert observers before and after)
#     Calibrate the prepared module (pass it representative data)
#     Convert the calibrated module (replace with quantized version)

import torch
from torch import nn

m = nn.Sequential(
     nn.Conv1d(2,64,(8,)),
     nn.ReLU(),
     nn.Conv1d(64, 128, (1,)),
     nn.ReLU()
)

## EAGER MODE
"""Fuse"""
torch.quantization.fuse_modules(m, ['0','1'], inplace=True) # fuse first Conv-ReLU pair
torch.quantization.fuse_modules(m, ['2','3'], inplace=True) # fuse second Conv-ReLU pair

"""Insert stubs"""
m = nn.Sequential(torch.quantization.QuantStub(), 
                  *m, 
                  torch.quantization.DeQuantStub())

"""Prepare"""
m.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(m, inplace=True)

"""Calibrate"""
with torch.inference_mode():
  for _ in range(10):
    x = torch.rand(1,2,28)
    m(x)
    
"""Convert"""
torch.quantization.convert(m, inplace=True)

"""Check"""
print(m[1].weight().element_size()) # 1 instead of 4 for FP32


## FX GRAPH
from torch.quantization import quantize_fx
qconfig_dict = {"": torch.quantization.get_default_qconfig('fbgemm')}
model_to_quantize.eval()
# Prepare
model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
# Calibrate 
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

Traceable models can be easily quantized with FX Graph Mode, but it's possible the model you're using is not traceable end-to-end. Maybe it has loops or `if` statements on inputs (dynamic control flow), or relies on third-party libraries. The model I use in this example has [dynamic control flow and uses third-party libraries](https://github.com/facebookresearch/demucs/blob/v2/demucs/model.py). As a result, it cannot be symbolically traced directly. In this code walkthrough, I show how you can bypass this limitation by quantizing the child modules individually for FX Graph Mode, and how to patch Quant/DeQuant stubs in Eager Mode.

Download the [notebook](https://gist.github.com/suraj813/735357e56321237950a0348b50f2f3b4) or run it on [Colab](https://colab.research.google.com/gist/suraj813/735357e56321237950a0348b50f2f3b4/fx-and-eager-mode-quantization-example.ipynb) (note that Colab runtimes may differ significantly from local machines).


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
[Quantization Docs](https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization)
[Integer quantization for deep learning inference: Principles and empirical evaluation (arxiv)](https://arxiv.org/abs/2004.09602)
[A Survey of Quantization Methods for Efficient Neural Network Inference (arxiv)](https://arxiv.org/pdf/2103.13630.pdf)
arxiv