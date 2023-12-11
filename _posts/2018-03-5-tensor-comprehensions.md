---
layout: blog_detail
title: 'Tensor Comprehensions in PyTorch'
author: Priya Goyal (FAIR), Nicolas Vasilache (FAIR), Oleksandr Zinenko (Inria & DI ENS), Theodoros Theodoridis (ETH Zürich), Zachary DeVito (FAIR), William S. Moses (MIT CSAIL), Sven Verdoolaege (FAIR), Andrew Adams (FAIR), Albert Cohen (Inria & DI ENS & FAIR)
redirect_from: /2018/03/05/tensor-comprehensions.html
---

Tensor Comprehensions (TC) is a tool that lowers the barrier for writing high-performance code. It generates GPU code from a simple high-level language and autotunes the code for specific input sizes.

**We highly recommend reading the [Tensor Comprehensions blogpost](https://research.fb.com/announcing-tensor-comprehensions/) first.**

If you ran into any of the following scenarios, TC is a useful tool for you.

- Your PyTorch layer is large and slow, and you contemplated writing a dedicated C++ or CUDA code for it. But you don't know how to program in CUDA or write low-level code.

- You wrote a CUDA layer, but it took a week to write, debug, optimize for speed. You wished you could do this in an hour.

- You want to fuse multiple layers like Conv-ReLU-BatchNorm or Linear-ReLU-Linear-ReLU in your network for speed, but it was quite difficult to comprehend

- Your research involves weird Tensor shapes that CuDNN and MKL are not optimized for. For example, you do convolutions of 13 x 24 with an input image of 143 x 55. You tried running it with CuDNN and it was slower than you wished.

- Your code is slowed-down by transposing Tensors constantly to fit a particular memory layout. You wish it was easy to write custom code that operates efficiently on your input layout.


Tensor Comprehensions are seamless to use in PyTorch, interoperating with PyTorch Tensors and `nn` Variables.

Let us run through using TC with PyTorch.

#### 1. Install the package

```bash
conda install -c pytorch -c tensorcomp tensor_comprehensions
```

At this time we only provide Linux-64 binaries which have been tested on Ubuntu 16.04 and CentOS7.

TC depends on heavyweight C++ projects such as [Halide](http://halide-lang.org/), [Tapir-LLVM](https://github.com/wsmoses/Tapir-LLVM) and ISL. Hence, we rely on Anaconda to distribute these dependencies reliably. For the same reason, TC is not available via PyPI.

#### 2. Import the python package

```python
import tensor_comprehensions as tc
```

#### 3. Define the TC expression and create a python function

```python
lang = """
def fcrelu(float(B,M) I, float(N,M) W1, float(N) B1) -> (O1) {
    O1(b, n) +=! I(b, m) * W1(n, m)
    O1(b, n) = O1(b, n) + B1(n)
    O1(b, n) = fmax(O1(b, n), 0)
}
"""
fcrelu = tc.define(lang, name="fcrelu")
```

This `fcrelu` function takes PyTorch Tensors as input and returns a PyTorch Tensor. It takes input `I`, weight `W1`, bias `B1` and returns output `O1`.

#### 4. Let's create some dummy input tensors

```python
B, M, N = 100, 128, 100
I, W1, B1 = torch.randn(B, M).cuda(), torch.randn(N, M).cuda(), torch.randn(N).cuda()
```

#### 5. Now autotune the function for your input sizes

```python
fcrelu.autotune(I, W1, B1, cache="fcrelu_100_128_100.tc")
```

The autotuner is your biggest friend. You generally do not want to use a `tc` function without autotuning it first.

When the autotuning is running, the current best performance is displayed. If you are satisfied with the current result or you are out of time, stop the tuning procedure by pressing `Ctrl+C`.

`cache` saves the results of the autotuned kernel search and saves it to the file `fcrelu_100_128_100.tc`. The next time you call the same line of code, it loads the results of the autotuning without recomputing it.

The autotuner has a few hyperparameters (just like your ConvNet has learning rate, number of layers, etc.). We pick reasonable defaults, but you can read about using advanced options [here](https://facebookresearch.github.io/TensorComprehensions/framework/pytorch_integration/writing_layers.html#specifying-mapping-options).

#### 6. Call the function with the inputs, to get your result

```python
out = fcrelu(I, W1, B1)
```

Now, let's look at how to write TC expressions.

## A quick primer on the TC language

The TC notation focuses on the mathematical nature of the layer, leaving performance considerations to it's backend code that uses Halide and polyhedral compilation techniques which accumulate decades of cutting edge Loop Nest Optimization (LNO) research.

TC is close to [np.einsum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html). We shall quickly learn TC by example

```python
lang = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
"""
```

In this example, we define a function `matmul` which takes two input `A` and `B` of shapes `M x N` and `N x K` and returns a single `output`. The shape of `output` is automatically inferred by the TC language (discussed below).

Let's look at this line:

```python
output(i, j) +=! A(i, kk) * B(kk, j)
```

It says:

- `output(i, j)` means output is 2D.
- for each location `output(i, j)`, we add (`+=`) `A(i, kk) * B(kk, j)`.
- `i` is well-defined as all locations in `A` dim=0, i.e. `i in range(0, M)`
- `j` is well-defined as all locations in `B` dim=1, i.e. `j in range(0, K)`
- `kk` is inferred as all locations from `0` to `N`

The shape of output is inferred from the maximum values `i` and `j` can take, which is `M` and `K`, so output is of size `M x K`.

The `!` symbol initializes output with `0.0`. It is equivalent to:

```python
output(i, j) = 0
output(i, j) += A(i, kk) * B(kk, j)
```

**Scalar inputs and range constraints: implement AvgPool2d**

```python
"""

{% raw %}def avgpool(float(B, C, H, W) input) -> (output) {{{% endraw %}
  output(b, c, h, w) += input(b, c, h * {sH} + kh, w * {sW} + kw) where kh in 0:{kH}, kw in 0:{kW}
{% raw %}}}{% endraw %}

"""
avgpool = tc.define(LANG, name="avgpool", constants={"sH":1, "sW":1, "kH":2, "kW":2})
```

here the `where` keyword can take ranges of values to operate on. `0:{kH}` is equivalent `range(kH)` in Python.

Note: the syntax for passing in scalars is subject to change in the next release.

## torch.nn layers

We added some sugar-coating around the basic PyTorch integration of TC to make it easy to integrate TC into larger `torch.nn` models by defining the forward and backward TC expressions and taking `Variable` inputs / outputs. 

## Some essentials that you will miss (we're working on them)

### Autotuning for variable-length sequences

The TC auto-tuner requires all input sizes to be specified before-hand. For example, if you have input `I1` which is an image batch, the autotuner wants to know the exact shape of `I1` to generate an optimized kernel. You cannot specify: `image with height between 200 and 300`. This is more essential in sequence data such as NLP, where each sentence can have a different length.

The reason why the autotuner is non-parametric is because it's harder and harder to auto-tune parametric constraints, this is active research. Hence, for the first release, we made a conscious decision to give you the tool in a form where we know it works well.

As a work-around, if you know that you have a few specific shapes of interest, you can run the autotuner with these multiple shapes.

```python
relu = tc.define(LANG, name="relu")
batch, channels = 16, 3
tc.autotune((batch, channels, 32, 32)) # image of size 32 x 32
tc.autotune((batch, channels, 48, 48)) # image of size 48 x 48
tc.autotune((batch, channels, 64, 64)) # image of size 64 x 64
```

Now the autotuner is tuned for these three specific image sizes `32x32`, `48x48` and `64x64`.

### Lack of loops

If you want to write an RNN, it's easy to see it as a `for` loop over time. However, the TC language does not have loops yet. If you really want to write RNNs, you can write unrolled loops.

### Strided-Tensors

The TC backend does not support non-contiguous Tensors yet. If the inputs you give are not contiguous, they are made contiguous before passing to the TC backend.

### Reshaping Tensors within a TC expression

You cannot write this operation in TC: `torch.matmul(...).view(...).mean(...)`. Whenever there is need for a `view` to change the shape of an input, you have to get the output, `view` it at the PyTorch level.

## Getting Started

- [Walk through Tutorial](https://facebookresearch.github.io/TensorComprehensions/framework/pytorch_integration/writing_layers.html) to quickly get started with understanding and using Tensor Comprehensions PyTorch package.
- Over 20 examples of various ML layers with TC, including `avgpool`, `maxpool`, `matmul`, matmul - give output buffers and `batch-matmul`, `convolution`, `strided-convolution`, `batchnorm`, `copy`, `cosine similarity`, `Linear`, `Linear + ReLU`, `group-convolutions`, strided `group-convolutions`, `indexing`, `Embedding` (lookup table), small-mobilenet, `softmax`, `tensordot`, `transpose`
- [Detailed docs](https://facebookresearch.github.io/TensorComprehensions/framework/pytorch_integration/getting_started.html) on Tensor Comprehensions and integration with PyTorch.

## Communication

- Slack: For discussion around framework integration, build support, collaboration, etc. join our slack channel.
- Email: tensorcomp@fb.com
- [GitHub](https://github.com/facebookresearch/TensorComprehensions): bug reports, feature requests, install issues, RFCs, thoughts, etc.

## Acknowledgements

We would like to thank Soumith Chintala, [Edward Yang](https://github.com/ezyang) and [Sam Gross](https://github.com/colesbury) for their immense guidance and help in making the integration API nice and smooth. We would also like to thank rest of the PyTorch team and our pre-release users for their helpful feedback that guided us in making the integration better.
