---
layout: blog_detail
title: "Differential Privacy Series Part 3 | Efficient Per-Sample Gradient Computation for More Layers in Opacus"
author: Alex Sablayrolles, Ashkan Yousefpour, Karthik Prasad, Peter Romov, Davide Testuggine, Igor Shilov, and Ilya Mironov
featured-img: "/assets/images/blog-2022-10-31-Efficient-Per-Sample-Convolution-Layer.gif"
---

## Introduction

In the previous [blog post](https://bit.ly/per-sample-gradient-computing-opacus), we covered how performance-improving vectorized computation is done in Opacus and why [Opacus](https://opacus.ai) can compute “per-sample gradients” a lot faster than “microbatching”. We had also introduced the vectorized computation for nn.linear layers. In this blog post, we explain further on how per-sample gradients efficiently for other layer types: convolutions, RNNs, LSTMs, normalizations, embeddings, and multi-head attentions.

## Recap

In the previous [blog post](https://bit.ly/per-sample-gradient-computing-opacus), we covered the following:

- One of the features of Opacus is “vectorized computation”, in that it can compute per-sample gradients a lot faster than microbatching. To do so, we derive the per-sample gradient formula, and implement a vectorized version of it.

- The per-sample gradient formula is

<p align="center">
<img src="/assets/images/blog-2022-10-31-Efficient-Per-Sample-Gradient.png" width="20%">
</p>

- We call the gradients with respect to activations the “highway gradients” and the gradients with respect to the weights the “exit gradients”.

- Highway gradients retain per-sample information, but exit gradients do not.

- `einsum` facilitates vectorized computation.

We invite you to check out the previous [blog post](https://bit.ly/per-sample-gradient-computing-opacus) for details; briefly, Opacus computes per-sample gradients with the help of PyTorch hooks - we access the activation values with the forward hooks and the highway gradients for calculating per-sample gradients with the backward hooks. Some linear algebra on top of these captured values gives us the end result.

The basic concepts for other module types (other than nn.linear) remains the same; what changes is the linear algebra we do with `einsum` on activations and highway-gradients. This is what we explore in this post.

## Extending the idea to other modules

Now that we have seen how to efficiently compute per-sample gradients for linear layers (the building blocks of multilayer perceptrons (MLPs)), we can apply the underlying techniques to other layers too. First of all, note that this should be possible. Why? Let us explain. All a linear layer does is a matrix multiplication ([matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)) between the inputs and the parameters. All other kinds of layers are probably doing something like this too! The only difference is that they come with additional constraints, such as weight sharing in a convolution, or sequential accumulation in the backward pass in an LSTM. Here is how we do it for convolutions, LSTMs, multi-head attention, normalization, GRUs, and embedding layers.

### Convolution

As a refresher, let’s look at the forward pass of a convolution module. For simplicity, we consider a Conv2D with a 2x2 kernel operating on an input with just one channel (shape 1x3x3).

<p align="center">
<img src="/assets/images/blog-2022-10-31-Efficient-Per-Sample-Convolution-Layer.gif" width="90%">
</p>

Evidently, this operation is more than just a simple matrix multiplication that we see with an `nn.linear` module. However, if we were to “unfold” the input (read more about it here), we achieve the same results by performing a simple matrix multiplication (and some reshaping) as follows:

<p align="center">
<img src="/assets/images/blog-2022-10-31-Efficient-Per-Sample-Gradient-still.png" width="90%">
</p>

Now that we have rewritten convolution using matrix multiplication, we can implement an efficient matrix multiplication using `einsum` as we did for linear layers before. Opacus does exactly this: `unfold, matmul, reshape`. (see [here](https://github.com/pytorch/opacus/blob/main/opacus/grad_sample/conv.py) for the code)

Now is that the only way to compute vectorized per-sample gradients for Conv layers? No it’s not. Another approach is by exploiting the fact that the gradient of a convolution is [yet another convolution](https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c). It’s possible compute per-sample gradients, at a lower memory footprint using this approach, but at slower speed on pre-Volta GPUs (see [https://github.com/pytorch/opacus/issues/145](https://github.com/pytorch/opacus/issues/145) )

### Recurrent: RNN, GRU, and LSTM

A little background. Recurrent neural networks catch temporal effects by using intermediate hidden states connected in a sequence. Similar to other neural network blocks they map a sequence of input vectors to a sequence of output vectors. A recurrent neural network can be represented as a series of consequent flat layers, each consisting of a chain of cells (directed either forward or backward). A cell, a basic element of a recurrent neural network, transforms a single input token or its intermediate representation and updates the hidden state vector of the cell. The parameters of a recurrent layer are basically represented by the parameters of the underlying cells. All cells in one flat sublayer share the same set of parameters, i.e., regardless of the time, the input and the current hidden state go through the same transformation. There are different approaches to handling temporal dependencies and implementing recurrent neural networks. [RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html), [GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU) and [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM) are the three most popular implementations. They introduce different cell types, all based on a parameterized linear transformation, but the basic form of the neural network remains unchanged.

Okay, now let’s now talk about how to handle recurrent layers in Opacus. To efficiently compute per-sample gradients for recurrent layers, we need to overcome a little obstacle: the recurrent layers in PyTorch are implemented at the cuDNN layer, which means that it is not possible for Opacus to add a hook to the internal components of the cells.

To overcome this obstacle, we re-implemented all three cell types based on the [RNNLinear](https://github.com/pytorch/opacus/blob/fc71e2b627e5b0bf7119d8dee866af9057f78bb1/opacus/layers/dp_rnn.py#L39) layer. Basically, we clone `nn.Linear` to have a separate per-sample gradient computation function that accumulates gradients in a chain instead of concatenating them. Put simply, using RNNLinear linear tells Opacus that multiple occurrences of the same cell in the neural network are not for different training examples, rather they are for different tokens in one example. This allows Opacus to sum across the time dimension and save a lot of memory.

The final piece to add is a set of compatible replacements of the [original layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers): DPRNN, DPGRU, DPLSTM. They implement the same logic as the original modules from `torch.nn`, but based on the cells compatible with Opacus.

## Multi-Head Attention

A refresher on muli-head attention: Multi-head attention is one of the main components of a transformer. Multi-head attention computes queries, keys, and values by applying three linear layers on a sequence of input vectors, and returns a combination of the values weighted by the attention. The attention itself is obtained via softmax on the dot product between queries and keys. In Pytorch, all these components are fused together at the cuDNN level to allow for more efficient computation.

We implemented multi-head attention in Opacus in two steps:

- We rewrote the multi-head attention which has the underlying three linear layers. Opacus automatically hooks itself to these linear layers to compute per-sample gradients; these linear layers use `einsum` to compute grad samples, as discussed in the previous blog post.

- We implemented an additional SequenceBias layer which adds a bias vector to the whole sequence augmented with per-sample gradient computation. Note that the main part of implementation is SequenceBias, which is a pretty straightforward module.

In other words, Multi Head Attention is basically a collection of Linear layers, each of which uses `einsum` to compute per-sample gradients.

### Normalization Layers

With Differential Privacy, batch normalization layers are prohibited because they mix information across samples of a batch. Nevertheless, other types of normalization - such as `LayerNorm`, `InstanceNorm`, or `GroupNorm`- are allowed and supported as they do not normalize over the batch dimension and hence do not mix information.

`LayerNorm` normalizes over all the channels of a particular sample and `InstanceNorm` normalizes over one channel of a particular sample. `GroupNorm‘s` operation lies in between those of LayerNorm and InstanceNorm; it normalizes over a “group” of channels of a particular sample.

These normalization layers are illustrated in the following image (borrowed from [https://arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494) )

<p align="center">
<img src="/assets/images/blog-2022-10-31-Efficient-Per-Sample-Normalization.png" width="90%">
</p>

It is easily seen that these normalization layers can be split into a `linear` layer (have you realized the pattern of our tricks yet? :) ) and a non-parameterized layer (that performs the mean/variance normalization - the normalization layer). Consequently, the implementations for computing per-sample gradients are also quite simple and similar to that of a linear layer.

### Embedding

An embedding layer can (once again) be viewed as a special case of a linear layer where the input is one-hot encoded, as shown in this figure.

<p align="center">
<img src="/assets/images/blog-2022-10-31-Efficient-Per-Sample-matrix-multiplication.png" width="90%">
</p>

Thus, the layer’s gradient is the outer product of the one-hot input and the gradient of the output: concretely, this means that the layer’s gradient is a matrix of zeros, except at the row corresponding to the input index, where the value is the gradient of the output. In particular, the gradient with respect to the embedding layer is very sparse (the only updated embeddings are those from the current data sample). Hence for implementing per-sample gradients, we instantiate a zero-matrix for the embedding gradient and add the gradient only to the input positions.

## Discussion

In summary, Opacus computes per-sample gradients by (1) capturing the activations and highway gradients, and then (2) efficiently performing matrix multiplications.

For modules that are not readily amenable to matrix multiplications (e.g., Conv, normalization), we do some linear algebra circus to get it in the right form. For modules that do not allow us to attach hooks (e.g., RNNs, MultiHeadAttention), we reimplement them using `nn.Linear` and proceed as usual.

When we re-implement the modules, we ensure that their `param_dict()` is fully compatible with that of their non-DP counterparts. This way, for instance, when you finish training your `DPMultiHeadAttention`, you can directly load its weights onto a `nn.MultiheadAttention` and serve it in production for inference, without even requiring that you have Opacus installed!

A module can be either a **building block**, or a **composite**:

1. **building block.** These are “atomic” trainable modules (i.e., “default classes”) that have their own hooks, and can be used directly, for example, `nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,` and the normalization layers (`nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm`). See points 1,2,3 [here](https://github.com/pytorch/opacus/tree/main/opacus#supported-modules).

2. **Composite.** These are modules that are composed of building blocks. Composite modules are supported as long as all trainable submodules are supported. Frozen submodules need not be supported; A nn.Module can be frozen in PyTorch by unsetting `requires_grad` in each of its parameters.

Needless to say, modules with no trainable parameters (e.g. nn.ReLU, and nn.Tanh) and modules that are frozen don’t need their per-sample gradients computed, and hence these modules are supported out of the box.

The above discussion about supported modules for efficient gradient computation is summarized in this [README](https://github.com/pytorch/opacus/blob/master/opacus/README.md).

That’s all folks! That’s how Opacus implements other layers (e.g., convolutions, embedding, normaliations) and how it supports custom modules.

## Conclusion

In this blog post, we explained the idea of efficiently computing per-sample gradients in Opacus for other layers: convolutions, LSTMs, multi-head attentions, normalizations, GRUs, RNNs, LSTMs, and embeddings. We also explained how arbitrary modules can be supported in Opacus, as long as they consist of building block and composite modules.

Also with the [release of Opacus v1.2](https://github.com/pytorch/opacus/releases/tag/v1.2.0) which is focused on incorporating major improvements to the per sample gradient computation recently added to the core PyTorch, namely functorch and ExpandedWeights, Opacus now is even more flexible in computing per sample gradients. With functorch, Opacus can now handle almost all input models, removing previous limitation where we could only handle certain standard layers. With ExpandedWeight, per sample gradient computation will become up to 30% faster for the majority of the most popular models that are still composed of standard layers.

Stay tuned for more posts in the series and share your thoughts and feedback.
