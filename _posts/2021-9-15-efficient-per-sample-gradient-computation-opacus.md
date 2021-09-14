---
layout: blog_detail
title: 'Differential Privacy Series Part 2 | Efficient Per-Sample Gradient Computation in Opacus'
author: Ashkan Yousefpour, Davide Testuggine, Alex Sablayrolles, and Ilya Mironov
featured-img: 'assets/images/image-opacus.png'
---

## Introduction 

In our [previous blog post](https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3), we went over the basics of the DP-SGD algorithm and introduced [Opacus](https://opacus.ai), a PyTorch library for training ML models with differential privacy. In this blog post, we explain how performance-improving vectorized computation is done in Opacus and why Opacus can compute “per-sample gradients” a lot faster than “microbatching” (read on to see what all these terms mean!)  

## Context

Recall that differential privacy is all about worst-case guarantees, which means we need to check the gradient of each and every sample in a batch of data.

Conceptually, this is akin to writing the following PyTorch code:

```python
optimizer = torch.optim.SGD(lr=args.lr)

for batch in Dataloader(train_dataset, batch_size):
    all_per_sample_gradients = [] # will have len = batch_size
    for x,y in batch:
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()  # Now p.grad for this x is filled
        
        # Need to clone it to save it
        per_sample_gradients = [p.grad.detach().clone() for p in model.parameters()]
        
        all_per_sample_gradients.append(per_sample_gradients)
        model.zero_grad()  # p.grad is cumulative so we'd better reset it
```

While the above procedure (called the “micro batch method”, or “micro batching”) does indeed yield correct per-sample gradients, it’s grossly inefficient: GPUs really like vectorized computation, and going sample-by-sample in a for-loop entirely misses that. We acknowledged this in the ending of our last entry, leaving the explanation of how to make this faster for a later post. This is it, folks!

## Vectorized Computation

One of the features of Opacus is “vectorized computation”, in that it can compute per-sample gradients a lot faster than microbatching (they depend on the model, but we observed speedups from ~10x for small MNIST examples to ~50x for Transformers). Microbatching is simply not fast enough to run experiments and conduct research.

So, how do we do vectorized computation in Opacus? We derive the per-sample gradient formula, and implement a vectorized version of it. We will get to this soon. Let us mention that there are other methods  (like [this](https://arxiv.org/abs/1510.01799) and [this](https://arxiv.org/abs/2009.03106)) that rely on computing the norm of the per-sample gradients directly. It is worth noting that since these approaches are based on computing the norm of the per-sample gradients, they do two passes of back-propagation to compute the per-sample gradients: one pass for obtaining the norm, and one pass for using the norm as a weight (see the links above for details). Although they are considered efficient, in Opacus we set out to be even more efficient (!) and do everything in one back-propagation pass. 

In this blog post, we focus on the approach for efficiently computing per-sample gradients that is based on deriving the per-sample gradient formula and implementing a vectorized version of it. To make this blog post short, we focus on simple linear layers - building blocks for multi-layer perceptrons (MLPs). In our next blog post, we will talk about how we extend this approach to other layers (e.g., convolutions, or LSTMs) in Opacus.

## Efficient Per-Sample Gradient Computation for MLP

To understand the idea for efficiently computing per-sample gradients, let’s start by talking about how AutoGrad works in the commonly-used deep learning frameworks. We’ll focus on PyTorch from now on, but to the best of our knowledge the same applies to other frameworks (with the exception of Jax). 

For simplicity of explanation, we focus on one linear layer in a neural network, with weight matrix W. Also, we omit the bias from the forward pass equation: assume the forward pass is denoted by Y=WX where X is the input and Y is the output of the linear layer. If we are processing a single sample, X is a vector. On the other hand, if we are processing a batch (and that’s what we do in Opacus), X is a matrix of size B*d, with B rows (B is the batch size), where each row is an input vector of dimension d. Similarly, the output matrix Y would be of size B*r where each row is the output vector corresponding to an element in the batch and r is the output dimension. 

The forward pass can be written as the following equation that captures the computation for each element in the matrix Y:

Yi(b)=j=1dWi,jXj(b)

We will return to this equation shortly. Yi(b)denotes the element at row b (batch b) and column i (remember that the dimension of Y is  B*r).

In any machine learning problem, we normally need the derivative of the loss with respect to weights W. Comparably, in Opacus we need the “per-sample” version of that, meaning, per-sample derivative of the loss with respect to weights W. Let’s first get the derivative of the loss with respect to weights, and soon, we will get to the per-sample part. 

To obtain the derivative of the loss with respect to weights, we use the chain rule, whose general form is:

Lz=LY*Yz,

which can be written as

Lz=b=1Bi'=1rLYi'(b)Yi'(b)z.

Now, we can replace z with Wi,jand get

LWi,j=b=1Bi'=1rLYi'(b)Yi'(b)Wi,j.

We know from the equation Y=WX that Yi'(b)Wi,j is Xj(b) when i=i’, and is 0 otherwise. Hence, we will have

LWi,j=b=1BLYi(b)Xj(b)(*)

This equation corresponds to a matrix multiplication in PyTorch.
 		
As we can see, the gradient of loss with respect to the weight relies on the gradient of loss with respect to the output Y. In a regular backpropagation, the gradients of loss with respect to weights (or simply put, the “gradients”) are computed for the output of each layer, but they are reduced (i.e., summed up over the batch). Since Opacus requires computing **per-sample gradients**, what we need is the following

LbatchWi,j=LYi(b)Xj(b)(**)

Note that the two equations are very similar; one equation has the sum over the batch and the other one does not. Let’s now focus on how we compute the per-sample gradient (equation ** ) in Opacus efficiently. 

<p align="center">
<img src="{{ site.url }}/assets/images/image-opacus.png" width="560">
<br>
Figure 6. The partition boundary is in the middle of a skip connection
</p>

A bit of notation and terminology. Recall that we used the notation Y = WX for forward pass of a single layer of a neural network. When the neural network has more layers, a better notation would be Z(l+1)= W (l+1)Z(l), where l corresponds to each layer of the neural network. In that case, we can call the gradients with respect to any activations Z(l) the “highway gradients” and the gradients with respect to the weights the “exit gradients”.

If we go with this definition, explaining the issue with Autograd is a one-liner: highway gradients retain per-sample information, but exit gradients do not. Or, highway gradients are per-sample, but exit gradients are not necessarily. This is unfortunate because the per-sample exit gradients are exactly what we need!

So here’s the question for us: given that we do have vectorized information in the highway, can we compute the per-sample exit gradients efficiently?

Luckily for us, there is a solution for this:
1. Store the activations somewhere.
2. Find a way to access the highway gradients.

So far, so good; but how do we store the activations and how do we access the highway gradients? Well, PyTorch has a feature to do just these: module (and tensor) hooks! Read on.

Under the hood, PyTorch is event-based and will call the hooks at the right places (your forward and backward functions are indeed being hooked where they need to go). In addition, PyTorch exposes hooks so that anyone can leverage them. The ones we care about here are these:

1. **Parameter hook**. This attaches to a nn.Module's Parameter tensor and will always run during the backward pass. The signature is this: ```hook(grad) -> Tensor or None```
2. **nn.Module hook**. There are two types of these:
    1. **Forward hook**. The signature for this is ```hook(module, input, output) -> None```
    2. **Backward hook**. The signature for this is ```hook(module, grad_input, grad_output) -> Tensor or None```
    
To learn more about these fundamental primitives, check out our [official tutorial](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks) on hooks, or one of the excellent explainers, such as [Paperspace’s](https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/) or this [Kaggle notebook](https://www.kaggle.com/sironghuang/understanding-pytorch-hooks). Finally, if you want to play with hooks more interactively, we also made a [notebook](https://colab.research.google.com/drive/1zDidGCNI3DJk1oSPIpmB89b5cCWyuHao?usp=sharing) for you.

We use two hooks, one forward hook and one backward hook. In the forward hook, we simply store the activations:

```python
def forward_hook(module, input, output):
    module.activations = input
```

In the backward hook, we use the grad_output (highway gradient), along with the stored activations (input to the layer) to compute the per-sample gradient as below:

```python
def backward_hook(module, grad_input, grad_output):
    module.grad_sample = compute_grad_sample(module.activations, grad_output)
    module.activations = input
```

Now the final piece of the puzzle is the computation of the per-sample gradient itself, in the method ```compute_grad_sample``` above. Recall from Equation (* ) that the (average) gradient of loss with respect to the weights is the result of a matrix multiplication. In order to get the per-sample gradient, we want to remove the sum reduction, as in Equation (** ).  This corresponds to replacing the matrix multiplication with a batched outer product. Luckily for us, torch einsum allows us to do that in vectorized form. The method ```compute_grad_sample``` is defined based on einsum throughout our code. For instance, for the linear layer, the meat of the code is

```python
def compute_linear_grad_sample(input, grad_output):
    return torch.einsum("n...i,n...j->nij", B, A)
```

You can find the full implementation for the linear module [here](https://github.com/pytorch/opacus/blob/204328947145d1759fcb26171368fcff6d652ef6/opacus/grad_sample/linear.py). The actual code has some bookkeeping around the einsum call, but the einsum call is the main building block of the efficient per-sample computation for us.

Since this post is already long, we refer the interested reader to read about [einsum in PyTorch](https://rockt.github.io/2018/04/30/einsum) and do not get into the details of einsum. However, we really encourage you to check it out, as it’s kind of a magical thing! Just as an example, a matrix multiplication describe in 

Cij=kAikBkj

can be implemented beautifully in this line:

```c = torch.einsum('ik,kj->ij', [a, b])```

We like to highlight that einsum is really the key for us to have vectorized computation. That is it folks! We just explained the last piece of the puzzle, computation of the per-sample gradient.

## Conclusion

 In this blog post, we explained how vectorized computation is done in Opacus and why Opacus can compute per-sample gradients a lot faster than micro batching (for reference, [TensorFlow Privacy](https://github.com/tensorflow/privacy) is based on micro batching 😜). We explained the idea to compute per-sample gradients efficiently for an MLP. Stay tuned for more blog posts! In our next blog post, we will talk about how we compute per-sample gradients efficiently for other layers (e.g. convolutions, or LSTMs).
