---
layout: blog_detail
title: 'Overview of PyTorch Autograd Engine'
author: Vasilis Vryniotis and Francisco Massa
---

### Preliminaries

This blog post is based on PyTorch version 1.8, although it should apply for older versions too, since most of the mechanics have remained constant.

To help understand the concepts explained here, it is recommended that you read the awesome blog post by @ezyang: [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/) if you are not familiar with PyTorch architecture components such as ATen or c10d.

### What is autograd?

**Background**

PyTorch computes the gradient of a function with respect to the inputs by using automatic differentiation. Automatic differentiation is a technique that, given a computational graph, calculates the gradients of the inputs. Automatic differentiation can be performed in two different ways; forward and reverse mode. Forward mode means that we calculate the gradients along with the result of the function, while reverse mode requires us to evaluate the function first, and then we calculate the gradients starting from the output. While both modes have their pros and cons, the reverse mode is the de-facto choice since the number of outputs is smaller than the number of inputs, which allows a much more efficient computation. Check [3] to learn more about this.

Automatic differentiation relies on a classic calculus formula known as the chain-rule. The chain rule allows us to calculate very complex derivatives by splitting them and recombining them later.

Formally speaking, given a composite function <a href="https://www.codecogs.com/eqnedit.php?latex=f(g(x))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(g(x))" title="f(g(x))" /></a>, we can calculate its derivative as <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial}{\partial&space;x}&space;f(g(x))&space;=&space;f'(g(x))g'(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;x}&space;f(g(x))&space;=&space;f'(g(x))g'(x)" title="\frac{\partial}{\partial x} f(g(x)) = f'(g(x))g'(x)" /></a>. This result is what makes automatic differentiation work.
By combining the derivatives of the simpler functions that compose a larger one, such as a neural network, it is possible to compute the exact value of the gradient at a given point rather than relying on the numerical approximation, which would require multiple perturbations in the input to obtain a value.

To get the intuition of how the reverse mode works, let’s look at a simple function <a href="https://www.codecogs.com/eqnedit.php?latex=f(x,&space;y)&space;=&space;log(x*y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x,&space;y)&space;=&space;log(x*y)" title="f(x, y) = log(x*y)" /></a>. Figure 1 shows its computational graph where the inputs x, y in the left, flow through a series of operations to generate the output z.

The automatic differentiation engine will normally execute this graph. It will also extend it to calculate the derivatives of w with respect to the inputs x, y, and the intermediate result v.

The example function can be decomposed in f and g, where <a href="https://www.codecogs.com/eqnedit.php?latex=f(x,&space;y)&space;=&space;log(g(x,&space;y))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x,&space;y)&space;=&space;log(g(x,&space;y))" title="f(x, y) = log(g(x, y))" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=g(x,&space;y)&space;=&space;xy" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(x,&space;y)&space;=&space;xy" title="g(x, y) = xy" /></a>.  Every time the engine executes an operation in the graph, the derivative of that operation is added to the graph to be executed later in the backward pass. Note, that the engine knows the derivatives of the basic functions.

In the example above, when multiplying x and y to obtain v, the engine will extend the graph to calculate the partial derivatives of the multiplication by using the multiplication derivative definition that it already knows. <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial}{\partial&space;x}&space;g(x,&space;y)&space;=&space;y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;x}&space;g(x,&space;y)&space;=&space;y" title="\frac{\partial}{\partial x} g(x, y) = y" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial}{\partial&space;y}&space;g(x,&space;y)&space;=&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;y}&space;g(x,&space;y)&space;=&space;x" title="\frac{\partial}{\partial y} g(x, y) = x" /></a> . The resulting extended graph is shown in Figure 2, where the *MultDerivative* node also calculates the product of the resulting gradients by an input gradient to apply the chain rule; this will be explicitly seen in the following operations. Note that the backward graph (green nodes) will not be executed until all the forward steps are completed.
