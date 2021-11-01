---
layout: blog_detail
title: 'Overview of PyTorch Autograd Engine'
author: Preferred Networks, Inc.
---

This blog post is based on PyTorch version 1.8, although it should apply for older versions too, since most of the mechanics have remained constant.

To help understand the concepts explained here, it is recommended that you read the awesome blog post by [@ezyang](https://twitter.com/ezyang): [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/) if you are not familiar with PyTorch architecture components such as ATen or c10d.

### What is autograd?

**Background**

PyTorch computes the gradient of a function with respect to the inputs by using automatic differentiation. Automatic differentiation is a technique that, given a computational graph, calculates the gradients of the inputs. Automatic differentiation can be performed in two different ways; forward and reverse mode. Forward mode means that we calculate the gradients along with the result of the function, while reverse mode requires us to evaluate the function first, and then we calculate the gradients starting from the output. While both modes have their pros and cons, the reverse mode is the de-facto choice since the number of outputs is smaller than the number of inputs, which allows a much more efficient computation. Check [3] to learn more about this.

Automatic differentiation relies on a classic calculus formula known as the chain-rule. The chain rule allows us to calculate very complex derivatives by splitting them and recombining them later.

Formally speaking, given a composite function <a href="https://www.codecogs.com/eqnedit.php?latex=f(g(x))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(g(x))" title="f(g(x))" /></a>, we can calculate its derivative as <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial}{\partial&space;x}&space;f(g(x))&space;=&space;f'(g(x))g'(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;x}&space;f(g(x))&space;=&space;f'(g(x))g'(x)" title="\frac{\partial}{\partial x} f(g(x)) = f'(g(x))g'(x)" /></a>. This result is what makes automatic differentiation work.
By combining the derivatives of the simpler functions that compose a larger one, such as a neural network, it is possible to compute the exact value of the gradient at a given point rather than relying on the numerical approximation, which would require multiple perturbations in the input to obtain a value.

To get the intuition of how the reverse mode works, let’s look at a simple function <a href="https://www.codecogs.com/eqnedit.php?latex=f(x,&space;y)&space;=&space;log(x*y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x,&space;y)&space;=&space;log(x*y)" title="f(x, y) = log(x*y)" /></a>. Figure 1 shows its computational graph where the inputs x, y in the left, flow through a series of operations to generate the output z.

<div class="text-center">
  <img src="{{ site.baseurl }}/assets/images/f_x_y_graph.png" width="100%">
  <p>Figure 1: Computational graph of f(x, y) = log(x*y)</p>
</div>

The automatic differentiation engine will normally execute this graph. It will also extend it to calculate the derivatives of w with respect to the inputs x, y, and the intermediate result v.

The example function can be decomposed in f and g, where <a href="https://www.codecogs.com/eqnedit.php?latex=f(x,&space;y)&space;=&space;log(g(x,&space;y))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x,&space;y)&space;=&space;log(g(x,&space;y))" title="f(x, y) = log(g(x, y))" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=g(x,&space;y)&space;=&space;xy" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(x,&space;y)&space;=&space;xy" title="g(x, y) = xy" /></a>.  Every time the engine executes an operation in the graph, the derivative of that operation is added to the graph to be executed later in the backward pass. Note, that the engine knows the derivatives of the basic functions.

In the example above, when multiplying x and y to obtain v, the engine will extend the graph to calculate the partial derivatives of the multiplication by using the multiplication derivative definition that it already knows. <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial}{\partial&space;x}&space;g(x,&space;y)&space;=&space;y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;x}&space;g(x,&space;y)&space;=&space;y" title="\frac{\partial}{\partial x} g(x, y) = y" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial}{\partial&space;y}&space;g(x,&space;y)&space;=&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;y}&space;g(x,&space;y)&space;=&space;x" title="\frac{\partial}{\partial y} g(x, y) = x" /></a> . The resulting extended graph is shown in Figure 2, where the *MultDerivative* node also calculates the product of the resulting gradients by an input gradient to apply the chain rule; this will be explicitly seen in the following operations. Note that the backward graph (green nodes) will not be executed until all the forward steps are completed.

<div class="text-center">
  <img src="{{ site.baseurl }}/assets/images/multi_derivative_graph.png" width="100%">
  <p>Figure 2: Computational graph extended after executing the logarithm</p>
</div>

Continuing, the engine now calculates the <a href="https://www.codecogs.com/eqnedit.php?latex=log(v)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log(v)" title="log(v)" /></a> operation and extends the graph again with the log derivative that it knows to be <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{v}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{v}" title="\frac{1}{v}" /></a>. This is shown in figure 3. This operation generates the result <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;w}{\partial&space;v}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;w}{\partial&space;v}" title="\frac{\partial w}{\partial v}" /></a> that when propagated backward and multiplied by the multiplication derivative as in the chain rule, generates the derivatives <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;w}{\partial&space;x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;w}{\partial&space;x}" title="\frac{\partial w}{\partial x}" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;w}{\partial&space;x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;w}{\partial&space;x}" title="\frac{\partial w}{\partial x}" /></a>.

<div class="text-center">
  <img src="{{ site.baseurl }}/assets/images/extended_computational_graph.png" width="100%">
  <p>Figure 3: Computational graph extended after executing the logarithm</p>
</div>

The original computation graph is extended with a new dummy variable z that is the same w. The derivative of z with respect to w is 1 as they are the same variable, this trick allows us to apply the chain rule to calculate the derivatives of the inputs. After the forward pass is complete, we start the backward pass, by supplying the initial value of 1.0 for <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;z}{\partial&space;w}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z}{\partial&space;w}" title="\frac{\partial z}{\partial w}" /></a>. This is shown in Figure 4.

<div class="text-center">
  <img src="{{ site.baseurl }}/assets/images/computational_graph_reverse_auto_differentiation.png" width="100%">
  <p>Figure 4: Computational graph extended for reverse auto differentiation</p>
</div>

Then following the green graph we execute the LogDerivative operation <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{v}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{v}" title="\frac{1}{v}" /></a> that the auto differentiation engine introduced, and multiply its result by <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;z}{\partial&space;w}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z}{\partial&space;w}" title="\frac{\partial z}{\partial w}" /></a> to obtain the gradient <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;z}{\partial&space;v}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z}{\partial&space;v}" title="\frac{\partial z}{\partial v}" /></a> as per the chain rule states. Next, the multiplication derivative is executed in the same way, and the desired derivatives <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;z}{\partial&space;x}&space;\frac{\partial&space;z}{\partial&space;y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z}{\partial&space;x}&space;\frac{\partial&space;z}{\partial&space;y}" title="\frac{\partial z}{\partial x} \frac{\partial z}{\partial y}" /></a> are finally obtained.

Formally, what we are doing here, and PyTorch autograd engine also does, is computing a Jacobian-vector product (Jvp) to calculate the gradients of the model parameters, since the model parameters and inputs are vectors.

**The Jacobian-vector product**

When we calculate the gradient of a vector-valued function <a href="https://www.codecogs.com/eqnedit.php?latex=f(\overline{x})&space;=&space;\overline{y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(\overline{x})&space;=&space;\overline{y}" title="f(\overline{x}) = \overline{y}" /></a> (a function whose inputs and outputs are vectors), we are essentially constructing a Jacobian matrix .

Thanks to the chain rule, multiplying the Jacobian matrix of a function <a href="https://www.codecogs.com/eqnedit.php?latex=f(\overline{x})&space;=&space;\overline{y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(\overline{x})&space;=&space;\overline{y}" title="f(\overline{x}) = \overline{y}" /></a> by a vector <a href="https://www.codecogs.com/eqnedit.php?latex=v" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v" title="v" /></a> with the previously calculated gradients of a scalar function <a href="https://www.codecogs.com/eqnedit.php?latex=z&space;=&space;g(\overline{y})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z&space;=&space;g(\overline{y})" title="z = g(\overline{y})" /></a> results in the gradients <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;z}{\partial&space;x_1}&space;\cdots&space;\frac{\partial&space;z}{\partial&space;x_n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z}{\partial&space;x_1}&space;\cdots&space;\frac{\partial&space;z}{\partial&space;x_n}" title="\frac{\partial z}{\partial x_1} \cdots \frac{\partial z}{\partial x_n}" /></a> of the scalar output with respect to the vector-valued function inputs.

As an example, let’s look at some functions in python notation to show how the chain rule applies.
<div class="outlined-code-block">
  <a href="https://www.codecogs.com/eqnedit.php?latex=f(x_1,&space;x_2)&space;=&space;(log(x_1&space;x_2),&space;sin(x_2))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x_1,&space;x_2)&space;=&space;(log(x_1&space;x_2),&space;sin(x_2))" title="f(x_1, x_2) = (log(x_1 x_2), sin(x_2))" /></a>

  <pre>def f(x1, x2):
      a = x1 * x2
      y1 = log(a)
      y2 = sin(x2)
      return (y1, y2)
  </pre>

  <a href="https://www.codecogs.com/eqnedit.php?latex=g(y_1,&space;y_2)&space;=&space;y_1&space;y_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(y_1,&space;y_2)&space;=&space;y_1&space;y_2" title="g(y_1, y_2) = y_1 y_2" /></a>

  <pre>def g(y1, y2):
      return y1 * y2
  </pre>

</div>

Now, if we derive this by hand using the chain rule and the definition of the derivatives, we obtain the following set of identities that we can directly plug into the Jacobian matrix of <a href="https://www.codecogs.com/eqnedit.php?latex=f(x_1,&space;x_2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x_1,&space;x_2)" title="f(x_1, x_2)" /></a>

<div class="outlined-code-block">
  <p><a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;y_1}{\partial&space;x_1}&space;=&space;\frac{\partial&space;y_1}{\partial&space;a}\frac{\partial&space;a}{\partial&space;x_1}&space;=&space;\frac{1}{x_1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;y_1}{\partial&space;x_1}&space;=&space;\frac{\partial&space;y_1}{\partial&space;a}\frac{\partial&space;a}{\partial&space;x_1}&space;=&space;\frac{1}{x_1}" title="\frac{\partial y_1}{\partial x_1} = \frac{\partial y_1}{\partial a}\frac{\partial a}{\partial x_1} = \frac{1}{x_1}" /></a></p>
  <p><a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;y_1}{\partial&space;x_2}&space;=&space;\frac{\partial&space;y_1}{\partial&space;a}\frac{\partial&space;a}{\partial&space;x_2}&space;=&space;\frac{1}{x_2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;y_1}{\partial&space;x_2}&space;=&space;\frac{\partial&space;y_1}{\partial&space;a}\frac{\partial&space;a}{\partial&space;x_2}&space;=&space;\frac{1}{x_2}" title="\frac{\partial y_1}{\partial x_2} = \frac{\partial y_1}{\partial a}\frac{\partial a}{\partial x_2} = \frac{1}{x_2}" /></a></p>
  <p><a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;y_2}{\partial&space;x_1}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;y_2}{\partial&space;x_1}&space;=&space;0" title="\frac{\partial y_2}{\partial x_1} = 0" /></a></p>
  <p><a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;y_2}{\partial&space;x_2}&space;=&space;cos(x_2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;y_2}{\partial&space;x_2}&space;=&space;cos(x_2)" title="\frac{\partial y_2}{\partial x_2} = cos(x_2)" /></a></p>
</div>

Next, let’s consider the gradients for the scalar function <a href="https://www.codecogs.com/eqnedit.php?latex=z&space;=&space;g(y_1,&space;y_2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z&space;=&space;g(y_1,&space;y_2)" title="z = g(y_1, y_2)" /></a>

<div class="outlined-code-block">
  <p><a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;z}{y_1}&space;=&space;y_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z}{y_1}&space;=&space;y_2" title="\frac{\partial z}{y_1} = y_2" /></a></p>
  <p><a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;z}{y_2}&space;=&space;y_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z}{y_2}&space;=&space;y_1" title="\frac{\partial z}{y_2} = y_1" /></a></p>
</div>

If we now calculate the transpose-Jacobian vector product obeying the chain rule, we obtain the following expression:
<div style="overflow:scroll">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\begin{pmatrix}\frac{\partial&space;y_1}{\partial&space;x_1}&space;&&space;\frac{\partial&space;y_1}{\partial&space;x_2}&space;\\\\&space;\frac{\partial&space;y_2}{\partial&space;x_1}&space;&&space;\frac{\partial&space;y_2}{\partial&space;x_2}&space;\end{pmatrix}^{t}&space;\begin{pmatrix}&space;y_2\\y_1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}\frac{1}{x_1}&space;&&space;\frac{1}{x_2}&space;\\\\&space;0&space;&&space;cos(x_2))&space;\end{pmatrix}^{t}&space;\begin{pmatrix}&space;y_2\\y_1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;\frac{1}{x_1}y_2\\\frac{1}{x_2}y_2&plus;cos(x_2)y_1&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{pmatrix}\frac{\partial&space;y_1}{\partial&space;x_1}&space;&&space;\frac{\partial&space;y_1}{\partial&space;x_2}&space;\\\\&space;\frac{\partial&space;y_2}{\partial&space;x_1}&space;&&space;\frac{\partial&space;y_2}{\partial&space;x_2}&space;\end{pmatrix}^{t}&space;\begin{pmatrix}&space;y_2\\y_1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}\frac{1}{x_1}&space;&&space;\frac{1}{x_2}&space;\\\\&space;0&space;&&space;cos(x_2))&space;\end{pmatrix}^{t}&space;\begin{pmatrix}&space;y_2\\y_1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;\frac{1}{x_1}y_2\\\frac{1}{x_2}y_2&plus;cos(x_2)y_1&space;\end{pmatrix}" title="\begin{pmatrix}\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} \\\\ \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} \end{pmatrix}^{t} \begin{pmatrix} y_2\\y_1 \end{pmatrix} = \begin{pmatrix}\frac{1}{x_1} & \frac{1}{x_2} \\\\ 0 & cos(x_2)) \end{pmatrix}^{t} \begin{pmatrix} y_2\\y_1 \end{pmatrix} = \begin{pmatrix} \frac{1}{x_1}y_2\\\frac{1}{x_2}y_2+cos(x_2)y_1 \end{pmatrix}" /></a>
</div>

Evaluating the Jvp for <a href="https://www.codecogs.com/eqnedit.php?latex=(x_1,&space;x_2)&space;=&space;(0.5,&space;0.75)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(x_1,&space;x_2)&space;=&space;(0.5,&space;0.75)" title="(x_1, x_2) = (0.5, 0.75)" /></a> yields the result:
<a href="https://www.codecogs.com/eqnedit.php?latex=(\frac{dy}{x_1},&space;\frac{dy}{x_2})&space;=&space;(1.3633,&space;0.1912)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\frac{dy}{x_1},&space;\frac{dy}{x_2})&space;=&space;(1.3633,&space;0.1912)" title="(\frac{dy}{x_1}, \frac{dy}{x_2}) = (1.3633, 0.1912)" /></a>
We can execute the same expression in PyTorch and calculate the gradient of the input:
<div class="outlined-code-block">
  <pre>>>> import torch</pre>
  <pre>>>> x = torch.tensor([0.5, 0.75], requires_grad=True)</pre>
  <pre>>>> y = torch.log(x[0] * x[1]) * torch.sin(x[1])</pre>
  <pre>>>> y.backward(1.0)</pre>
  <pre>>>> x.grad</pre>
  tensor([1.3633,
          0.1912])</pre>
</div>

The result is the same as our hand-calculated Jacobian-vector product!
However, PyTorch never constructed the matrix as it could grow prohibitively large but instead, created a graph of operations that traversed backward while applying the Jacobian-vector products defined in [tools/autograd/derivatives.yaml](https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml).

**Going through the graph**

Every time PyTorch executes an operation, the autograd engine constructs the graph to be traversed backward.
The reverse mode auto differentiation starts by adding a scalar variable at the end <a href="https://www.codecogs.com/eqnedit.php?latex=z&space;=&space;w" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z&space;=&space;w" title="z = w" /></a> so that <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;z}{\partial&space;w}&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z}{\partial&space;w}&space;=&space;1" title="\frac{\partial z}{\partial w} = 1" /></a> as we saw in the introduction. This is the initial gradient value that is supplied to the Jvp engine calculation as we saw in the section above.

In PyTorch, the initial gradient is explicitly set by the user when he calls the backward method.

Then, the Jvp calculation starts but it never constructs the matrix. Instead, when PyTorch records the computational graph, the derivatives of the executed forward operations are added (Backward Nodes). Figure 5 shows a backward graph generated by the execution of the functions <a href="https://www.codecogs.com/eqnedit.php?latex=f(x_1,&space;x_2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x_1,&space;x_2)" title="f(x_1, x_2)" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=g(y_1,&space;y_2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(y_1,&space;y_2)" title="g(y_1, y_2)" /></a> seen before.

<div class="text-center">
  <img src="{{ site.baseurl }}/assets/images/computational_graph_backward_pass.png" width="100%">
  <p>Figure 5: Computational Graph extended with the backward pass</p>
</div>

Once the forward pass is done, the results are used in the backward pass where the derivatives in the computational graph are executed. The basic derivatives are stored in the [tools/autograd/derivatives.yaml](https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml) file and they are not regular derivatives but the Jvp versions of them [3]. They take their primitive function inputs and outputs as parameters along with the gradient of the function outputs with respect to the final outputs. By repeatedly multiplying the resulting gradients by the next Jvp derivatives in the graph, the gradients up to the inputs will be generated following the chain rule.

<div class="text-center">
  <img src="{{ site.baseurl }}/assets/images/chain_rule_backward_differentiation.png" width="100%">
  <p>Figure 6: How the chain rule is applied in backward differentiation</p>
</div>

Figure 6 represents the process by showing the chain rule. We started with a value of 1.0 as detailed before which is the already calculated gradient <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;y}{\partial&space;u}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;y}{\partial&space;u}" title="\frac{\partial y}{\partial u}" /></a> highlighted in green. And we move to the next node in the graph. The *backward* function registered in [derivatives.yaml](https://github.com/pytorch/pytorch/blob/a0a7a2d648f05b0192e6943c9684406cdf404fbf/tools/autograd/derivatives.yaml#L635-L636) will calculate the associated
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;u}{\partial&space;v}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;u}{\partial&space;v}" title="\frac{\partial u}{\partial v}" /></a> value highlighted in red and multiply it by <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;y}{\partial&space;u}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;y}{\partial&space;u}" title="\frac{\partial y}{\partial u}" /></a>. By the chain rule this results in <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;y}{\partial&space;v}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;y}{\partial&space;v}" title="\frac{\partial y}{\partial v}" /></a> which will be the already calculated gradient (green) when we process the next backward node in the graph.

You may also have noticed that in Figure 5 there is a gradient generated from two different sources. When two different functions share an input, the gradients with respect to the output are aggregated for that input, and calculations using that gradient can’t proceed unless all the paths have been aggregated together.

Let’s see an example of how the derivatives are stored in PyTorch.

Suppose that we are currently processing the backward propagation of the <a href="https://www.codecogs.com/eqnedit.php?latex=log" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log" title="log" /></a> function, in the *LogBackward* node in Figure 2.  The derivative of <a href="https://www.codecogs.com/eqnedit.php?latex=log" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log" title="log" /></a> in [`derivatives.yaml`](https://github.com/pytorch/pytorch/blob/a0a7a2d648f05b0192e6943c9684406cdf404fbf/tools/autograd/derivatives.yaml#L635-L636) is specified as `grad.div(self.conj())`. `grad` is the already calculated gradient <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;z}{\partial&space;y1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z}{\partial&space;y1}" title="\frac{\partial z}{\partial y1}" /></a> and `self.conj()` is the complex conjugate of the input vector. For complex numbers PyTorch calculates a special derivative called the conjugate Wirtinger derivative [6]. This derivative takes the complex number and its conjugate and by operating some magic that is described in [6], they are the direction of steepest descent when plugged into optimizers.

This code translates to <a href="https://www.codecogs.com/eqnedit.php?latex=(\frac{\partial&space;z}{\partial&space;y1}&space;\frac{1}{v})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\frac{\partial&space;z}{\partial&space;y1}&space;\frac{1}{v})" title="(\frac{\partial z}{\partial y1} \frac{1}{v})" /></a>, the corresponding green, and red squares in Figure 3. Continuing, the autograd engine will execute the next operation; backward of the multiplication. As before, the inputs are the original function’s inputs and the gradient calculated from the <a href="https://www.codecogs.com/eqnedit.php?latex=log" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log" title="log" /></a> backward step. This step will keep repeating until we reach the gradient with respect to the inputs and the computation will be finished. The gradient of <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;z}{x2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z}{x2}" title="\frac{\partial z}{x2}" /></a> is only completed once the multiplication and sin gradients are added together. As you can see, we computed the equivalent of the Jvp but without constructing the matrix.

In the next post we will dive inside PyTorch code to see how this graph is constructed and where are the relevant pieces should you want to experiment with it!

### References

<ol class="reference-list">
  <li><a href="https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html">https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html</a></li>
  <li><a href="https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf">https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf</a></li>
  <li><a href="https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf">https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf</a></li>
  <li><a href="https://mustafaghali11.medium.com/how-pytorch-backward-function-works-55669b3b7c62">https://mustafaghali11.medium.com/how-pytorch-backward-function-works-55669b3b7c62</a></li>
  <li><a href="https://indico.cern.ch/event/708041/contributions/3308814/attachments/1813852/2963725/automatic_differentiation_and_deep_learning.pdf">https://indico.cern.ch/event/708041/contributions/3308814/attachments/1813852/2963725/automatic_differentiation_and_deep_learning.pdf</a></li>
  <li><a href="https://pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc">https://pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc</a></li>
  <p>Recommended: shows why the backprop is formally expressed with the Jacobian</p>
  <li><a href="https://www.cs.ubc.ca/~fwood/CS340/lectures/AD1.pdf">https://cs.ubc.ca/~fwood/CS340/lectures/AD1.pdf</a></li>
</ol>
