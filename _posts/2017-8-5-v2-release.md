---
layout: post
title: "[0.2] Higher order gradients, Distributed PyTorch, Broadcasting, Advanced Indexing, New Layers and more"
author: "The PyTorch Team"
date: 2017-08-05 12:00:00 -0500
---

Here comes the next major release of PyTorch, just in time for ICML.

We're introducing long-awaited features such as Broadcasting, Advanced Indexing, Higher-order gradients and finally: Distributed PyTorch.

Important: Due to introducing Broadcasting, the code behavior for certain broadcastable situations is different from behavior in 0.1.12. We've provided easy ways of identifying this ambiguous code in the [Important Breakages and Workarounds]() section.

Table of contents:
- Tensor Broadcasting (numpy-style)
- Advanced Indexing for Tensors and Variables
- Higher-order gradients
- Distributed PyTorch (multi-node training, etc.)
- Neural Network layers and features: SpatialTransformers, WeightNorm, EmbeddingBag, etc.
- New in torch and autograd: matmul, inverse, etc.
- Easier debugging, better error messages
- Bug Fixes
- **Important Breakages and Workarounds**

## Tensor Broadcasting (numpy-style)

In short, if a PyTorch operation supports broadcasting, then its Tensor arguments can be automatically expanded to be of equal sizes (without making copies of the data).

PyTorch Broadcasting semantics [closely follow numpy-style broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#module-numpy.doc.broadcasting); if you are familiar with numpy broadcasting, things should just work as expected.

### General Semantics

Two tensors are “broadcastable” if the following rules hold:
- Each tensor has at least one dimension.
- When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

For Example:

```python
>>> x=torch.FloatTensor(5,7,3)
>>> y=torch.FloatTensor(5,7,3)
# same shapes are always broadcastable (i.e. the above rules always hold)

# can line up trailing dimensions
>>> x=torch.FloatTensor(5,3,4,1)
>>> y=torch.FloatTensor(  3,1,1)

# x and y are broadcastable.
# 1st trailing dimension: both have size 1
# 2nd trailing dimension: y has size 1
# 3rd trailing dimension: x size == y size
# 4th trailing dimension: y dimension doesn't exist

# but:
>>> x=torch.FloatTensor(5,2,4,1)
>>> y=torch.FloatTensor(  3,1,1)
# x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3
```

If two tensors x, y are "broadcastable", the resulting tensor size is calculated as follows:
- If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length.
- Then, for each dimension size, the resulting dimension size is the max of the sizes of x and y along that dimension.

For Example:

```python
# can line up trailing dimensions to make reading easier
>>> x=torch.FloatTensor(5,1,4,1)
>>> y=torch.FloatTensor(  3,1,1)
>>> (x+y).size()
torch.Size([5, 3, 4, 1])

# error case
>>> x=torch.FloatTensor(5,2,4,1)
>>> y=torch.FloatTensor(  3,1,1)
>>> (x+y).size()
RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
```

More details [can be found on the PyTorch documentation site](http://pytorch.org/docs/0.2.0/notes/broadcasting.html).  Also, each torch function lists its broadcasting semantics in the documentation.

## Advanced Indexing for Tensors and Variables

PyTorch now supports a subset of NumPy style [advanced indexing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing). This allows users to select arbitrary indices at each dimension of the Tensor, including non-adjacent indices and duplicate indices, using the same `[]`-style operation. This allows for a more flexible indexing strategy without needing calls to PyTorch's `Index[Select, Add, ...]`  functions.

Let's look at some examples:

```python
x = torch.Tensor(5, 5, 5)
```

**Pure Integer Array Indexing - specify arbitrary indices at each dimension**

```python
x[[1, 2], [3, 2], [1, 0]]
--> yields a 2-element Tensor (x[1][3][1], x[2][2][0])
```

**also supports broadcasting, duplicates**

```python
x[[2, 3, 2], [0], [1]]
--> yields a 3-element Tensor (x[2][0][1], x[3][0][1], x[2][0][1])
```

**arbitrary indexer shapes allowed**

```python
x[[[1, 0], [0, 1]], [0], [1]].shape
--> yields a 2x2 Tensor [[x[1][0][1], x[0][0][1]],
                         [x[0][0][1], x[1][0][1]]]
```

**can use colon, ellipse**

```python
x[[0, 3], :, :]
x[[0, 3], ...]
--> both yield a 2x5x5 Tensor [x[0], x[3]]
```

**also use Tensors to index!**

```python
y = torch.LongTensor([0, 2, 4])
x[y, :, :]
--> yields a 3x5x5 Tensor [x[0], x[2], x[4]]
```

**selection with less than ndim, note the use of comma**

```python
x[[1, 3], ]
--> yields a 2x5x5 Tensor [x[1], x[3]]
```

## Higher order gradients

Now you can evaluate higher order differentials in PyTorch. For example, you can compute Hessian-Vector products, penalize the norm of the gradients of your model, implement Unrolled GANs and Improved WGANs, etc.

In the `0.2` release, we've enabled the ability to compute higher order gradients for all of `torch.XXX` functions and the most popular `nn`layers. The rest will be covered in the next release.

Here's a short example that penalizes the norm of the weight gradients of a Resnet-18 model, so that the volume of weights is slow-changing.

```python
import torch
from torchvision.models import resnet18
from torch.autograd import Variable

model = resnet18().cuda()

# dummy inputs for the example
input = Variable(torch.randn(2,3,224,224).cuda(), requires_grad=True)
target = Variable(torch.zeros(2).long().cuda())

# as usual
output = model(input)
loss = torch.nn.functional.nll_loss(output, target)

grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
# torch.autograd.grad does not accumuate the gradients into the .grad attributes
# It instead returns the gradients as Variable tuples.

# now compute the 2-norm of the grad_params
grad_norm = 0
for grad in grad_params:
    grad_norm += grad.pow(2).sum()
grad_norm = grad_norm.sqrt()

# take the gradients wrt grad_norm. backward() will accumulate
# the gradients into the .grad attributes
grad_norm.backward()

# do an optimization step
optimizer.step()
```

We see two new concepts here:

1. [torch.autograd.grad](http://pytorch.org/docs/master/autograd.html#torch.autograd.grad) is a function that takes in [outputs, list of inputs (for which you want gradients)], and returns the gradients wrt. these inputs as a tuple, rather than accumulating the gradients into the `.grad` attributes. This is useful if you want to further operate on the gradients.
2. You can operate on the gradients, and call `backward()` on them.

The list of `nn` layers that support higher order gradients are:
- `AvgPool*d`, `BatchNorm*d`, `Conv*d`, `MaxPool1d,2d`, `Linear`, `Bilinear`
- `pad`, `ConstantPad2d`, `ZeroPad2d`, `LPPool2d`,  `PixelShuffle`
- `ReLU6`, `LeakyReLU`, `PReLU`, `Tanh`, `Tanhshrink`, `Threshold`, `Sigmoid`, `HardTanh`, `ELU`, `Softsign`, `SeLU`
- `L1Loss`, `NLLLoss`, `PoissonNLLLoss`, `LogSoftmax`, `Softmax2d`
The rest will be enabled in the next release.

To enable higher order gradients, we've introduced a new style of writing `autograd.Function` (the current/old style of writing functions is fully backward compatible). [You can read more about the new style of functions here](http://pytorch.org/docs/0.2.0/notes/extending.html).

Most of you dont write your own `autograd.Function`s, they are low-level primitives that introduce
new operations to the autograd engine, where you specify the forward and backward calls.

## Distributed PyTorch

We introduce the [torch.distributed](http://pytorch.org/docs/0.2.0/distributed.html) package that allows you to exchange Tensors among multiple machines. Using this package, you can scale your network training over multiple machines and larger mini-batches. For example, you are given the primitives to implement [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).

The `distributed` package follows an MPI-style programming model. This means that there are functions provided to you such as `send`, `recv`, `all_reduce` that will exchange Tensors among nodes (machines).

For each of the machines to first identify each other and assign unique numbers to each other (ranks), we provide simple initialization methods:
- shared file system (requires that all processes can access a single file system)
- IP multicast (requires that all processes are in the same network)
- environment variable (requires you to manually assign ranks and know an address of a node reachable from all processes)

Our package documentation contains more details on initialization and available backends, but here's an example of initializing using a multicast address:

```python
import torch.distributed as dist

dist.init_process_group(backend='tcp',
                        init_method='tcp://[ff15:1e18:5d4c:4cf0:d02d:b659:53ba:b0a7]:23456',
                        world_size=4)

print('Hello from process {} (out of {})!'.format(
        dist.get_rank(), dist.get_world_size()))
```

This would print `Hello from process 2 (out of 4)`on the 3rd machine.

World size is the number of processes that will participate in the job. Each will be assigned a rank, which is a number between 0 and world_size - 1, unique within this job. It will serve as a process identifier and will be used instead of an address to, for example, specify to which process should a tensor be sent.

Here's a snippet that shows how simple point-to-point communication can be performed:

```python
# All processes (receiving ones too!) need to have tensors of appropriate
# size preallocated.
x = torch.Tensor(10)
if dist.get_rank() == 0:
    x.normal_()
    # Send x to process with rank 1
    dist.send(x, dst=1)
else:  # rank == 1
    # Receive data from process with rank 0 and save result in x
    dist.recv(x, src=0)
```

Asynchronous p2p functions (`isend`, `irecv`) are available too.

However, some communication patterns appear so often that more efficient collective calls have been developed. They typically engage the whole process group and are much faster than naive algorithms using `send`/`recv`. One example is `all_reduce`:

```python
x = torch.Tensor([dist.get_rank()])
# Add tensors from all processes such that they all receive the result.
# x is an input and output to this operation.
dist.all_reduce(x)
```

The distributed package is fairly low-level, so that it allows to implement more advanced algorithms and tailor the code to very specific purposes, but data-parallel training is such a common one that we have created high-level helpers for it.

Hence, we've introduced `DistributedDataParallel`, which is meant to be a nearly drop-in replacement for nn.DataParallel.
Here's a code snippet demonstrating changes necessary to add it to existing training code:

```python
# Wrap model in DistributedDataParallel (CUDA only for the moment)
model = torch.nn.parallel.DistributedDataParallel(model.cuda())

# Use a DistributedSampler to restrict each process to a distinct subset
# of the dataset.
train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.workers,
    pin_memory=True, sampler=train_sampler)

for epoch in range(args.num_epochs):
    # Use .set_epoch() method to reshuffle the dataset partition at every iteration
    train_sampler.set_epoch(epoch)
    # training loop
    ...
```

You can see a fuller [Imagenet training example here](https://github.com/pytorch/examples/tree/master/imagenet)

## New nn layers: SpatialTransformers, WeightNorm, EmbeddingBag, etc.

#### New features
- [forward_pre_hook](http://pytorch.org/docs/master/nn.html#torch.nn.Module.register_forward_pre_hook) is introduced to execute user-specified closures right before a forward function is called.
- Convenient access to non-leaf gradients:
Currently, to access and inspect gradients of intermediate values, we have to use `hooks`. This is not convenient for doing simple inspections. Hence, we introduce `retain_grad`. It is best explained via an example:

```python
input = Variable(torch.rand(1, 3), requires_grad=True)
h1 = input * 3
out = (h1 * h1).sum()

h1.retain_grad()
out.backward()

print(h1.grad)
# without calling retain_grad(), h1.grad is None
```
- DataParallel now supports dicts as inputs

#### New Layers

- Spatial Transformer Networks via `F.grid_sample` and `F.affine_grid`
- `nn.SeLU` and `nn.AlphaDropout` are introduced, from the paper: [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- `nn.GLU` (Gated Linear Unit) is introduced from the paper [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
- [Weight Normalization](https://arxiv.org/abs/1602.07868) is now implemented via [torch.utils.weight_norm](http://pytorch.org/docs/master/nn.html#torch.nn.utils.weight_norm).
- You can now ignore specific target indices while computing `cross_entropy_loss` and `nll_loss` using the `ignore_index` argument. This is a cheap and useful way of implementing masking, where you can have a `mask` index that is ignored in computing the loss.
- `F.normalize` implements dimension-wise renormalization
- `F.upsample` and `nn.Upsample` consolidate multiple Upsampling layers into one function. It implements 2d and 3d bilinear/trilinear/nearest upsampling.
- `nn.EmbeddingBag`: When build bag-of-words models, doing an `Embedding` followed by `Sum` or `Mean` is common. For variable length sequences, computing bags of embeddings involves masking. We provide a singe `nn.EmbeddingBag` which is much more efficent and faster to compute bags of embeddings, especially for variable length sequences.
- Numerically stable Binary Cross-Entropy loss via `bce_with_logits`
- A negative log-likelihood loss with Poisson distribution of the target via `PoissonNLLLoss`
- `cosine_similarity`: Returns cosine similarity between x1 and x2, computed along dim.

## New in torch and autograd

- All reduce functions such as `sum` and `mean`now default to squeezing the reduced dimension. For example `torch.sum(torch.randn(10, 20))` returns a 1D Tensor.
- `x.shape`, similar to numpy. A convenience `property` that is equivalent to `x.size()`
- `torch.matmul`, similar to np.matmul
- bitwise and, or, xor, lshift, rshift
- autograd support for `inverse`, `gesv`, `cumprod`, `atan2`
- unbiased `var` and `std` now available via keyword argument option
- `torch.scatter_add` - torch.scatter, except when duplicate indices are encountered, the values are summed.
- torch.median behaves similar to torch.sum when no arguments are given, i.e. it reduces all the dimensions and returns a single median value of the flattened Tensor.
- masked_copy_ has been renamed to masked_scatter_ (with deprecation on masked_copy_)
- torch.manual_seed now seeds all CUDA devices as well
- You can now specify the random number generator object via keyword arguments `torch.rand(1000, generator=gen)`

## Bug-fixes and small improvements

- Now we emit an error when a Variable is converted to a bool. For example:

```
b = Variable(torch.zeros(1))
if b[0]: # errors now
```

- Fix correctness bugs in qr decomposition on CUDA.
- Support for IBM PowerPC64 platform
- Check that the CuDNN version at compile-time is the same version at run-time.
- Improve error message in CUDA forked subprocess
- Faster transposed-copy on CPU
- Improve error messages in InstanceNorm
- Add more argument checking for various routines, especially BatchNorm and Convolution routines.
- Better error messages around shape reporting across the CPU backend.
- Support more than 8 GPUs per machine (work-around a CUDA p2p restriction)
- Improve error message when accessing attributes that don't exist
- t() of Variable consistent with Tensor
- prevent divide-by-zero when dropout p=1
- fix sharing of CUDA tensors on non-current devices
- when BN epsilon < allowed CuDNN value, fallback to THNN
- Fix thread-trashing when using different number of threads for MKL and OMP
- improve memory usage when using CuDNN RNN
- Fix ZeroPad2d backwards with negative padding
- add dummy tensor.data property, to provide interpretable error message to users
- Fix in-place division for Python3
- Raise error when call from_numpy on 0-dim array
- Empty Tensors dont error out when shared across multiprocessing
- fix baddbmm for expanded tensors
- Let parallel_apply accept arbitrary inputs
- keyword arguments in Tensor and Variable are now consistent
- fix torch.inverse when Magma is not available
- Add logical not operator for ByteTensor
- add device asserts in scatter/gather kernels

## Important Breakages and Workarounds

As you've read, we've introduced two important changes that are not
backward compatible:
- Numpy-style Broadcasting
- Reduction functions such as `sum(1)` now default to `keepdim=False`

We provide different levels of Python warnings that you can enable to alert you if you are using deprecated behavior or if the behavior of your code has changed.

#### tl;dr
Here is a code snippet that you can add to the top of your scripts.
Adding this code will generate warnings highlighting incompatible code.

Fix your code to no longer generate warnings.

```python
# insert this to the top of your scripts (usually main.py)
import sys, warnings, traceback, torch
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
    traceback.print_stack(sys._getframe(2))
warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
```
Once all warnings disappear, you can remove the code snippet.

#### More elaborately

Now, let us see the three incompatible changes with examples.

##### Using the (now deprecated) 1-dimensional view pointwise function

Prior versions of PyTorch allowed certain pointwise functions to execute on tensors with different shapes, as long as the number of elements in each tensor was equal.  The pointwise operation would then be carried out by viewing each tensor as 1-dimensional. PyTorch now supports broadcasting. The “1-dimensional” pointwise behavior is considered deprecated and will generate a Python warning in cases where tensors are not broadcastable, but have the same number of elements.

For example:

```python
>>> torch.add(torch.ones(4), torch.ones(2,2))
__main__:1: UserWarning: self and other not broadcastable, but have the same
number of elements.  Falling back to deprecated pointwise behavior.
2
2
2
2
[torch.FloatTensor of size 4]
```

##### Broadcasting in code where it didn't happen before
The introduction of broadcasting can cause backwards incompatible changes in the case where two tensors do not have the same shape,
but are broadcastable and have the same number of elements.

For example:

```python
>>> torch.add(torch.ones(4,1), torch.randn(4))
```

would previously produce a Tensor with size: `torch.Size([4,1])`,
but now produces a Tensor with size: `torch.Size([4,4])`.

In order to help identify cases in your code where backwards incompatibilities introduced by broadcasting may exist, you may set `torch.utils.backcompat.broadcast_warning.enabled` to `True`, which will generate a python warning in such cases.

For Example:

```python
>>> torch.utils.backcompat.broadcast_warning.enabled=True
>>> torch.add(torch.ones(4,1), torch.ones(4))
__main__:1: UserWarning: self and other do not have the same shape, but are broadcastable, and have the same number of elements.
```
Note that this setting can trigger warnings for valid uses of broadcasting (including in library code), so you probably want to turn this warning off after migrating your code.

##### KeepDim=False for Reduction Functions

To get a warning when using a dimensional reduction function with the default keepdim argument, set `torch.utils.backcompat.keepdim_warning.enabled` to `True`.  For example:

```python
>>> torch.sum(torch.ones(2,3), 1)
__main__:1: UserWarning: backwards compatibility: call to "sum" uses default value for keepdim which has changed default to False.  Consider passing as kwarg.
3
3
[torch.FloatTensor of size 2]
```

As with `torch.utils.backcompat.broadcast_warning.enabled`, this warning can trigger from valid code, so you most likely want to disable this warning after migrating your code.

Note also that using `keepdim=False` can cause your existing code to "just work" with broadcasting.  For example:

```python
# behavior with (old) keepdim=True, causes accidental broadcast
>>> torch.add(torch.ones(4), torch.ones(4,4).sum(dim=1, keepdim=True))
5  5  5  5
5  5  5  5
5  5  5  5
5  5  5  5
[torch.FloatTensor of size 4x4]

# new behavior with keepdim=False is equivalent to non-broadcasted result
>>> torch.add(torch.ones(4), torch.ones(4,4).sum(dim=1, keepdim=False))
5
5
5
5
[torch.FloatTensor of size 4]
```
