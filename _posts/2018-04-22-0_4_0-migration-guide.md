---
layout: post
title: "PyTorch 0.4.0 Migration Guide"
author: The PyTorch Team
date: 2018-04-22 12:00:00 -0500
---

Welcome to the migration guide for PyTorch 0.4.0. In this release we introduced [many exciting new features and critical bug fixes](https://github.com/pytorch/pytorch/releases/tag/v0.4.0), with the goal of providing users a better and cleaner interface. In this guide, we will cover the most important changes in migrating existing code from previous versions:
* ``Tensors`` and ``Variables`` have merged
* Support for 0-dimensional (scalar) ``Tensors``
* Deprecation of the ``volatile`` flag
* ``dtypes``, ``devices``, and Numpy-style ``Tensor`` creation functions
* Writing device-agnostic code
* New edge-case constraints on names of submodules, parameters, and buffers in ``nn.Module``


## Merging [``Tensor``](http://pytorch.org/docs/0.4.0/tensors.html) and ``Variable`` and classes

[``torch.Tensor``](http://pytorch.org/docs/0.4.0/tensors.html) and ``torch.autograd.Variable`` are now the same class.  More precisely, [``torch.Tensor``](http://pytorch.org/docs/0.4.0/tensors.html) is capable of tracking history and behaves like the old ``Variable``; ``Variable`` wrapping continues to work as before but returns an object of type [``torch.Tensor``](http://pytorch.org/docs/0.4.0/tensors.html).  This means that you don't need the ``Variable`` wrapper everywhere in your code anymore.

### The `type()` of a [``Tensor``](http://pytorch.org/docs/0.4.0/tensors.html) has changed

Note also that the ``type()`` of a Tensor no longer reflects the data type. Use ``isinstance()`` or ``x.type()`` instead:

```python
>>> x = torch.DoubleTensor([1, 1, 1])
>>> print(type(x))  # was torch.DoubleTensor
"<class 'torch.Tensor'>"
>>> print(x.type())  # OK: 'torch.DoubleTensor'
'torch.DoubleTensor'
>>> print(isinstance(x, torch.DoubleTensor))  # OK: True
True
```

### When does [``autograd``](http://pytorch.org/docs/0.4.0/autograd.html) start tracking history now?

``requires_grad``, the central flag for [``autograd``](http://pytorch.org/docs/0.4.0/autograd.html), is now an attribute on ``Tensors``.  The same rules previously used for ``Variables`` applies to ``Tensors``; [``autograd``](http://pytorch.org/docs/0.4.0/autograd.html) starts tracking history when any input ``Tensor`` of an operation has ``requires_grad=True``. For example,

```python
>>> x = torch.ones(1)  # create a tensor with requires_grad=False (default)
>>> x.requires_grad
False
>>> y = torch.ones(1)  # another tensor with requires_grad=False
>>> z = x + y
>>> # both inputs have requires_grad=False. so does the output
>>> z.requires_grad
False
>>> # then autograd won't track this computation. let's verify!
>>> z.backward()
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
>>>
>>> # now create a tensor with requires_grad=True
>>> w = torch.ones(1, requires_grad=True)
>>> w.requires_grad
True
>>> # add to the previous result that has require_grad=False
>>> total = w + z
>>> # the total sum now requires grad!
>>> total.requires_grad
True
>>> # autograd can compute the gradients as well
>>> total.backward()
>>> w.grad
tensor([ 1.])
>>> # and no computation is wasted to compute gradients for x, y and z, which don't require grad
>>> z.grad == x.grad == y.grad == None
True
```

#### Manipulating ``requires_grad`` flag

Other than directly setting the attribute, you can change this flag **in-place** using [``my_tensor.requires_grad_()``](http://pytorch.org/docs/0.4.0/tensors.html#torch.Tensor.requires_grad_), or, as in the above example, at creation time by passing it in as an argument (default is ``False``), e.g.,

```python
>>> existing_tensor.requires_grad_()
>>> existing_tensor.requires_grad
True
>>> my_tensor = torch.zeros(3, 4, requires_grad=True)
>>> my_tensor.requires_grad
True
```

### What about ``.data``?

``.data`` was the primary way to get the underlying ``Tensor`` from a ``Variable``. After this merge, calling ``y = x.data`` still has similar semantics. So ``y`` will be a ``Tensor`` that shares the same data with ``x``, is unrelated with the computation history of ``x``, and has ``requires_grad=False``.

However, ``.data`` can be unsafe in some cases. Any changes on ``x.data`` wouldn't be tracked by ``autograd``, and the computed gradients would be incorrect if ``x`` is needed in a backward pass. A safer alternative is to use [``x.detach()``](http://pytorch.org/docs/master/autograd.html#torch.Tensor.detach), which also returns a ``Tensor`` that shares data with ``requires_grad=False``, but will have its in-place changes reported by ``autograd`` if ``x`` is needed in backward.

Here is an example of the difference between ``.data`` and ``x.detach()`` (and why we recommend using ``detach`` in general).

If you use ``Tensor.detach()``, the gradient computation is guaranteed to be correct.

```
>>> a = torch.tensor([1,2,3.], requires_grad = True)
>>> out = a.sigmoid()
>>> c = out.detach()
>>> c.zero_()
tensor([ 0.,  0.,  0.])

>>> out  # modified by c.zero_() !!
tensor([ 0.,  0.,  0.])

>>> out.sum().backward()  # Requires the original value of out, but that was overwritten by c.zero_()
RuntimeError: one of the variables needed for gradient computation has been modified by an 
```

However, using ``Tensor.data`` can be unsafe and can easly result in incorrect gradients 
when a tensor is required for gradient computation but modified in-place.

```
>>> a = torch.tensor([1,2,3.], requires_grad = True)
>>> out = a.sigmoid()
>>> c = out.data
>>> c.zero_()
tensor([ 0.,  0.,  0.])

>>> out  # out  was modified by c.zero_()
tensor([ 0.,  0.,  0.])

>>> out.sum().backward()
>>> a.grad  # The result is very, very wrong because `out` changed!
tensor([ 0.,  0.,  0.])
```

## Support for 0-dimensional (scalar) Tensors

Previously, indexing into a ``Tensor`` vector (1-dimensional tensor) gave a Python number but indexing into a ``Variable`` vector gave (incosistently!) a vector of size ``(1,)``!  Similar behavior existed with reduction functions, e.g. `tensor.sum()` would return a Python number, but `variable.sum()` would retun a vector of size `(1,)`.

Fortunately, this release introduces proper scalar (0-dimensional tensor) support in PyTorch!  Scalars can be created using the new `torch.tensor` function (which will be explained in more detail later; for now just think of it as the PyTorch equivalent of `numpy.array`).  Now you can do things like:

```python
>>> torch.tensor(3.1416)         # create a scalar directly
tensor(3.1416)
>>> torch.tensor(3.1416).size()  # scalar is 0-dimensional
torch.Size([])
>>> torch.tensor([3]).size()     # compare to a vector of size 1
torch.Size([1])
>>>
>>> vector = torch.arange(2, 6)  # this is a vector
>>> vector
tensor([ 2.,  3.,  4.,  5.])
>>> vector.size()
torch.Size([4])
>>> vector[3]                    # indexing into a vector gives a scalar
tensor(5.)
>>> vector[3].item()             # .item() gives the value as a Python number
5.0
>>> mysum = torch.tensor([2, 3]).sum()
>>> mysum
tensor(5)
>>> mysum.size()
torch.Size([])
```

### Accumulating losses

Consider the widely used pattern ``total_loss += loss.data[0]``.  Before 0.4.0. ``loss`` was a ``Variable`` wrapping a tensor of size ``(1,)``, but in 0.4.0 ``loss`` is now a scalar and has ``0`` dimensions. Indexing into a scalar doesn't make sense (it gives a warning now, but will be a hard error in 0.5.0).  Use ``loss.item()`` to get the Python number from a scalar.

Note that if you don't convert to a Python number when accumulating losses, you may find increased memory usage in your program. This is because the right-hand-side of the above expression used to be a Python float, while it is now a zero-dim Tensor.  The total loss is thus accumulating Tensors and their gradient history, which may keep around large autograd graphs for much longer than necessary.


## Deprecation of ``volatile`` flag

The ``volatile`` flag is now deprecated and has no effect. Previously, any computation that involves a ``Variable`` with ``volatile=True`` wouldn't be tracked by ``autograd``. This has now been replaced by [a set of more flexible context managers](http://pytorch.org/docs/0.4.0/torch.html#locally-disabling-gradient-computation) including ``torch.no_grad()``, ``torch.set_grad_enabled(grad_mode)``, and others.

```python
>>> x = torch.zeros(1, requires_grad=True)
>>> with torch.no_grad():
...     y = x * 2
>>> y.requires_grad
False
>>>
>>> is_train = False
>>> with torch.set_grad_enabled(is_train):
...     y = x * 2
>>> y.requires_grad
False
>>> torch.set_grad_enabled(True)  # this can also be used as a function
>>> y = x * 2
>>> y.requires_grad
True
>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False
```


## [``dtypes``](http://pytorch.org/docs/0.4.0/tensor_attributes.html#torch.torch.dtype), [``devices``](http://pytorch.org/docs/0.4.0/tensor_attributes.html#torch.torch.device) and NumPy-style creation functions

In previous versions of PyTorch, we used to specify data type (e.g. float vs double), device type (cpu vs cuda) and layout (dense vs sparse) together as a "tensor type". For example, ``torch.cuda.sparse.DoubleTensor`` was the ``Tensor`` type respresenting the ``double`` data type, living on CUDA devices, and with [COO sparse tensor layout](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)).

In this release, we introduce [``torch.dtype``](http://pytorch.org/docs/0.4.0/tensor_attributes.html#torch.torch.dtype), [``torch.device``](http://pytorch.org/docs/0.4.0/tensor_attributes.html#torch.torch.device) and [``torch.layout``](http://pytorch.org/docs/0.4.0/tensor_attributes.html#torch.torch.layout) classes to allow better management of these properties via NumPy-style creation functions.

### [``torch.dtype``](http://pytorch.org/docs/0.4.0/tensor_attributes.html#torch.torch.dtype)

Below is a complete list of available [``torch.dtype``](http://pytorch.org/docs/0.4.0/tensor_attributes.html#torch.torch.dtype)s (data types) and their corresponding tensor types.

| Data type                 | ``torch.dtype``                        | Tensor types              |
|:------------------------- |:-------------------------------------- | :------------------------ |
| 32-bit floating point     | ``torch.float32`` or ``torch.float``   | ``torch.*.FloatTensor``   |
| 64-bit floating point     | ``torch.float64`` or ``torch.double``  | ``torch.*.DoubleTensor``  |
| 16-bit floating point     | ``torch.float16`` or ``torch.half``    | ``torch.*.HalfTensor``    |
| 8-bit integer (unsigned)  | ``torch.uint8``                        | ``torch.*.ByteTensor``    |
| 8-bit integer (signed)    | ``torch.int8``                         | ``torch.*.CharTensor``    |
| 16-bit integer (signed)   | ``torch.int16``   or ``torch.short``   | ``torch.*.ShortTensor``   |
| 32-bit integer (signed)   | ``torch.int32``   or ``torch.int``     | ``torch.*.IntTensor``     |
| 64-bit integer (signed)   | ``torch.int64``   or ``torch.long``    | ``torch.*.LongTensor``    |

The dtype of a tensor can be access via its ``dtype`` attribute.

<!---
Use [``torch.set_default_dtype``](http://pytorch.org/docs/0.4.0/torch.html#torch.set_default_dtype) and [``torch.get_default_dtype``](http://pytorch.org/docs/0.4.0/torch.html#torch.get_default_dtype) to manipulate default ``dtype`` for floating point tensors.
--->

### [``torch.device``](http://pytorch.org/docs/0.4.0/tensor_attributes.html#torch.torch.device)

A [``torch.device``](http://pytorch.org/docs/0.4.0/tensor_attributes.html#torch.torch.device) contains a device type (``'cpu'`` or ``'cuda'``) and optional device ordinal (id) for the device type. It can be initilized with ``torch.device('{device_type}')`` or ``torch.device('{device_type}:{device_ordinal}')``.

If the device ordinal is not present, this represents the current device for the device type; e.g., ``torch.device('cuda')`` is equivalent to ``torch.device('cuda:X')`` where ``X`` is the result of ``torch.cuda.current_device()``.

The device of a tensor can be accessed via its ``device`` attribute.

### [``torch.layout``](http://pytorch.org/docs/0.4.0/tensor_attributes.html#torch.torch.layout)

[``torch.layout``](http://pytorch.org/docs/0.4.0/tensor_attributes.html#torch.torch.layout) represents the data layout of a [``Tensor``](http://pytorch.org/docs/0.4.0/tensors.html). Currently``torch.strided`` (dense tensors, the default) and ``torch.sparse_coo`` (sparse tensors with COO format) are supported.

The layout of a tensor can be access via its ``layout`` attribute.

### Creating ``Tensors``

[Methods that create a ``Tensor``](http://pytorch.org/docs/0.4.0/torch.html#creation-ops) now also take in ``dtype``, ``device``, ``layout``, and ``requires_grad`` options to specify the desired attributes on the returned ``Tensor``. For example,

```python
>>> device = torch.device("cuda:1")
>>> x = torch.randn(3, 3, dtype=torch.float64, device=device)
tensor([[-0.6344,  0.8562, -1.2758],
        [ 0.8414,  1.7962,  1.0589],
        [-0.1369, -1.0462, -0.4373]], dtype=torch.float64, device='cuda:1')
>>> x.requires_grad  # default is False
False
>>> x = torch.zeros(3, requires_grad=True)
>>> x.requires_grad
True
```

#### [``torch.tensor(data, ...)``](http://pytorch.org/docs/0.4.0/torch.html#torch.tensor)
[``torch.tensor``](http://pytorch.org/docs/0.4.0/torch.html#torch.tensor) is one of the newly added [tensor creation methods](http://pytorch.org/docs/0.4.0/torch.html#creation-ops). It takes in array-like data of all kinds and copies the contained values into a new ``Tensor``. As mentioned earlier, [``torch.tensor``](http://pytorch.org/docs/0.4.0/torch.html#torch.tensor) is the PyTorch equivalent of NumPy's ``numpy.array`` constructor.  Unlike the ``torch.*Tensor`` methods, you can also create zero-dimensional ``Tensor``s (aka scalars) this way (a single python number is treated as a Size in the``torch.*Tensor``  methods). Moreover, if a ``dtype`` argument isn't given, it will infer the suitable ``dtype`` given the data. It is the recommended way to create a tensor from existing data like a Python list. For example,

```python
>>> cuda = torch.device("cuda")
>>> torch.tensor([[1], [2], [3]], dtype=torch.half, device=cuda)
tensor([[ 1],
        [ 2],
        [ 3]], device='cuda:0')
>>> torch.tensor(1)               # scalar
tensor(1)
>>> torch.tensor([1, 2.3]).dtype  # type inferece
torch.float32
>>> torch.tensor([1, 2]).dtype    # type inferece
torch.int64
```

We've also added more tensor creation methods. Some of them have ``torch.*_like`` and/or ``tensor.new_*`` variants.

1. ``torch.*_like`` takes in an input ``Tensor`` instead of a shape. It returns a ``Tensor`` with same attributes as the input ``Tensor`` by default unless otherwise specified:

    ```python
    >>> x = torch.randn(3, dtype=torch.float64)
    >>> torch.zeros_like(x)
    tensor([ 0.,  0.,  0.], dtype=torch.float64)
    >>> torch.zeros_like(x, dtype=torch.int)
    tensor([ 0,  0,  0], dtype=torch.int32)
    ```

2. ``tensor.new_*`` can also create ``Tensor``s with same attributes as ``tensor``, but it always takes in a shape argument:

    ```python
    >>> x = torch.randn(3, dtype=torch.float64)
    >>> x.new_ones(2)
    tensor([ 1.,  1.], dtype=torch.float64)
    >>> x.new_ones(4, dtype=torch.int)
    tensor([ 1,  1,  1,  1], dtype=torch.int32)
    ```

To specify the desired shape, you can either use a tuple (e.g., ``torch.zeros((2, 3))``) or variable arguments (e.g., ``torch.zeros(2, 3)``) in most cases.

| Name                                                       | Returned ``Tensor``                                       | ``torch.*_like`` variant | ``tensor.new_*`` variant |
|:-----------------------------------------------------------|-----------------------------------------------------------|--------------------------|--------------------------|
| [``torch.empty``](http://pytorch.org/docs/0.4.0/torch.html#torch.empty)                                            | unintialized memory                                       | ✔                        | ✔                        |
| [``torch.zeros``](http://pytorch.org/docs/0.4.0/torch.html#torch.zeros)                                            | all zeros                                                 | ✔                        | ✔                        |
| [``torch.ones``](http://pytorch.org/docs/0.4.0/torch.html#torch.ones)                                             | all ones                                                  | ✔                        | ✔                        |
| [``torch.full``](http://pytorch.org/docs/0.4.0/torch.html#torch.full)                                             | filled with a given value                                 | ✔                        | ✔                        |
| [``torch.rand``](http://pytorch.org/docs/0.4.0/torch.html#torch.rand)                                             | i.i.d. continuous ``Uniform[0, 1)``                       | ✔                        |                          |
| [``torch.randn``](http://pytorch.org/docs/0.4.0/torch.html#torch.randn)                                            | i.i.d. ``Normal(0, 1)``                                   | ✔                        |                          |
| [``torch.randint``](http://pytorch.org/docs/0.4.0/torch.html#torch.randint)                                          | i.i.d. discrete Uniform in given range                    | ✔                        |                          |
| [``torch.randperm``](http://pytorch.org/docs/0.4.0/torch.html#torch.randperm)                                         | random permutation of ``{0, 1, ..., n - 1}``              |                          |                          |
| [``torch.tensor``](http://pytorch.org/docs/0.4.0/torch.html#torch.tensor)                                           | copied from existing data (`list`, NumPy `ndarray`, etc.) |                          | ✔                        |
| [``torch.from_numpy``](http://pytorch.org/docs/0.4.0/torch.html#torch.from_numpy)*                                      | from NumPy ``ndarray`` (sharing storage without copying)  |                          |                          |
| [``torch.arange``](http://pytorch.org/docs/0.4.0/torch.html#torch.arange), <br>[``torch.range``](http://pytorch.org/docs/0.4.0/torch.html#torch.range), and <br>[``torch.linspace``](http://pytorch.org/docs/0.4.0/torch.html#torch.linspace)  | uniformly spaced values in a given range                  |                          |                          |
| [``torch.logspace``](http://pytorch.org/docs/0.4.0/torch.html#torch.logspace)                                         | logarithmically spaced values in a given range            |                          |                          |
| [``torch.eye``](http://pytorch.org/docs/0.4.0/torch.html#torch.eye)                                              | identity matrix                                           |                          |                          |

<span class='note'>*: [``torch.from_numpy``](http://pytorch.org/docs/0.4.0/torch.html#torch.from_numpy) only takes in a NumPy ``ndarray`` as its input argument.</span>

<!---
we need some special formatting to make the above table and note look nicer
-->
<style>
  .content table code { font-size: inherit; }
  .content table td { white-space: nowrap; }
  .content .note { font-size: 85%; }
  .content .note code { font-size: 13px; }
</style>

## Writing device-agnostic code

Previous versions of PyTorch made it difficult to write code that was device agnostic (i.e. that could run on both CUDA-enabled and CPU-only machines without modification).

PyTorch 0.4.0 makes this easier in two ways:
* The `device` attribute of a Tensor gives the [``torch.device``](http://pytorch.org/docs/0.4.0/tensor_attributes.html#torch.torch.device) for all Tensors (`get_device` only works for CUDA tensors)
* The `to` method of ``Tensors`` and ``Modules`` can be used to easily move objects to different devices (instead of having to call `cpu()` or `cuda()` based on the context)


We recommend the following pattern:
```python
# at beginning of the script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

...

# then whenever you get a new Tensor or Module
# this won't copy if they are already on the desired device
input = data.to(device)
model = MyModule(...).to(device)
```

## New edge-case constraints on names of submodules, parameters, and buffers in ``nn.Module``

`name` that is an empty string or contains `"."` is no longer permitted in `module.add_module(name, value)`, `module.add_parameter(name, value)` or `module.add_buffer(name, value)` because such names may cause lost data in the `state_dict`. If you are loading a checkpoint for modules containing such names, please update the module definition and patch the `state_dict` before loading it.

## Code Samples (Putting it all together)

To get a flavor of the overall recommended changes in 0.4.0, let's look at a quick example for a common code pattern in both 0.3.1 and 0.4.0:


+ 0.3.1 (old):

    ```python
    model = MyRNN()
    if use_cuda:
        model = model.cuda()

    # train
    total_loss = 0
    for input, target in train_loader:
        input, target = Variable(input), Variable(target)
        hidden = Variable(torch.zeros(*h_shape))  # init hidden
        if use_cuda:
            input, target, hidden = input.cuda(), target.cuda(), hidden.cuda()
        ...  # get loss and optimize
        total_loss += loss.data[0]

    # evaluate
    for input, target in test_loader:
        input = Variable(input, volatile=True)
        if use_cuda:
            ...
        ...
    ```

+ 0.4.0 (new):

    ```python
    # torch.device object used throughout this script
    device = torch.device("cuda" if use_cuda else "cpu")

    model = MyRNN().to(device)

    # train
    total_loss = 0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        hidden = input.new_zeros(*h_shape)  # has the same device & dtype as `input`
        ...  # get loss and optimize
        total_loss += loss.item()           # get Python number from 1-element Tensor

    # evaluate
    with torch.no_grad():                   # operations inside don't track history
        for input, target in test_loader:
            ...
    ```


Thank you for reading! Please refer to our [documentation](http://pytorch.org/docs/0.4.0/index.html) and [release notes](https://github.com/pytorch/pytorch/releases/tag/v0.4.0) for more details.

Happy PyTorch-ing!
