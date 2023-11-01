---
layout: blog_detail
title: "Compiling NumPy code into C++ or CUDA via torch.compile"
author: Evgeni Burovski, Ralf Gommers and Mario Lezcano
---

Quansight engineers have implemented support for tracing through NumPy code via 
`torch.compile` in PyTorch 2.1. This feature leverages PyTorch’s compiler to 
generate efficient fused vectorized code without having to modify your original 
NumPy code. Even more, it also allows for executing NumPy code on CUDA 
just by running it through `torch.compile` under `torch.device("cuda")`!

In this post, we go over how to use this feature and give a few tips and tricks 
to make the most out of it.


## Compiling NumPy code into Parallel C++

We take as our running example one step in a K-Means algorithm. 
This piece of code is borrowed from this [NumPy book](https://realpython.com/numpy-array-programming/#clustering-algorithms)


```
import numpy as np

def kmeans(X, means):
    return np.argmin(np.linalg.norm(X - means[:, None], axis=2), axis=0)
```


We create a synthetic dataset with 20M random 2-D points. We can see that, 
given that the means are chosen appropriately, the function returns the correct 
cluster for all of them


```
npts = 10_000_000
X = np.repeat([[5, 5], [10, 10]], [npts, npts], axis=0)
X = X + np.random.randn(*X.shape)  # 2 distinct "blobs"
means = np.array([[5, 5], [10, 10]])
np_pred = kmeans(X, means)
```


Benchmarking this function gives us a baseline of **1.26s** on an AMD 3970X CPU.

Compiling this function is now as easy as wrapping it with `torch.compile` and 
executing it with the example inputs


```
import torch

compiled_fn = torch.compile(kmeans)
compiled_pred = compiled_fn(X, means)
assert np.allclose(np_pred, compiled_pred)
```


The compiled function yields a 9x speed-up when running it on 1 core. Even 
better, as opposed to NumPy, our generated code does take advantage of all the 
cores in a processor. As such, when we run it on 32 cores, we get a **57x 
speed-up**. Note that PyTorch always uses all the available cores unless 
explicitly restricted, so this is the default behavior you get when using 
`torch.compile`.

We may inspect the generated C++ code by running the script with the 
environment variable `TORCH_LOGS=output_code`. When doing so, we can see that 
`torch.compile` was able to compile the broadcasting and the two reductions 
into just one for-loop, and parallelize it using OpenMP


```
extern "C" void kernel(const double* in_ptr0, const long* in_ptr1, long* out_ptr0) {
    #pragma omp parallel num_threads(32)
    #pragma omp for
    for(long i0=0L; i0<20000000L; i0+=1L) {
        auto tmp0 = in_ptr0[2L*i0];
        auto tmp1 = in_ptr1[0L];
        auto tmp5 = in_ptr0[1L + (2L*i0)];
        auto tmp6 = in_ptr1[1L];
        // Rest of the kernel omitted for brevity
```



## Compiling NumPy code into CUDA

Compiling our code so that it runs on CUDA is as simple as setting the 
default device to be CUDA


```
with torch.device("cuda"):
    cuda_pred = compiled_fn(X, means)
assert np.allclose(np_pred, cuda_pred)
```


By inspecting the generated code via `TORCH_LOGS=output_code`, we see that, 
rather than generating CUDA code directly, `torch.compile` generates rather 
readable [triton](https://triton-lang.org/main/index.html) code


```
def triton_(in_ptr0, in_ptr1, out_ptr0, XBLOCK : tl.constexpr):
    xnumel = 20000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    // Rest of the kernel omitted for brevity
```


Running this small snippet on an RTX 2060 gives an **8x speed-up** over the 
original NumPy code. This is something, but it is not particularly impressive, 
given the speed-ups we have seen on CPU. Let’s have a look into how to squeeze 
the most out of our GPU via a couple minor changes.

`float64` vs `float32`. Many GPUs, in particular consumer-grade ones, are 
rather sluggish when running operations on `float64`. For this reason, changing 
the data generation to `float32`, the original NumPy code just gets a bit 
faster, about a 9%, but our CUDA code gets <strong>40% faster</strong>, yielding a <strong>11x 
speed-up</strong> over the plain NumPy code.

`torch.compile`, by default, respects the NumPy semantics, and as such, it uses 
`np.float64` as its default dtype for all its creation ops. As discussed, this 
can hinder performance, so it is possible to change this default by setting


```
from torch._dynamo import config
config.numpy_default_float = "float32"
```


**CPU &lt;> CUDA copies**. An 11x speed-up is good, but it is not even close to 
the CPU numbers. This is caused by a small transformation that `torch.compile 
`does behind the scenes. The code above takes NumPy arrays and returns NumPy 
arrays. All of these arrays are on CPU, but the computations are performed on 
the GPU. This means that every time the function is called, `torch.compile` has 
to copy all these arrays from CPU to the GPU, and then copy the result back to 
CPU to preserve the original semantics. There is no native solution to this 
issue in NumPy, as NumPy does not have the notion of a `device`. That being 
said, we can work around it by creating a wrapper to this function so that it 
accepts PyTorch tensors and returns PyTorch tensors.


```
@torch.compile
def tensor_fn(X, means):
    X, means = X.numpy(), means.numpy()
    ret = kmeans(X, means)
    return torch.from_numpy(ret)

def cuda_fn(X, means):
    with torch.device("cuda"):
        return tensor_fn(X, means)
```


This function now takes tensors in CUDA memory and returns tensors in CUDA 
memory, but the function itself is written in NumPy! `torch.compile` uses the 
`numpy()` and the `from_numpy()` calls as hints, and optimizes them away, and 
internally it simply works with PyTorch tensors without moving the memory at 
all. When we keep the tensors in CUDA and perform the computations in 
`float32`, we see a **200x speed-up** over the initial NumPy implementation on 
`float32` arrays.

**Mixing NumPy and PyTorch**. In this example, we had to write a small adaptor 
to convert tensors to ndarrays and then back to tensors. In programs that mix 
PyTorch and NumPy converting a tensor into an ndarray is often implemented as 
`x.detach().cpu().numpy()`, or simply `x.numpy(force=True)`. Since when running 
under `torch.compile` we can run NumPy code in CUDA, we can implement this 
conversion pattern as call to `x.numpy()`, as we did above. Doing so and 
running the resulting code under `device("cuda")` will generate efficient CUDA 
code from original NumPy calls without copying the data from CUDA to CPU at 
all. Note that the resulting code does not run without `torch.compile`. For it 
to run in eager mode one would need to rollback to `x.numpy(force=True)`.


## Further Speed-up tricks

**General advice**. The CUDA code we have shown is already quite efficient, but 
it is true that the running example is rather short. When dealing with larger 
programs, we may need to tweak parts of it to make it more efficient. A good 
place to start is the multiple [tutorials and FAQs for torch.compile](https://pytorch.org/docs/main/torch.compiler.html#read-more). 
This showcases a number of ways to inspect the tracing process, and how to 
identify problematic code that may cause slowdowns.

**Advice when compiling NumPy code**. NumPy, even if rather similar to PyTorch, 
is often used very differently. It is rather common to perform computations in 
NumPy and then do an if/else depending on values within the array, or perform 
operations in-place, perhaps via boolean masks. These constructions, while 
supported by `torch.compile`, hamper its performance. Changes like writing the 
code in a branchless way to avoid graph breaks, or avoiding in-place ops can go 
a long way.

To write fast NumPy code, it is best to avoid loops, but sometimes they are 
unavoidable. When tracing through a loop, `torch.compile` will try to fully 
unroll it. This is sometimes desirable, but sometimes it may not even be 
possible, like when we have a dynamic stopping condition, like in a while loop. 
In these cases, it may be best to just compile the body of the loop, perhaps a 
few iterations at a time (loop unrolling).

**Debugging NumPy code**. Debugging is rather tricky when a compiler is 
involved. To figure out whether an error you are hitting is a `torch.compile 
`error, or an error from the program, you can execute your NumPy program without 
`torch.compile` by replacing the NumPy import by `import torch._numpy as np`. 
This is should just be used for **debugging purposes** and is in no way a 
replacement for the PyTorch API, as it is **much slower** and, as a private API, 
**may change without notice**. See also [this FAQ](https://pytorch.org/docs/stable/torch.compiler_faq.html#does-numpy-work-with-torch-compile) for other tricks.


## Differences between NumPy and `torch.compile` NumPy

**NumPy scalars**. NumPy returns NumPy scalars in almost any case where PyTorch 
would return a 0-D tensor (e.g. from `np.sum`). Under `torch.compile`, NumPy 
scalars are treated as 0-D arrays. This is just fine in most cases. The only 
case when their behavior diverges is when NumPy scalars are implicitly used as 
Python scalars. For example,


```
>>> np.asarray(2) * [1, 2, 3]  # 0-D array is an array-like
array([2, 4, 6])
>>> u = np.int32(2)
>>> u * [1, 2, 3]              # scalar decays into a Python int
[1, 2, 3, 1, 2, 3]
>>> torch.compile(lambda: u * [1, 2, 3])()
array([2, 4, 6])               # acts as a 0-D array, not as a scalar ?!?!
```


If we compile the first two lines, we see that `torch.compile` treats `u` as a 
0-D array. To recover the eager semantics, we just need to make the casting 
explicit


```
>>> torch.compile(lambda: int(u) * [1, 2, 3])()
[1, 2, 3, 1, 2, 3]
```


**Type promotion and versioning**. NumPy’s type promotion rules may be, at 
times, a bit surprising


```
>>> np.zeros(1, dtype=np.int8) + 127
array([127], dtype=int8)
>>> np.zeros(1, dtype=np.int8) + 128
array([128], dtype=int16)
```


NumPy 2.0 is changing these rules to follow others that are closer to those 
PyTorch. The relevant technical document is [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html). 
`torch.compile` went ahead and implemented NEP 50 rather than the about-to-be-deprecated rules.

In general, NumPy within torch.compile follows NumPy 2.0 pre-release.


## Beyond NumPy: SciPy and scikit-learn

In parallel to this effort of making `torch.compile` understand NumPy code, 
other Quansight engineers have designed and proposed a way to support PyTorch 
tensors within scikit-learn and SciPy. This was received enthusiastically by 
other maintainers from these libraries, as it was shown that using PyTorch as a 
backend would often yield considerable speed-ups. Both projects have now merged 
initial support for PyTorch tensors across a number of APIs and submodules.

This sets the stepping stone to move towards a future where PyTorch tensors can 
be used within other libraries in the Python data ecosystem. Even more, this 
will enable running these other libraries on GPUs and even compiling code 
mixing these libraries and PyTorch, similar to what we have been discussed in 
this post.

If you want to learn more about this effort, how to use it, or how to help 
moving it forward, see [this other blogpost](https://labs.quansight.org/blog/array-api-support-scikit-learn).


## Conclusion

PyTorch has committed since its inception to be a framework compatible with the 
rest of the Python ecosystem. Enabling compiling NumPy programs, and 
establishing the tools necessary to do the same for other prominent libraries 
are two more steps in this direction. Quansight and Meta continue working hand 
on hand, improving the compatibility between PyTorch and the rest of the 
ecosystem.

From Quansight, we would like to thank Mengwei, Voz, and Ed for their 
invaluable help in integrating our work with `torch.compile`. We would also 
like to thank Meta for funding this project as well as previous work on 
improving NumPy compatibility within PyTorch, and the project that led to 
supporting PyTorch within scikit-learn and SciPy. These are giant leaps towards 
consolidating PyTorch as the framework of choice within the open source Python 
data ecosystem.