---
layout: blog_detail
title: 'The torch.fft module: Accelerated Fast Fourier Transforms with Autograd in PyTorch'
author: Mike Ruberry, Peter Bell, and Joe Spisak 
---

The Fast Fourier Transform (FFT) calculates the Discrete Fourier Transform in O(n log n) time. It is foundational to a wide variety of numerical algorithms and signal processing techniques since it makes working in signals’ “frequency domains” as tractable as working in their spatial or temporal domains.

As part of PyTorch’s goal to support hardware-accelerated deep learning and scientific computing, we have invested in improving our FFT support, and with PyTorch 1.8, we are releasing the ``torch.fft`` module. This module implements the same functions as NumPy’s ``np.fft`` module, but with support for accelerators, like GPUs, and autograd. 

## Getting started

Getting started with the new ``torch.fft`` module is easy whether you are familiar with NumPy’s ``np.fft`` module or not. While complete documentation for each function in the module can be found [here](https://pytorch.org/docs/1.8.0/fft.html), a breakdown of what it offers is:

* ``fft``, which computes a complex FFT over a single dimension, and ``ifft``, its inverse
* the more general ``fftn`` and ``ifftn``, which support multiple dimensions
* The “real” FFT functions, ``rfft``, ``irfft``, ``rfftn``, ``irfftn``,  designed to work with signals that are real-valued in their time domains
* The "Hermitian" FFT functions, ``hfft`` and ``ihfft``, designed to work with signals that are real-valued in their frequency domains
* Helper functions, like ``fftfreq``, ``rfftfreq``, ``fftshift``, ``ifftshift``, that make it easier to manipulate signals

We think these functions provide a straightforward interface for FFT functionality, as vetted by the NumPy community, although we are always interested in feedback and suggestions!

To better illustrate how easy it is to move from NumPy’s ``np.fft`` module to PyTorch’s ``torch.fft`` module, let’s look at a NumPy implementation of a simple low-pass filter that removes high-frequency variance from a 2-dimensional image, a form of noise reduction or blurring:

```python
import numpy as np
import numpy.fft as fft

def lowpass_np(input, limit):
    pass1 = np.abs(fft.rfftfreq(input.shape[-1])) < limit
    pass2 = np.abs(fft.fftfreq(input.shape[-2])) < limit
    kernel = np.outer(pass2, pass1)
    
    fft_input = fft.rfft2(input)
    return fft.irfft2(fft_input * kernel, s=input.shape[-2:])
```

Now let’s see the same filter implemented in PyTorch:

```python
import torch
import torch.fft as fft

def lowpass_torch(input, limit):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) < limit
    pass2 = torch.abs(fft.fftfreq(input.shape[-2])) < limit
    kernel = torch.outer(pass2, pass1)
    
    fft_input = fft.rfft2(input)
    return fft.irfft2(fft_input * kernel, s=input.shape[-2:])
```

Not only do current uses of NumPy’s ``np.fft`` module translate directly to ``torch.fft``, the ``torch.fft`` operations also support tensors on accelerators, like GPUs and autograd. This makes it possible to (among other things) develop new neural network modules using the FFT.


## Performance

The ``torch.fft`` module is not only easy to use — it is also fast! PyTorch natively supports Intel’s MKL-FFT library on Intel CPUs, and NVIDIA’s cuFFT library on CUDA devices, and we have carefully optimized how we use those libraries to maximize performance. While your own results will depend on your CPU and CUDA hardware, computing Fast Fourier Transforms on CUDA devices can be many times faster than computing it on the CPU, especially for larger signals.

In the future, we may add support for additional math libraries to support more hardware. See below for where you can request additional hardware support.

## Updating from older PyTorch versions

Some PyTorch users might know that older versions of PyTorch also offered FFT functionality with the ``torch.fft()`` function. Unfortunately, this function had to be removed because its name conflicted with the new module’s name, and we think the new functionality is the best way to use the Fast Fourier Transform in PyTorch. In particular, ``torch.fft()`` was developed before PyTorch supported complex tensors, while the ``torch.fft`` module was designed to work with them.

PyTorch also has a “Short Time Fourier Transform”, ``torch.stft``, and its inverse ``torch.istft``. These functions are being kept but updated to support complex tensors. 

## Future

As mentioned, PyTorch 1.8 offers the torch.fft module, which makes it easy to use the Fast Fourier Transform (FFT) on accelerators and with support for autograd. We encourage you to try it out!

While this module has been modeled after NumPy’s ``np.fft`` module so far, we are not stopping there. We are eager to hear from you, our community, on what FFT-related functionality you need, and we encourage you to create posts on our forums at [https://discuss.pytorch.org/](https://discuss.pytorch.org/), or [file issues on our Github](https://github.com/pytorch/pytorch/issues/new?assignees=&labels=&template=feature-request.md) with your feedback and requests. Early adopters have already started asking about Discrete Cosine Transforms and support for more hardware platforms, for example, and we are investigating those features now.

We look forward to hearing from you and seeing what the community does with PyTorch’s new FFT functionality!
