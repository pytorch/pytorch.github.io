---
layout: blog_detail
title: 'PyTorch framework for crypto secure random number generation, torchcsprng, now available'
author: Team PyTorch
---

[torchcsprng](https://github.com/pytorch/csprng) is a PyTorch [C++/CUDA extension](https://pytorch.org/tutorials/advanced/cpp_extension.html) that provides [cryptographically secure pseudorandom number generators](https://en.wikipedia.org/wiki/Cryptographically_secure_pseudorandom_number_generator) for PyTorch.

One of the key components of modern cryptography is the pseudorandom number generator. The use of badly designed or inappropriate random number generators can often leave a good cryptosystem vulnerable to attack. Particular care must be taken to use a random number generator that is designed for cryptographic use, rather than a "general-purpose" random number generator which may be fine for some applications but not ones that are required to be cryptographically secure. Additionally, most pseudorandom number generators scale poorly to massively parallel high-performance computation because of their sequential nature. Others don’t satisfy cryptographically secure properties.

## torchcsprng overview 

Historically, PyTorch had only two pseudorandom number generator implementations: Mersenne Twister for CPU and Nvidia’s cuRAND Philox for CUDA. Despite good performance properties, neither of them are suitable for cryptographic applications. Over the course of the past several months, the PyTorch team developed the torchcsprng extension API. Based on PyTorch dispatch mechanism and operator registration, it allows the users to extend c10::GeneratorImpl and implement their own custom pseudorandom number generator.

torchcsprng generates a random 128-bit key on the CPU using one of its generators and then runs AES128 in [CTR mode](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Counter_(CTR)) either on CPU or GPU using CUDA. This then generates a random 128-bit state and applies a transformation function to map it to target tensor values. This approach is based on [Parallel Random Numbers: As Easy as 1, 2, 3 (John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw, D. E. Shaw Research)](http://www.thesalmons.org/john/random123/papers/random123sc11.pdf). It makes torchcsprng both crypto-secure and parallel on both CPU and CUDA.

<div class="text-center">
  <img src="{{ site.url }}/assets/images/torchcsprng.jpg" width="100%">
</div>

Since torchcsprng is a PyTorch extension, it is available on the platforms where PyTorch is available (support for Windows-CUDA will be available in the coming months). 

## Using torchcsprng

The torchcsprng API is very simple to use and is fully compatible with the PyTorch random infrastructure:

**Step 1: Install via binary distribution**

Anaconda:

`conda install torchcsprng -c pytorch`

pip:

`pip install torchcsprng`

**Step 2: import packages as usual but add csprng**

`import torch`
`import torchcsprng as csprng`

**Step 3: Create a cryptographically secure pseudorandom number generator from /dev/urandom:**

`urandom_gen = csprng.create_random_device_generator('/dev/urandom')`

and simply use it with the existing PyTorch methods:

`torch.randn(10, device='cpu', generator=urandom_gen)`

**Step 4: Test with Cuda**

One of the advantages of torchcsprng generators is that they can be used with both CPU and CUDA tensors:

`torch.randn(10, device='cuda', generator=urandom_gen)`

Another advantage of torchcsprng generators is that they are parallel on CPU unlike the default PyTorch CPU generator.

## Getting Started

The easiest way to get started with torchcsprng is by visiting the [GitHub page](https://github.com/pytorch/csprng) where you can find installation and build instructions, and more how-to examples. 

Cheers,
The PyTorch Team






