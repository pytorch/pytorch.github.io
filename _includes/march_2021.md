
# Issue \#1

Welcome to the first issue of the PyTorch Contributors newsletter! Keeping track of everything that’s happening in the PyTorch developer world is a big task; here you will find curated news including RFCs, feature roadmaps, notable PRs, editorials from developers, and more. If you have questions or suggestions for the newsletter, we'd love to [hear from you](https://forms.gle/2KApHZa3oDHuAQ288)

## PyTorch 1.8.0

PyTorch 1.8 was released on March 4th with support for functional transformations using `torch.fx`, stabilized frontend APIs for scientific computing (`torch.fft`, `torch.linalg`, Autograd for complex tensors) and significant improvements to distributed training. Read the full [Release Notes](https://github.com/pytorch/pytorch/releases/tag/v1.8.0){:target="_blank"}.

## PyTorch Ecosystem Day

On April 21, we’re hosting a virtual event for our ecosystem and industry communities to showcase their work and discover new opportunities to collaborate. The day will be filled with discussion on new developments, trends, challenges and best practices through posters, breakout sessions and networking. 

## [The PyTorch open source process](http://blog.ezyang.com/2021/01/pytorch-open-source-process/){:target="_blank"}

[@ezyang](https://github.com/ezyang){:target="_blank"} describes the challenges of maintaining a PyTorch-scale project, and the current open source processes (triaging and CI oncalls, RFC discussions) to help PyTorch operate effectively.

## Developers forum

We launched https://dev-discuss.pytorch.org/ a low-traffic high-signal forum for long-form discussions about PyTorch internals.

## [RFC] [Dataloader v2](https://github.com/pytorch/pytorch/issues/49440)

[@VitalyFedyunin](https://github.com/VitalyFedyunin) proposes redesigning the DataLoader to support lazy loading, sharding, pipelining data operations (including async) and shuffling & sampling in a more modular way. Join the discussion [here](https://github.com/pytorch/pytorch/issues/49440).

## [RFC] [Improving TorchScript Usability](https://dev-discuss.pytorch.org/t/torchscript-usability/55)

In a series of 3 blog posts ([1](https://lernapparat.de/scripttorch/), [2](https://lernapparat.de/jit-python-graphops/), [3](https://lernapparat.de/jit-fallback/)) [@t-vi](https://github.com/t-vi) explores ideas to improve the user and developer experience of TorchScript.

## [RFC] [CSR and DM storage formats for sparse tensors](https://github.com/pytorch/rfcs/pull/13)

[@pearu](https://github.com/pearu) proposes an [RFC](https://github.com/pytorch/rfcs/pull/13) to make linear algebra operations more performant by

- implementing the CSR storage format, where a 2D array is defined by shape and 1D tensors for compressed row indices, column indices, and values (PyTorch 1D tensor)
- introducing the Dimension Mapping storage format that generalizes a 2D CSR to multidimensional arrays using a bijective mapping between the storage and wrapper elements.

## [RFC] [Forward Mode AD](https://github.com/pytorch/rfcs/pull/11)

[@albanD](https://github.com/albanD) proposes an [RFC](https://github.com/pytorch/rfcs/pull/11) to implement forward mode autodiff using Tensor-based [dual numbers](https://blog.demofox.org/2014/12/30/dual-numbers-automatic-differentiation/), where the real part represents the tensor and the *dual* part stores the forward gradient of the tensor. The core of the feature has landed [(PR)](https://github.com/pytorch/pytorch/pull/49734), with more formulas in WIP. Complete forward mode AD is expected to land by July 2021.
