---
layout: blog_detail
title: "A Better Transformer for Fast Transformer Encoder Inference"
author: Michael Gschwind, Eric Han, Scott Wolchok, Rui Zhu
featured-img: "/assets/images/METAPT-002-BarGraph-02-static.png"
---

**tl;dr** Transformers achieve state-of-the-art performance for NLP, and are becoming popular for a myriad of other tasks. They are computationally expensive which has been a blocker to their widespread productionisation. Launching with PyTorch 1.12, Better Transformer implements a backwards-compatible fast path of `torch.nn.TransformerEncoder` for Transformer Inference and does not require model authors to modify their models. Better Transformer provides 23-44% speedup for unpadded sequences and up to 2x+ speedup for padded sequences.  To use BetterTransformer, install PyTorch 1.12 and start using `torch.nn.TransformerEncoder` today.

<p align="center">
  <img src="/assets/images/2022-6-28-a-better-transformer-for-fast-transformer-encoder-inference-1.png" width="30%">
</p>

<p align="center">
The Encoder Structure of the Transformer Architecture<br>
Taken from "<a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a>".
</p>

## Performance Improvements

BetterTransformer launches with accelerated native implementations of MultiHeadAttention and TransformerEncoderLayer for CPUs and GPUs. These fast paths are integrated in the standard PyTorch Transformer APIs, and will accelerate [TransformerEncoder](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py#L165), [TransformerEncoderLayer](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py#L300) and [MultiHeadAttention](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py#L882) nn.modules. These new modules implement two types of optimizations: (1) fused kernels combine multiple individual operators normally used to implement Transformers to provide a more efficient implementation, and (2) take advantage of sparsity in the inputs to avoid performing unnecessary operations on padding tokens.  Padding tokens frequently account for a large fraction of input batches in many Transformer models used for Natural Language Processing. 

## Backwards compatibility

Advantageously, **no model changes are necessary to benefit from the performance boost offered by BetterTransformer.**  To benefit from fast path execution, inputs and operating conditions must satisfy some access conditions (see below). While the internal implementation of Transformer APIs has changed, PyTorch 1.12 maintains strict compatibility with Transformer modules shipped in previous versions, enabling PyTorch users to read and process models created and trained with previous PyTorch releases while benefiting from the Better Transformer improvements. 

In addition to enabling the PyTorch nn.Modules, Better Transformer provides improvements for the PyTorch libraries. Performance benefits will become available through two different enablement paths:

1. **Transparent acceleration:** Users of PyTorch nn.Modules such as MultiHeadAttention as well as higher-level Transformer components will benefit from the improved performance of the new nn.Modules transparently. (An example of this is the [ViT visual transformer](https://arxiv.org/abs/2010.11929) implementation used in the torchvision library.) 
2. **Torchtext library acceleration:** As part of this project, we have optimized torchtext to build on the PyTorch core API to benefit from BetterTransformer enhancements while maintaining strict and transparent compatibility with previous library versions.

Using PyTorch Transformers in Torchtext also ensures that torchtext will automatically benefit from all future enhancements to the PyTorch Transformer implementation. The updated torchtext library supports all torchtext models with provide full compatibility and transparent performance enhancements with models created and also use weights trained with previous versions of torchtext.

## Taking advantage of the Fastpath

As mentioned previously, Better Transformer is a fastpath for the PyTorch Transformer API. The fastpath is a native, specialized implementation of key Transformer functions for CPU and GPU that applies to the most commonly used subset of execution patterns for Transformers. If the criteria are not met, control flows to a Pytorch implementation which is fully compatible with the Pytorch Transformer API. At this point, fastpath execution applies to encoder models used during inference, with plans to expand the fastpath execution to models based on transformer decoder to support popular seq2seq and decoder-only (e.g., GPT) model architectures, and to training.

As previously mentioned, fastpath execution is subject to some criteria, most importantly executing the model in inference mode and operating on input tensors that do not collect gradient tape information (e.g., running with torch.no_grad). The [full list of conditions for fastpath execution](https://github.com/pytorch/pytorch/blob/29189d2ba8e583b2355cd0e9517a1ee742ba12cf/torch/nn/modules/activation.py#L1060) may be found in the nn.Module.MultiHeadAttention which implements access control to the fast path and can provide diagnostic output. 

Execution may occur either on CPU or GPU, with the performance improvements being more substantial for GPU execution due to the higher performance headroom provided by accelerators. To benefit from fastpath execution, models must be composed of TransformerEncoders, TransformerEncoderLayer and MHA. Libraries and models that implement their own TransformerEncoders, TransformerEncoder layers or MHA will benefit from BetterTransformer improvements to the extent these build on PyTorch Transformer API modules.  

The following graphs show the performance achieved for the BERT-base model with small and large-scale inputs:

<p align="center">
  <img src="/assets/images/2022-6-28-a-better-transformer-for-fast-transformer-encoder-inference-2.png" width="80%">
</p>

<p align="center">
  <img src="/assets/images/2022-6-28-a-better-transformer-for-fast-transformer-encoder-inference-3.png" width="80%">
</p>

<p align="center">
<b>Figure: PyTorch 1.12 Improvements with Better Transformer fastpath execution</b>
</p>


BetterTransformer includes two types of optimization: (1) fused kernels implementing multiple operations more efficiently in a single kernel, and (2) exploiting sparsity by avoiding unnecessary processing on padding tokens. Enhanced performance for small input sizes benefits primarily from the fused kernel implementations, and shows a constant performance improvement regardless of padding amount. While large inputs still benefit from fused kernels, the computation heavy processing limits the benefits that may be obtained by the fused kernels as baseline performance is already closer to the theoretical peak. However, as we increase the amount of padding, performance increases dramatically as increasingly large amounts of computation can be avoided by exploiting the sparsity introduced by padding in NLP workloads.


As part of our ongoing work on PyTorch BetterTransformer, we are working on extending BetterTransformer improvements to Transformer Decoders.  In addition, we are partnering to enable additional libraries such as FairSeq, MetaSeq and HuggingFace to benefit all Transformer-based PyTorch models.  We’ll provide future updates on the progress of Better Transformer accelerations for the larger PyTorch ecosystem as part of this blog series.  



Acknowledgements: The authors would like to thank Lin Qiao, Ajit Mathews, Andrew Tulloch, Dmytro Dzhulgakov, Natalia Gimelshein, Christian Puhrsch, Emad El-Haraty, Mark Saroufim, Adnan Aziz, Geeta Chauhan, and Hamid Shojanazeri for their support, contributions and many helpful suggestions throughout the course of this project, and in the preparation of this blog.