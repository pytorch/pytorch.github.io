---
layout: blog_detail
title: "A BetterTransformer for Fast Transformer Inference"
author: Michael Gschwind, Eric Han, Scott Wolchok, Rui Zhu, Christian Puhrsch
featured-img: "/assets/images/2022-7-12-a-better-transformer-for-fast-transformer-encoder-inference-3.png"
---

**tl;dr** Transformers achieve state-of-the-art performance for NLP, and are becoming popular for a myriad of other tasks. They are computationally expensive which has been a blocker to their widespread productionisation. Launching with PyTorch 1.12, BetterTransformer implements a backwards-compatible fast path of `torch.nn.TransformerEncoder` for Transformer Encoder Inference and does not require model authors to modify their models. BetterTransformer improvements can exceed 2x in speedup and throughput for many common execution scenarios. To use BetterTransformer, [install](https://pytorch.org/get-started/locally/) PyTorch 1.12 and start using high-quality, high-performance Transformer models with the PyTorch API today.

<p align="center">
  <img src="/assets/images/2022-7-12-a-better-transformer-for-fast-transformer-encoder-inference-1.png" width="100%">
</p>

<p align="center">
Diagram of the Transformer Encoder Architecture (from "<a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a>"). During Inference, the entire module will execute as a single PyTorch-native function.
</p>

In this blog post, we share the following topics — Performance Improvements, Backwards compatibility, and Taking advantage of the FastPath. Learn more about these topics below.  

## Performance Improvements

BetterTransformer launches with accelerated native implementations of MultiHeadAttention and TransformerEncoderLayer for CPUs and GPUs. These fast paths are integrated in the standard PyTorch Transformer APIs, and will accelerate [TransformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html), [TransformerEncoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) and [MultiHeadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) nn.modules. These new modules implement two types of optimizations: (1) fused kernels combine multiple individual operators normally used to implement Transformers to provide a more efficient implementation, and (2) take advantage of sparsity in the inputs to avoid performing unnecessary operations on padding tokens. Padding tokens frequently account for a large fraction of input batches in many Transformer models used for Natural Language Processing. 

## Backwards compatibility

Advantageously, **no model changes are necessary to benefit from the performance boost offered by BetterTransformer.** To benefit from fast path execution, inputs and operating conditions must satisfy some access conditions (see below). While the internal implementation of Transformer APIs has changed, PyTorch 1.12 maintains strict compatibility with Transformer modules shipped in previous versions, enabling PyTorch users to use models created and trained with previous PyTorch releases while benefiting from BetterTransformer improvements.

In addition to enabling the PyTorch nn.Modules, BetterTransformer provides improvements for PyTorch libraries. Performance benefits will become available through two different enablement paths:

1. <u><strong>Transparent acceleration:</strong></u> Current users of PyTorch nn.Modules such as [MultiHeadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) as well as higher-level Transformer components will benefit from the improved performance of the new nn.Modules automatically. An example of this is the [visual transformer (ViT)](https://arxiv.org/abs/2010.11929) implementation used in the torchvision library ([code link](https://github.com/pytorch/vision/blob/87cde716b7f108f3db7b86047596ebfad1b88380/torchvision/models/vision_transformer.py#L103)).

2. <u><strong>Torchtext library acceleration:</strong></u> As part of this project, we have optimized Torchtext to build on the PyTorch core API to benefit from BetterTransformer enhancements while maintaining strict and transparent compatibility with previous library versions and models trained with previous Torchtext versions. Using PyTorch Transformers in Torchtext also ensures that Torchtext will benefit from expected future enhancements to the PyTorch Transformer implementation.

## Taking advantage of the Fastpath

BetterTransformer is a fastpath for the PyTorch Transformer API. The fastpath is a native, specialized implementation of key Transformer functions for CPU and GPU that applies to common Transformer use cases. 

To take advantage of input sparsity (i.e. padding) in accelerating your model (see Figure 2), set the keyword argument `enable_nested_tensor=True` when instantiating a [TransformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html) and pass in the `src_key_padding_mask` argument (which denotes padding tokens) during inference. This requires the padding mask to be contiguous, which is the typical case.

Currently, the BetterTransformer speedup only applies to transformer encoder models used in inference. To benefit from fastpath execution, models must be composed of any of the following components: [TransformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html), [TransformerEncoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) or [MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) (MHA). Fastpath execution is also subject to some criteria. Most importantly, the model must be executed in inference mode and operate on input tensors that do not collect gradient tape information (e.g., running with torch.no_grad). The full list of conditions can be found at these links for [nn.MultiHeadAttention](https://github.com/pytorch/pytorch/blob/29189d2ba8e583b2355cd0e9517a1ee742ba12cf/torch/nn/modules/activation.py#L1060) and [nn.TransformerEncoder](https://github.com/pytorch/pytorch/blob/29189d2ba8e583b2355cd0e9517a1ee742ba12cf/torch/nn/modules/transformer.py#L206), respectively. If the criteria are not met, control flows to the legacy PyTorch 1.11 Transformer implementation which has the same API, but lacks the fastpath performance boost. 

Other transformer models (such as decoder models) which use the PyTorch MultiheadAttention module will benefit from the BetterTransformer fastpath. Planned future work is to expand the end-to-end BetterTransformer fastpath to models based on [TransformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html) to support popular seq2seq and decoder-only (e.g., [OPT](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/)) model architectures, and to training.

## Speedups

The following graphs show the performance achieved for the [BERT](https://arxiv.org/abs/1810.04805)-base model with small and large-scale inputs:

<p align="center">
  <img src="/assets/images/2022-7-12-a-better-transformer-for-fast-transformer-encoder-inference-2.png" width="80%">
</p>

<p align="center">
<b>Figure 1: PyTorch 1.12 Improvements with BetterTransformer fastpath execution</b>
</p>

<p align="center">
  <img src="/assets/images/2022-7-12-a-better-transformer-for-fast-transformer-encoder-inference-3.png" width="80%">
</p>

<p align="center">
<b>Figure 2: PyTorch 1.12 Improvements with BetterTransformer fastpath execution<br>
with sparsity optimization enabled by enable_nested_tensor=True</b>
</p>


BetterTransformer includes two types of optimization: (1) fused kernels implementing multiple operations more efficiently in a single kernel, and (2) exploiting sparsity by avoiding unnecessary processing on padding tokens. Enhanced performance for small input sizes benefits primarily from the fused kernel implementations, and shows a constant performance improvement regardless of padding amount. While large inputs still benefit from fused kernels, the computation heavy processing limits the benefits that may be obtained by the fused kernels as baseline performance is already closer to the theoretical peak. However, as we increase the amount of padding, performance increases dramatically as increasingly large amounts of computation can be avoided by exploiting the sparsity introduced by padding in NLP workloads.

## Future Work

As part of our ongoing work on PyTorch BetterTransformer, we are working on extending BetterTransformer improvements to Transformer Decoders. We aim to expand beyond inference to training as well.

We are partnering to enable BetterTransformer on additional libraries such as FairSeq, MetaSeq, and HuggingFace to benefit all Transformer-based PyTorch models. We’ll provide future updates on the progress of BetterTransformer accelerations for the larger PyTorch ecosystem as part of this blog series.

Acknowledgements: The authors would like to thank Lin Qiao, Ajit Mathews, Andrew Tulloch, Dmytro Dzhulgakov, Natalia Gimelshein, Emad El-Haraty, Mark Saroufim, Adnan Aziz, Geeta Chauhan, and Hamid Shojanazeri for their support, contributions and many helpful suggestions throughout the course of this project, and in the preparation of this blog.