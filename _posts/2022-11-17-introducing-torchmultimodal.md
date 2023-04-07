---
layout: blog_detail
title: "Introducing TorchMultimodal - a library for accelerating exploration in Multimodal AI"
author: Kartikay Khandelwal, Ankita De
featured-img: "assets/images/torch-multimodal-feature-image.png"
---

We are announcing TorchMultimodal Beta, a PyTorch domain library for training SoTA multi-task multimodal models at scale. The library provides composable building blocks (modules, transforms, loss functions) to accelerate model development, SoTA model architectures (FLAVA, MDETR, Omnivore) from published research, training and evaluation scripts, as well as notebooks for exploring these models. The library is under active development, and we’d love to hear your feedback! You can find more details on how to get started [here](https://github.com/facebookresearch/multimodal#installation).

## Why TorchMultimodal?

Interest is rising around AI models that understand multiple input types (text, images, videos and audio signals), and optionally use this understanding to generate different forms of outputs (sentences, pictures, videos). Recent work from FAIR such as [FLAVA](https://arxiv.org/abs/2112.04482), [Omnivore](https://arxiv.org/pdf/2201.08377.pdf) and [data2vec](https://arxiv.org/abs/2202.03555) have shown that [multimodal models for understanding](https://ai.facebook.com/blog/advances-in-multimodal-understanding-research-at-meta-ai/) are competitive with unimodal counterparts, and in some cases are establishing the new state-of-the art. Generative models such as [Make-a-video](https://ai.facebook.com/blog/generative-ai-text-to-video/) and [Make-a-scene](https://ai.facebook.com/blog/greater-creative-control-for-ai-image-generation/) are redefining what modern AI systems can do.

As interest in multimodal AI has grown, researchers are looking for tools and libraries to quickly experiment with ideas, and build on top of the latest research in the field. While the PyTorch ecosystem has a rich repository of libraries and frameworks, it’s not always obvious how components from these interoperate with each other, or how they can be stitched together to build SoTA multimodal models.

TorchMultimodal solves this problem by providing:

- **Composable and easy-to-use building blocks** which researchers can use to accelerate model development and experimentation in their own workflows. These are designed to be modular, and can be easily extended to handle new modalities.

- **End-to-end examples for training and evaluating the latest models from research.** These should serve as starting points for ongoing/future research, as well as examples for using advanced features such as integrating with FSDP and activation checkpointing for scaling up model and batch sizes.

## Introducing TorchMultimodal

TorchMultimodal is a PyTorch domain library for training multi-task multimodal models at scale. In the repository, we provide:

- **[Building Blocks](https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal)**. A collection of modular and composable building blocks like models, fusion layers, loss functions, datasets and utilities. Some examples include:

  - [Contrastive Loss with Temperature](https://github.com/facebookresearch/multimodal/blob/4d2236877467ff8f56aa1935dd92d7782751b135/torchmultimodal/modules/losses/contrastive_loss_with_temperature.py#L145). Commonly used function for training models like CLIP and FLAVA. We also include variants such as [ImageTextContrastiveLoss](https://github.com/facebookresearch/multimodal/blob/4d2236877467ff8f56aa1935dd92d7782751b135/torchmultimodal/modules/losses/albef.py#L14) used in models like ALBEF.

  - [Codebook layers](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/layers/codebook.py#L31) which compresses high dimensional data by nearest neighbor lookup in an embedding space and is a vital component of VQVAEs (provided as a [model](https://github.com/facebookresearch/multimodal/blob/4d2236877467ff8f56aa1935dd92d7782751b135/torchmultimodal/models/vqvae.py#L26) in the repository).

  - [Shifted-window Attention](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/encoders/swin_transformer_3d_encoder.py#L76) window based multi-head self attention which is a vital component of encoders like Swin 3D Transformers.

  - [Components for CLIP.](https://github.com/facebookresearch/multimodal/tree/4d2236877467ff8f56aa1935dd92d7782751b135/torchmultimodal/models/clip) A popular model published by OpenAI which has proven to be extremely effective at learning text and image representations.

  - [Multimodal GPT.](https://github.com/facebookresearch/multimodal/blob/4d2236877467ff8f56aa1935dd92d7782751b135/torchmultimodal/models/gpt.py) An abstraction that extends OpenAI’s [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) architecture for multimodal generation when combined with the [generation utility](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/utils/generate.py#L33).

  - [MultiHeadAttention](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/layers/attention.py#L134). A critical component for attention-based models with support for fast auto-regressive decoding.

- **[Examples](https://github.com/facebookresearch/multimodal/tree/main/examples)**. A collection of examples that show how to combine these building blocks with components and common infrastructure (Lightning, TorchMetrics) from across the PyTorch Ecosystem to replicate state-of-the-art models published in literature. We currently provide five examples, which include.

  - [FLAVA](https://github.com/facebookresearch/multimodal/tree/main/examples/flava) \[[paper](https://arxiv.org/abs/2112.04482)\]. Official code for the paper accepted at CVPR, including a tutorial on finetuning FLAVA.

  - [MDETR](https://github.com/facebookresearch/multimodal/tree/main/examples/mdetr) \[[paper](https://arxiv.org/abs/2104.12763)\]. Collaboration with authors from NYU to provide an example which alleviates interoperability pain points in the PyTorch ecosystem, including a [notebook](https://github.com/facebookresearch/multimodal/blob/main/examples/mdetr/MDETRTutorial.ipynb) on using MDETR for phrase grounding and visual question answering.

  - [Omnivore](https://github.com/facebookresearch/multimodal/tree/main/examples/omnivore) \[[paper](https://arxiv.org/abs/2204.08058)\]. First example in TorchMultimodal of a model which deals with Video and 3D data, including a [notebook](https://github.com/facebookresearch/multimodal/blob/main/examples/omnivore/omnivore_inference_demo.ipynb) for exploring the model.

  - [MUGEN](https://github.com/facebookresearch/multimodal/tree/main/examples/mugen) \[[paper](https://arxiv.org/abs/2204.08058)\]. Foundational work for auto-regressive [generation](https://colab.research.google.com/drive/1C3ZbH_l19g_KqW3CPeX2-8Q2sOUCpmZo?usp=sharing) and [retrieval](https://colab.research.google.com/drive/1gZfz1jsy79CNCK9t2_r43yt3z7v-w4HS?usp=sharing), including demos for text-video generation and retrieval with a large-scale synthetic dataset enriched from OpenAI [coinrun](https://github.com/openai/coinrun).

  - [ALBEF](https://github.com/facebookresearch/multimodal/tree/main/examples/albef) \[[paper](https://arxiv.org/abs/2107.07651)\] Code for the model, including a [notebook](https://github.com/facebookresearch/multimodal/blob/main/examples/albef/vqa_with_albef.ipynb) for using this model for Visual Question Answering.

The following code snippet showcases an example usage of several TorchMultimodal components related to CLIP:

```python

# instantiate clip transform
clip_transform = CLIPTransform()

# pass the transform to your dataset. Here we use coco captions
dataset = CocoCaptions(root= ..., annFile=..., transforms=clip_transform)
dataloader = DataLoader(dataset, batch_size=16)

# instantiate model. Here we use clip with vit-L as the image encoder
model= clip_vit_l14()

# define loss and other things needed for training
clip_loss = ContrastiveLossWithTemperature()
optim = torch.optim.AdamW(model.parameters(), lr = 1e-5)
epochs = 1

# write your train loop
for _ in range(epochs):
	for batch_idx, batch in enumerate(dataloader):
		image, text = batch
		image_embeddings, text_embeddings = model(image, text)
		loss = contrastive_loss_with_temperature(image_embeddings, text_embeddings)
		loss.backward()
		optimizer.step()
```

Apart from the code, we are also **releasing a tutorial for fine-tuning multimodal foundation models, and a blog post (with code pointers) on how to scale up such models using techniques from PyTorch Distributed (FSDP and activation checkpointing)**. We hope such examples and tutorials will serve to demystify a number of advanced features available in the PyTorch ecosystem.

## What’s Next?

While this is an exciting launch, there’s a lot more to come. The library is under development and we are working on adding some of the exciting developments in the space of diffusion models, and examples to showcase common trends from research. As you explore and use the library, we’d love to hear any feedback you might have! You can find more details on how to get started [here](https://github.com/facebookresearch/multimodal#installation).

## Team

The primary contributors and developers of TorchMultimodal include Ankita De, Evan Smothers, Kartikay Khandelwal, Lan Gong, Laurence Rouesnel, Nahiyan Malik, Rafi Ayub and Yosua Michael Maranatha.
