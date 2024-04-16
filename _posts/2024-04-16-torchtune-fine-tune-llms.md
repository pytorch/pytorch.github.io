---
layout: blog_detail
title: "torchtune: Easily fine-tune LLMs using PyTorch"
---

We’re pleased to announce the alpha release of torchtune, a PyTorch-native library for easily fine-tuning large language models. 

Staying true to PyTorch's design principles, torchtune provides composable and modular building blocks along with easy-to-extend training recipes to fine-tune popular LLMs on a variety of consumer-grade and professional GPUs.

torchtune supports the full fine-tuning workflow from start to finish, including

* Downloading and preparing datasets and model checkpoints.
* Customizing the training with composable building blocks that support different model architectures, parameter-efficient fine-tuning (PEFT) techniques, and more.
* Logging progress and metrics to gain insight into the training process.
* Quantizing the model post-tuning.
* Evaluating the fine-tuned model on popular benchmarks.
* Running local inference for testing fine-tuned models.
* Checkpoint compatibility with popular production inference systems.

To get started, jump right into the [code](https://www.github.com/pytorch/torchtune) or walk through our many [tutorials](https://pytorch.org/torchtune/main/)!


## Why torchtune?

Over the past year there has been an explosion of interest in open LLMs. Fine-tuning these state of the art models has emerged as a critical technique for adapting them to specific use cases. This adaptation can require extensive customization from dataset and model selection all the way through to quantization, evaluation and inference. Moreover, the size of these models poses a significant challenge when trying to fine-tune them on consumer-level GPUs with limited memory. 

Existing solutions make it hard to add these customizations or optimizations by hiding the necessary pieces behind layers of abstractions. It’s unclear how different components interact with each other and which of these need to be updated to add new functionality. torchtune empowers developers to adapt LLMs to their specific needs and constraints with full control and visibility.


## torchtune’s Design

torchtune was built with the following principles in mind

* **Easy extensibility** - New techniques emerge all the time and everyone’s fine-tuning use case is different. torchtune’s recipes are designed around easily composable components and hackable training loops, with minimal abstraction getting in the way of fine-tuning your fine-tuning. Each [recipe](https://github.com/pytorch/torchtune/tree/main/recipes) is self-contained - no trainers or frameworks, and is designed to be easy to read - less than 600 lines of code! 
* **Democratize fine-tuning** - Users, regardless of their level of expertise, should be able to use torchtune. Clone and modify configs, or get your hands dirty with some code! You also don’t need beefy data center GPUs. Our memory efficient recipes have been tested on machines with a single 24GB gaming GPU.
* **Interoperability with the OSS LLM ecosystem** - The open source LLM ecosystem is absolutely thriving, and torchtune takes advantage of this to provide interoperability with a wide range of offerings. This flexibility puts you firmly in control of how you train and use your fine-tuned models.

Over the next year, open LLMs will become even more powerful, with support for more languages (multilingual), more modalities (multimodal) and more tasks. As the complexity of these models increases, we need to pay the same attention to “how” we design our libraries as we do to the features provided or performance of a training run. Flexibility will be key to ensuring the community can maintain the current pace of innovation, and many libraries/tools will need to play well with each other to power the full spectrum of use cases. torchtune is built from the ground up with this future in mind.

In the true PyTorch spirit, torchtune makes it easy to get started by providing integrations with some of the most popular tools for working with LLMs.



* **[Hugging Face Hub](https://huggingface.co/docs/hub/en/index)** - Hugging Face provides an expansive repository of open source models and datasets for fine-tuning. torchtune seamlessly integrates through the `tune download` CLI command so you can get started right away with fine-tuning your first model.
* **[PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)** - Scale your training using PyTorch FSDP. It is very common for people to invest in machines with multiple consumer level cards like the 3090/4090 by NVidia. torchtune allows you to take advantage of these setups by providing distributed recipes powered by FSDP.
* **[Weights & Biases](https://wandb.ai/site)** - torchtune uses the Weights & Biases AI platform to log metrics and model checkpoints during training. Track your configs, metrics and models from your fine-tuning runs all in one place!
* **[EleutherAI’s LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)** - Evaluating fine-tuned models is critical to understanding whether fine-tuning is giving you the results you need. torchtune includes a simple evaluation recipe powered by EleutherAI’s LM Evaluation Harness to provide easy access to a comprehensive suite of standard LLM benchmarks. Given the importance of evaluation, we will be working with EleutherAI very closely in the next few months to build an even deeper and more “native” integration. 
* **[ExecuTorch](https://pytorch.org/executorch-overview)** - Models fine-tuned with torchtune can be [easily exported](https://github.com/pytorch/executorch/tree/main/examples/models/llama2#optional-finetuning) to ExecuTorch, enabling efficient inference to be run on a wide variety of mobile and edge devices.
* **[torchao](https://github.com/pytorch-labs/ao)** - Easily and efficiently quantize your fine-tuned models into 4-bit or 8-bit using a simple [post-training recipe](https://github.com/pytorch/torchtune/blob/main/recipes/quantize.py) powered by the quantization APIs from torchao.


## What’s Next?

This is just the beginning and we’re really excited to put this alpha version in front of a vibrant and energetic community. In the coming weeks, we’ll continue to augment the library with more models, features and fine-tuning techniques. We’d love to hear any feedback, comments or feature requests in the form of GitHub issues on our repository, or on our [Discord channel](https://discord.com/invite/4Xsdn8Rr9Q). As always, we’d love any contributions from this awesome community. Happy Tuning!