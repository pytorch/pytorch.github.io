---
layout: blog_detail
title: 'PyTorch Adds New Ecosystem Projects for Encrypted AI and Quantum Computing, Expands PyTorch Hub'
author: Team PyTorch
---

The PyTorch ecosystem includes projects, tools, models and libraries from a broad community of researchers in academia and industry, application developers, and ML engineers. The goal of this ecosystem is to support, accelerate, and aid in your exploration with PyTorch and help you push the state of the art, no matter what field you are exploring. Similarly, we are expanding the recently launched PyTorch Hub to further help you discover and reproduce the latest research.

In this post, we’ll highlight some of the projects that have been added to the PyTorch ecosystem this year and provide some context on the criteria we use to evaluate community projects. We’ll also provide an update on the fast-growing PyTorch Hub and share details on our upcoming PyTorch Summer Hackathon.

<div class="text-center">
  <img src="{{ site.url }}/assets/images/pytorch-ecosystem.png" width="100%">
</div>

## Recently added ecosystem projects

From private AI to quantum computing, we’ve seen the community continue to expand into new and interesting areas. The latest projects include:

- [Advertorch](https://github.com/BorealisAI/advertorch): A Python toolbox for adversarial robustness research. The primary functionalities are implemented in PyTorch. Specifically, AdverTorch contains modules for generating adversarial perturbations and defending against adversarial examples, as well as scripts for adversarial training.

- [botorch](https://botorch.org/): A modular and easily extensible interface for composing Bayesian optimization primitives, including probabilistic models, acquisition functions, and optimizers.

- [Skorch](https://github.com/skorch-dev/skorch): A high-level library for PyTorch that provides full scikit-learn compatibility.

- [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric): A library for deep learning on irregular input data such as graphs, point clouds, and manifolds.

- [PySyft](https://github.com/OpenMined/PySyft): A Python library for encrypted, privacy preserving deep learning.

- [PennyLane](https://pennylane.ai/): A library for quantum ML, automatic differentiation, and optimization of hybrid quantum-classical computations.

- [Flair](https://github.com/zalandoresearch/flair): A very simple framework for state-of-the-art natural language processing (NLP).

### What makes a great project?

When we review project submissions for the PyTorch ecosystem, we take into account a number of factors that we feel are important and that we would want in the projects we use ourselves. Some of these criteria include:

1. *Well-tested:* Users should be confident that ecosystem projects will work well with PyTorch, and include support for CI to ensure that testing is occurring on a continuous basis and the project can run on the latest version of PyTorch.
2. *Clear utility:* Users should understand where each project fits within the PyTorch ecosystem and the value it brings.
3. *Permissive licensing:* Users must be able to utilize ecosystem projects without licensing concerns. e.g. BSD-3, Apache-2 and MIT licenses
4. *Easy onboarding:* Projects need to have support for binary installation options (pip/Conda), clear documentation and a rich set of tutorials (ideally built into Jupyter notebooks).
5. *Ongoing maintenance:* Project authors need to be committed to supporting and maintaining their projects.
6. *Community:* Projects should have (or be on track to building) an active, broad-based community.

If you would like to have your project included in the PyTorch ecosystem and featured on [pytorch.org/ecosystem](http://pytorch.org/ecosystem), please complete the form [here](https://pytorch.org/ecosystem/join). If you've previously submitted a project for consideration and haven't heard back, we promise to get back to you as soon as we can - we've received a lot of submissions!

## PyTorch Hub for reproducible research | New models

Since [launching](https://pytorch.org/blog/towards-reproducible-research-with-pytorch-hub/) the PyTorch Hub in beta, we’ve received a lot of interest from the community including the contribution of many new models. Some of the latest include [U-Net for Brain MRI](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/) contributed by researchers at Duke University, [Single Shot Detection](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/) from NVIDIA and Transformer-XL from HuggingFace.

We’ve seen organic integration of the PyTorch Hub by folks like [paperswithcode](https://paperswithcode.com/), making it even easier for you to try out the state of the art in AI research. In addition, companies like [Seldon](https://github.com/axsaucedo/seldon-core/tree/pytorch_hub/examples/models/pytorchhub) provide production-level support for PyTorch Hub models on top of Kubernetes.

### What are the benefits of contributing a model in the PyTorch Hub?

- *Compatibility:* PyTorch Hub models are prioritized first for testing by the TorchScript and Cloud TPU teams, and used as baselines for researchers across a number of fields.

- *Visibility:* Models in the Hub will be promoted on [pytorch.org](http://pytorch.org/) as well as on [paperswithcode](https://paperswithcode.com/).

- *Ease of testing and reproducibility:* Each model comes with code, clear preprocessing requirements, and methods/dependencies to run. There is also tight integration with [Google Colab](https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/facebookresearch_WSL-Images_resnext.ipynb#scrollTo=LM_l7vXJvnDM), making it a true single click to get started.

### PyTorch Hub contributions welcome!

We are actively looking to grow the PyTorch Hub and welcome contributions. You don’t need to be an original paper author to contribute, and we’d love to see the number of domains and fields broaden. So what types of contributions are we looking for?

- Artifacts of a published or an arXiv paper (or something of a similar nature that serves a different audience — such as ULMFit) that a large audience would need.

  AND

- Reproduces the published results (or better)

Overall these models are aimed at researchers either trying to reproduce a baseline, or trying to build downstream research on top of the model (such as feature-extraction or fine-tuning) as well as researchers looking for a demo of the paper for subjective evaluation. Please keep this audience in mind when contributing.

If you are short on inspiration or would just like to find out what the SOTA is an any given field or domain, checkout the Paperswithcode [state-of-the-art gallery](https://paperswithcode.com/sota).

## PyTorch Summer Hackathon

We’ll be hosting the first PyTorch Summer Hackathon next month. We invite you to apply to participate in the in-person hackathon on  August 8th to 9th at Facebook's Menlo Park campus. We'll be bringing the community together to work on innovative ML projects that can solve a broad range of complex challenges.

Applications will be reviewed and accepted on a rolling basis until spaces are filled. For those who cannot join this Hackathon in person, we’ll be following up soon with other ways to participate.

Please visit [this link to apply](https://www.eventbrite.com/e/pytorch-summer-hackathon-in-menlo-park-registration-63756668913).

Thank you for being part of the PyTorch community!

-Team PyTorch
