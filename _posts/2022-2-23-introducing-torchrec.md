---
layout: blog_detail
title: 'Introducing TorchRec, a library for modern production recommendation systems'
author: Donny Greenberg, Colin Taylor, Dmytro Ivchenko, Meta AI
featured-img: ''
---

We are excited to announce TorchRec (https://github.com/pytorch/torchrec), a PyTorch domain library for Recommendation Systems. This new library provides common sparsity and parallelism primitives, enabling researchers to build state-of-the-art personalization models and deploy them in production.

<p align="center">
  <img src="/assets/images/introducing-torchrec/torchrec_lookup.png" width="60%">
</p>

## How did we get here?
Recommendation Systems (RecSys) comprise a large footprint of production-deployed AI today, but you might not know it from looking at Github. Unlike areas like Vision and NLP, much of the ongoing innovation and development in RecSys is behind closed company doors. For academic researchers studying these techniques or companies building personalized user experiences, the field is far from democratized. Further, RecSys as an area is largely defined by learning models over sparse and/or sequential events, which has large overlaps with other areas of AI. Many of the techniques are transferable, particularly for scaling and distributed execution. A large portion of the global investment in AI is in developing these RecSys techniques, so cordoning them off blocks this investment from flowing into the broader AI field.

By mid-2020, the PyTorch team received a lot of feedback that there hasn't been a large-scale  production-quality recommender systems package in the open-source PyTorch ecosystem. While we were trying to find a good answer, a group of engineers at Meta wanted to contribute Meta’s production RecSys stack as a PyTorch domain library, with a strong commitment to growing an ecosystem around it. This seemed like a good idea that benefits researchers and companies across the RecSys domain. So, starting from Meta’s stack, we began modularizing and designing a fully-scalable codebase that is adaptable for diverse recommendation use-cases. Our goal was to extract the key building blocks from across Meta’s software stack to simultaneously enable creative exploration and scale. After nearly two years, a battery of benchmarks, migrations, and testing across Meta, we’re excited to finally embark on this journey together with the RecSys community. We want this package to open a dialogue and collaboration across the RecSys industry, starting with Meta as the first sizable contributor.


