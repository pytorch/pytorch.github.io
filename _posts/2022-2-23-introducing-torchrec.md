---
layout: blog_detail
title: 'Introducing TorchRec, a library for modern production recommendation systems'
author: Meta AI - Donny Greenberg, Colin Taylor, Dmytro Ivchenko, Xing Liu, Anirudh Sudarshan 
featured-img: ''
---

We are excited to announce [TorchRec](https://github.com/pytorch/torchrec), a PyTorch domain library for Recommendation Systems. This new library provides common sparsity and parallelism primitives, enabling researchers to build state-of-the-art personalization models and deploy them in production.

<p align="left">
  <img src="/assets/images/introducing-torchrec/torchrec_lockup.png" width="40%">
</p>

## How did we get here?
Recommendation Systems (RecSys) comprise a large footprint of production-deployed AI today, but you might not know it from looking at Github. Unlike areas like Vision and NLP, much of the ongoing innovation and development in RecSys is behind closed company doors. For academic researchers studying these techniques or companies building personalized user experiences, the field is far from democratized. Further, RecSys as an area is largely defined by learning models over sparse and/or sequential events, which has large overlaps with other areas of AI. Many of the techniques are transferable, particularly for scaling and distributed execution. A large portion of the global investment in AI is in developing these RecSys techniques, so cordoning them off blocks this investment from flowing into the broader AI field.

By mid-2020, the PyTorch team received a lot of feedback that there hasn't been a large-scale  production-quality recommender systems package in the open-source PyTorch ecosystem. While we were trying to find a good answer, a group of engineers at Meta wanted to contribute Meta’s production RecSys stack as a PyTorch domain library, with a strong commitment to growing an ecosystem around it. This seemed like a good idea that benefits researchers and companies across the RecSys domain. So, starting from Meta’s stack, we began modularizing and designing a fully-scalable codebase that is adaptable for diverse recommendation use-cases. Our goal was to extract the key building blocks from across Meta’s software stack to simultaneously enable creative exploration and scale. After nearly two years, a battery of benchmarks, migrations, and testing across Meta, we’re excited to finally embark on this journey together with the RecSys community. We want this package to open a dialogue and collaboration across the RecSys industry, starting with Meta as the first sizable contributor.


## Introducing TorchRec
TorchRec includes a scalable low-level modeling foundation alongside rich batteries-included modules. We initially target “two-tower” ([[1]], [[2]]) architectures that have separate submodules to learn representations of candidate items and the query or context. Input signals can be a mix of floating point “dense” features or high-cardinality categorical “sparse” features that require large embedding tables to be trained. Efficient training of such architectures involves combining data parallelism that replicates the “dense” part of computation and model parallelism that  partitions large embedding tables across many nodes.

In particular, the library includes:
- Modeling primitives, such as embedding bags and jagged tensors, that enable easy authoring of large, performant multi-device/multi-node models using hybrid data-parallelism and model-parallelism.
- Optimized RecSys kernels powered by [FBGEMM](https://github.com/pytorch/FBGEMM) , including support for sparse and quantized operations.
- A sharder which can partition embedding tables with a variety of different strategies including data-parallel, table-wise, row-wise, table-wise-row-wise, and column-wise sharding.
- A planner which can automatically generate optimized sharding plans for models.
- Pipelining to overlap dataloading device transfer (copy to GPU), inter-device communications (input_dist), and computation (forward, backward) for increased performance.
- GPU inference support.
- Common modules for RecSys, such as models and public datasets (Criteo & Movielens).

To showcase the flexibility of this tooling, let’s look at the following code snippet, pulled from our DLRM Event Prediction example:
```python
# Specify the sparse embedding layers
eb_configs = [
   EmbeddingBagConfig(
       name=f"t_{feature_name}",
       embedding_dim=64,
       num_embeddings=100_000,
       feature_names=[feature_name],
   )
   for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
]

# Import and instantiate the model with the embedding configuration
# The "meta" device indicates lazy instantiation, with no memory allocated
train_model = DLRM(
   embedding_bag_collection=EmbeddingBagCollection(
       tables=eb_configs, device=torch.device("meta")
   ),
   dense_in_features=len(DEFAULT_INT_NAMES),
   dense_arch_layer_sizes=[512, 256, 64],
   over_arch_layer_sizes=[512, 512, 256, 1],
   dense_device=device,
)

# Distribute the model over many devices, just as one would with DDP.
model = DistributedModelParallel(
   module=train_model,
   device=device,
)

optimizer = torch.optim.SGD(params, lr=args.learning_rate)
# Optimize the model in a standard loop just as you would any other model!
# Or, you can use the pipeliner to synchronize communication and compute
for epoch in range(epochs):
   # Train
```


## Scaling Performance
TorchRec has state-of-the-art infrastructure for scaled Recommendations AI, powering some of the largest models at Meta. It was used to train a 1.25 trillion parameter model, pushed to production in January, and a 3 trillion parameter model which will be in production soon. This should be a good indication that PyTorch is fully capable of the largest scale RecSys problems in industry. We’ve heard from many in the community that sharded embeddings are a pain point. TorchRec cleanly addresses that. Unfortunately it is challenging to provide large-scale benchmarks with public datasets, as most open-source benchmarks are too small to show performance at scale.


## Looking ahead
Open-source and open-technology have universal benefits. Meta is seeding the PyTorch community with a state-of-the-art RecSys package, with the hope that many join in on building it forward, enabling new research and helping many companies. The team behind TorchRec plan to continue this program indefinitely, building up TorchRec to meet the needs of the RecSys community, to welcome new contributors, and to continue to power personalization at Meta. We’re excited to begin this journey and look forward to contributions, ideas, and feedback!


## References
[[1]] Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations

[[2]] DLRM: An advanced, open source deep learning recommendation model


[1]: https://research.google/pubs/pub48840/
[2]: https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/
