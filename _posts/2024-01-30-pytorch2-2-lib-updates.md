---
layout: blog_detail
title: "New Library Updates in PyTorch 2.2"
---

## Summary

We are bringing a number of improvements to the current PyTorch libraries, alongside the PyTorch 2.2 release. These updates demonstrate our focus on developing common and extensible APIs across all domains to make it easier for our community to build ecosystem projects on PyTorch.


<table class="table table-bordered">
  <tr>
   <td colspan="3" style="font-weight: 600; text-align: center;">Latest Stable Library Versions (<a href="https://pytorch.org/docs/stable/index.html">Full List</a>)*
   </td>
  </tr>
  <tr>
   <td>TorchArrow 0.1.0
   </td>
   <td>TorchRec 0.6.0
   </td>
   <td>TorchVision 0.17
   </td>
  </tr>
  <tr>
   <td>TorchAudio 2.2.0
   </td>
   <td>TorchServe 0.9.0
   </td>
   <td>TorchX 0.7.0
   </td>
  </tr>
  <tr>
   <td>TorchData 0.7.1 
   </td>
   <td>TorchText 0.17.0
   </td>
   <td>PyTorch on XLA Devices 2.1
   </td>
  </tr>
</table>


*To see [prior versions](https://pytorch.org/docs/stable/index.html) or (unstable) nightlies, click on versions in the top left menu above ‘Search Docs’.


## TorchRL  

### Feature: TorchRL’s Offline RL Data Hub

TorchRL now provides one of the largest dataset hubs for offline RL and imitation learning, and it all comes under a single data format (TED, for TorchRL Episode Data format). This makes it possible to easily swap from different sources in a single training loop. It is also now possible to easily combine datasets of different sources through the ReplayBufferEnsemble class. The data processing is fully customizable. Sources include simulated tasks (Minari, D4RL, VD4RL), robotic datasets (Roboset, OpenX Embodied dataset) and gaming (GenDGRL/ProcGen, Atari/DQN). Check these out in the [documentation](https://pytorch.org/rl/reference/data.html#datasets). 

Aside from these changes, our replay buffers can now be dumped on disk using the `.dumps()` method which will serialize the buffers on disk using the TensorDict API which is faster, safer and more efficient than using torch.save. 

Finally, replay buffers can now be  read and written from separate processes on the same machine without any extra code needed from the user!


### TorchRL2Gym environment API

To facilitate TorchRL’s integration in existing code-bases and enjoy all the features of TorchRL’s environment API (execution on device, batched operations, transforms…) we provide a TorchRL-to-gym API that allows users to register any environment they want in gym or gymnasium. This can be used in turn to make TorchRL a universal lib-to-gym converter that works across stateless (eg, dm_control) and stateless (Brax, Jumanji) environments. The feature is thoroughly detailed in the [doc](https://pytorch.org/rl/reference/generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase.register_gym). The info_dict reading API has also been improved.


### Environment speedups

We added the option of executing environments on a different environment than the one used to deliver data in ParallelEnv. We also speeded up the GymLikeEnv class to a level that now makes it competitive with gym itself.


### Scaling objectives

The most popular objectives for RLHF and training at scale (PPO and A2C) are now compatible with FSDP and DDP models!


## TensorDict


### Feature: MemoryMappedTensor to replace MemmapTensor

We provide a much more efficient mmap backend for TensorDict; MemoryMappedTensor, which directly subclasses torch.Tensor. It comes with a bunch of utils to be constructed, such as `from_tensor`, `empty` and many more. MemoryMappedTensor is now much safer and faster than its counterpart. The library remains fully compatible with the previous class to facilitate transition. 

We also introduce a new set of multithreaded serialization methods that make tensordict serialization highly competitive with torch.save, with serialization and deserialization speeds for LLMs more than [3x faster than with torch.save](https://github.com/pytorch/tensordict/pull/592#issuecomment-1850761831).


### Feature: Non-tensor data within TensorDict

It is not possible to carry non-tensor data through the `NonTensorData` tensorclass. This makes it possible to build tensordicts with metadata. The `memmap`-API is fully compatible with these values, allowing users to seamlessly serialize and deserialize such objects. To store non-tensor data in a tensordict, simply assign it using the `__setitem__` method.


### Efficiency improvements

Several methods runtime have been improved, such as unbind, split, map or even TensorDict instantiation. Check our [benchmarks](https://pytorch.org/tensordict/dev/bench/)!


## TorchRec/fbgemm_gpu


### VBE

TorchRec now natively supports VBE (variable batched embeddings) within the `EmbeddingBagCollection` module. This allows variable batch size per feature, unlocking sparse input data deduplication, which can greatly speed up embedding lookup and all-to-all time. To enable, simply initialize `KeyedJaggedTensor `with `stride_per_key_per_rank` and `inverse_indices` fields, which specify batch size per feature and inverse indices to reindex the embedding output respectively.

In addition to the TorchRec library changes, [fbgemm_gpu](https://pytorch.org/FBGEMM/) has added the support for variable batch size per feature in TBE. [VBE](https://github.com/pytorch/FBGEMM/pull/1752) is enabled on split TBE training for both weighted and unweighted cases. To use VBE, please make sure to use the latest fbgemm_gpu version.  


### Embedding offloading 

This technique refers to using CUDA UVM to cache ‘hot’ embeddings (i.e. store embedding tables on host memory with cache on HBM memory), and prefetching the cache. Embedding offloading allows running a larger model with fewer GPUs, while maintaining competitive performance. Use the prefetching pipeline ([PrefetchTrainPipelineSparseDist](https://github.com/pytorch/torchrec/blob/main/torchrec/distributed/train_pipeline.py?#L1056)) and pass in [per-table cache load factor](https://github.com/pytorch/torchrec/blob/main/torchrec/distributed/types.py#L457) and the [prefetch_pipeline](https://github.com/pytorch/torchrec/blob/main/torchrec/distributed/types.py#L460) flag through constraints in the planner to use this feature.

Fbgemm_gpu has introduced [UVM cache pipeline prefetching](https://github.com/pytorch/FBGEMM/pull/1893) in [v0.5.0](https://github.com/pytorch/FBGEMM/releases/tag/v0.5.0) for TBE performance speedup. This allows cache-insert to be executed in parallel with TBE forward/backward. To enable this feature, please be sure to use the latest fbgemm_gpu version.


### Trec.shard/shard_modules

These APIs replace embedding submodules with its sharded variant. The shard API applies to an individual embedding module while the shard_modules API replaces all embedding modules and won’t touch other non-embedding submodules.

Embedding sharding follows similar behavior to the prior TorchRec DistributedModuleParallel behavior, except the ShardedModules have been made composable, meaning the modules are backed by [TableBatchedEmbeddingSlices](https://github.com/pytorch/torchrec/blob/main/torchrec/distributed/composable/table_batched_embedding_slice.py#L15) which are views into the underlying TBE (including .grad). This means that fused parameters are now returned with named_parameters(), including in DistributedModuleParallel.


## TorchVision 


### The V2 transforms are now stable!


The `torchvision.transforms.v2` namespace was still in BETA stage until now. It is now stable! Whether you’re new to Torchvision transforms, or you’re already experienced with them, we encourage you to start with [Getting started with transforms v2](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py) in order to learn more about what can be done with the new v2 transforms.

Browse our [main docs](https://pytorch.org/vision/stable/transforms.html#) for general information and performance tips. The available transforms and functionals are listed in the [API reference](https://pytorch.org/vision/stable/transforms.html#v2-api-ref). Additional information and tutorials can also be found in our [example gallery](https://pytorch.org/vision/stable/auto_examples/index.html#gallery), e.g. [Transforms v2: End-to-end object detection/segmentation example](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html#sphx-glr-auto-examples-transforms-plot-transforms-e2e-py) or [How to write your own v2 transforms](https://pytorch.org/vision/stable/auto_examples/transforms/plot_custom_transforms.html#sphx-glr-auto-examples-transforms-plot-custom-transforms-py).


### Towards `torch.compile()` support

We are progressively adding support for `torch.compile()` to torchvision interfaces, reducing graph breaks and allowing dynamic shape.

The torchvision ops (`nms`, `[ps_]roi_align`, `[ps_]roi_pool` and `deform_conv_2d`) are now compatible with `torch.compile` and dynamic shapes.

On the transforms side, the majority of [low-level kernels](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/__init__.py) (like `resize_image()` or `crop_image()`) should compile properly without graph breaks and with dynamic shapes. We are still addressing the remaining edge-cases, moving up towards full functional support and classes, and you should expect more progress on that front with the next release.
