---
layout: blog_detail
title: "New Library Updates in PyTorch 2.0"
---

## Summary

We are bringing a number of improvements to the current PyTorch libraries, alongside the [PyTorch 2.0 release](/blog/pytorch-2.0-release/). These updates demonstrate our focus on developing common and extensible APIs across all domains to make it easier for our community to build ecosystem projects on PyTorch. 

Along with 2.0, we are also releasing a series of beta updates to the PyTorch domain libraries, including those that are in-tree, and separate libraries including TorchAudio, TorchVision, and TorchText. An update for TorchX is also being released as it moves to community supported mode. Please find the list of the latest stable versions and updates below.

**Latest Stable Library Versions (<a href="https://pytorch.org/docs/stable/index.html">Full List</a>)**
<table class="table table-bordered">
  <tr>
   <td>TorchArrow 0.1.0
   </td>
   <td>TorchRec 0.4.0
   </td>
   <td>TorchVision 0.15
   </td>
  </tr>
  <tr>
   <td>TorchAudio 2.0
   </td>
   <td>TorchServe 0.7.1
   </td>
   <td>TorchX 0.4.0
   </td>
  </tr>
  <tr>
   <td>TorchData 0.6.0
   </td>
   <td>TorchText 0.15.0
   </td>
   <td>PyTorch on XLA Devices 1.14
   </td>
  </tr>
</table>


*To see [prior versions](https://pytorch.org/docs/stable/index.html) or (unstable) nightlies, click on versions in the top left menu above ‘Search Docs’.


## TorchAudio 

### [Beta] Data augmentation operators

The release adds several data augmentation operators under torchaudio.functional and torchaudio.transforms:
* torchaudio.functional.add_noise
* torchaudio.functional.convolve
* torchaudio.functional.deemphasis
* torchaudio.functional.fftconvolve
* torchaudio.functional.preemphasis
* torchaudio.functional.speed
* torchaudio.transforms.AddNoise
* torchaudio.transforms.Convolve
* torchaudio.transforms.Deemphasis
* torchaudio.transforms.FFTConvolve
* torchaudio.transforms.Preemphasis
* torchaudio.transforms.Speed
* torchaudio.transforms.SpeedPerturbation

The operators can be used to synthetically diversify training data to improve the generalizability of downstream models.

For usage details, please refer to the [functional](https://pytorch.org/audio/2.0.0/functional.html) and [transform](https://pytorch.org/audio/2.0.0/transforms.html) documentation and [Audio Data Augmentation](https://pytorch.org/audio/2.0.0/tutorials/audio_data_augmentation_tutorial.html) tutorial.


### [Beta] WavLM and XLS-R models

The release adds two self-supervised learning models for speech and audio.

* [WavLM](https://ieeexplore.ieee.org/document/9814838) that is robust to noise and reverberation.
* [XLS-R](https://arxiv.org/abs/2111.09296) that is trained on cross-lingual datasets.

Besides the model architectures, torchaudio also supports corresponding pre-trained pipelines:

* torchaudio.pipelines.WAVLM_BASE
* torchaudio.pipelines.WAVLM_BASE_PLUS
* torchaudio.pipelines.WAVLM_LARGE
* torchaudio.pipelines.WAV2VEC_XLSR_300M
* torchaudio.pipelines.WAV2VEC_XLSR_1B
* torchaudio.pipelines.WAV2VEC_XLSR_2B

For usage details, please refer to the [factory function](https://pytorch.org/audio/2.0.0/generated/torchaudio.models.Wav2Vec2Model.html#factory-functions) and [pre-trained pipelines](https://pytorch.org/audio/2.0.0/pipelines.html#id3) documentation.


## TorchRL  

The initial release of torchrl includes several features that span across the entire RL domain. TorchRL can already be used in online, offline, multi-agent, multi-task and distributed RL settings, among others. See below:


### [Beta] Environment wrappers and transforms

torchrl.envs includes several wrappers around common environment libraries. This allows users to swap one library with another without effort. These wrappers build an interface between these simulators and torchrl:

* dm_control: 
* Gym
* Brax
* EnvPool
* Jumanji
* Habitat

It also comes with many commonly used transforms and vectorized environment utilities that allow for a fast execution across simulation libraries. Please refer to the [documentation](https://pytorch.org/rl/reference/envs.html) for more detail.


### [Beta] Datacollectors

Data collection in RL is made easy via the usage of single process or multiprocessed/distributed data collectors that execute the policy in the environment over a desired duration and deliver samples according to the user’s needs. These can be found in torchrl.collectors and are documented [here](https://pytorch.org/rl/reference/collectors.html).


### [Beta] Objective modules

Several objective functions are included in torchrl.objectives, among which: 

* A generic PPOLoss class and derived ClipPPOLoss and KLPPOLoss
* SACLoss and DiscreteSACLoss
* DDPGLoss
* DQNLoss
* REDQLoss
* A2CLoss
* TD3Loss
* ReinforceLoss
* Dreamer

Vectorized value function operators also appear in the library. Check the documentation [here](https://pytorch.org/rl/reference/objectives.html).


### [Beta] Models and exploration strategies

We provide multiple models, modules and exploration strategies. Get a detailed description in [the doc](https://pytorch.org/rl/reference/modules.html).


### [Beta] Composable replay buffer

A composable replay buffer class is provided that can be used to store data in multiple contexts including single and multi-agent, on and off-policy and many more.. Components include:

* Storages (list, physical or memory-based contiguous storages)
* Samplers (Prioritized, sampler without repetition)
* Writers
* Possibility to add transforms

Replay buffers and other data utilities are documented [here](https://pytorch.org/rl/reference/data.html).


### [Beta] Logging tools and trainer

We support multiple logging tools including tensorboard, wandb and mlflow.

We provide a generic Trainer class that allows for easy code recycling and checkpointing.

These features are documented [here](https://pytorch.org/rl/reference/trainers.html).


## TensorDict

TensorDict is a new data carrier for PyTorch.


### [Beta] TensorDict: specialized dictionary for PyTorch

TensorDict allows you to execute many common operations across batches of tensors carried by a single container. TensorDict supports many shape and device or storage operations, and  can readily be used in distributed settings. Check the [documentation](https://pytorch.org/tensordict/) to know more.


### [Beta] @tensorclass: a dataclass for PyTorch

Like TensorDict, [tensorclass](https://pytorch.org/tensordict/reference/prototype.html) provides the opportunity to write dataclasses with built-in torch features such as shape or device operations. 


### [Beta] tensordict.nn: specialized modules for TensorDict

The [tensordict.nn module](https://pytorch.org/tensordict/reference/nn.html) provides specialized nn.Module subclasses that make it easy to build arbitrarily complex graphs that can be executed with TensorDict inputs. It is compatible with the latest PyTorch features such as functorch, torch.fx and torch.compile.


## TorchRec


### [Beta] KeyedJaggedTensor All-to-All Redesign and Input Dist Fusion

We observed performance regression due to a bottleneck in sparse data distribution for models that have multiple, large KJTs to redistribute. 

To combat this we altered the comms pattern to transport the minimum data required in the initial collective to support the collective calls for the actual KJT tensor data. This data sent in the initial collective, ‘splits’ means more data is transmitted over the comms stream overall, but the CPU is blocked for significantly shorter amounts of time leading to better overall QPS.

Furthermore, we altered the TorchRec train pipeline to group the initial collective calls for the splits together before launching the more expensive KJT tensor collective calls. This fusion minimizes the CPU blocked time as launching each subsequent input distribution is no longer dependent on the previous input distribution.

With this feature, variable batch sizes are now natively supported across ranks. These features are documented [here](https://github.com/pytorch/torchrec/commit/d0d23bef8aef5a79a1061fbc842c97bb68b91463).


## TorchVision 


### [Beta] Extending TorchVision’s Transforms to Object Detection, Segmentation & Video tasks 

TorchVision is extending its Transforms API! Here is what’s new:

* You can use them not only for Image Classification but also for Object Detection, Instance & Semantic Segmentation and Video Classification.
* You can use new functional transforms for transforming Videos, Bounding Boxes and Segmentation Masks.

Learn more about these new transforms [from our docs](https://pytorch.org/vision/stable/auto_examples/), and submit any feedback in our [dedicated issue](https://github.com/pytorch/vision/issues/6753).


## TorchText 

### [Beta] Adding scriptable T5 and Flan-T5 to the TorchText library with incremental decoding support!

TorchText has added the T5 model architecture with pre-trained weights for both the [original T5 paper](https://arxiv.org/abs/1910.10683) and [Flan-T5](https://arxiv.org/abs/2210.11416). The model is fully torchscriptable and features an optimized [multiheaded attention implementation](https://pytorch.org/docs/master/generated/torch.ao.nn.quantizable.MultiheadAttention.html?highlight=multihead#torch.ao.nn.quantizable.MultiheadAttention). We include several examples of how to utilize the model including summarization, classification, and translation.

For more details, please refer to [our docs](https://pytorch.org/text/stable/models.html).


## TorchX

TorchX is moving to community supported mode. More details will be coming in at a later time.