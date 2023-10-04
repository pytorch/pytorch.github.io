---
layout: blog_detail
title: "New Library Updates in PyTorch 2.1"
author: Team PyTorch
---

## **Summary**

We are bringing a number of improvements to the current PyTorch libraries, alongside the PyTorch 2.1 release. These updates demonstrate our focus on developing common and extensible APIs across all domains to make it easier for our community to build ecosystem projects on PyTorch. 

Along with 2.1, we are also releasing a series of beta updates to the PyTorch domain libraries including TorchAudio and TorchVision. Please find the list of the latest stable versions and updates below.

| Latest Stable Library Versions  |([Full List](https://pytorch.org/docs/stable/index.html))*                  |                             |
|--------------------------------------------|------------------|-----------------------------|
| TorchArrow 0.1.0                           | TorchRec 0.5.0   | TorchVision 0.16            |
| TorchAudio 2.1                             | TorchServe 0.8.2 | TorchX 0.5.0                |
| TorchData 0.7.0                            | TorchText 0.16.0 | PyTorch on XLA Devices 1.14 |

\*To see [prior versions](https://pytorch.org/docs/stable/index.html) or (unstable) nightlies, click on versions in the top left menu above ‘Search Docs’.

## **TorchAudio**

TorchAudio v2.1 introduces the following new features and backward-incompatible changes:

**\[Beta] A new API to apply filter, effects and codec**

\`torchaudio.io.AudioEffector\` can apply filters, effects and encodings to waveforms in online/offline fashion. You can use it as a form of augmentation.

Please refer to <https://pytorch.org/audio/2.1/tutorials/effector_tutorial.html> for the usage and examples.

**\[Beta] Tools for Forced alignment**

New functions and a pre-trained model for forced alignment were added. \`torchaudio.functional.forced\_align\` computes alignment from an emission and \`torchaudio.pipelines.MMS\_FA\` provides access to the model trained for multilingual forced alignment in [MMS: Scaling Speech Technology to 1000+ languages](https://ai.meta.com/blog/multilingual-model-speech-recognition/) project.

Please refer to <https://pytorch.org/audio/2.1/tutorials/ctc_forced_alignment_api_tutorial.html> for the usage of \`forced\_align\` function, and <https://pytorch.org/audio/2.1/tutorials/forced_alignment_for_multilingual_data_tutorial.html> for how one can use \`MMS\_FA\` to align transcript in multiple languages.

**\[Beta] TorchAudio-Squim : Models for reference-free speech assessment**

Model architectures and pre-trained models from the paper [TorchAudio-Sequim: Reference-less Speech Quality and Intelligibility measures in TorchAudio](https://arxiv.org/abs/2304.01448) were added.

You can use the pre-trained models \`torchaudio.pipelines.SQUIM\_SUBJECTIVE\` and \`torchaudio.pipelines.SQUIM\_OBJECTIVE\`. They can estimate the various speech quality and intelligibility metrics (e.g. STOI, wideband PESQ, Si-SDR, and MOS). This is helpful when evaluating the quality of speech generation models, such as Text-to-Speech (TTS).

Please refer to <https://pytorch.org/audio/2.1/tutorials/squim_tutorial.html> for the details.

**\[Beta] CUDA-based CTC decoder**

\`torchaudio.models.decoder.CUCTCDecoder\` performs CTC beam search in CUDA devices. The beam search is fast. It eliminates the need to move data from CUDA device to CPU when performing automatic speech recognition. With PyTorch's CUDA support, it is now possible to perform the entire speech recognition pipeline in CUDA.

Please refer to <https://pytorch.org/audio/2.1/tutorials/asr_inference_with_cuda_ctc_decoder_tutorial.html> for the detail.

**\[Prototype] Utilities for AI music generation**

We are working to add utilities that are relevant to music AI. Since the last release, the following APIs were added to the prototype.

Please refer to respective documentation for the usage.
- [torchaudio.prototype.chroma\_filterbank](https://pytorch.org/audio/main/generated/torchaudio.prototype.functional.chroma_filterbank.html)
- [torchaudio.prototype.transforms.ChromaScale](https://pytorch.org/audio/main/generated/torchaudio.prototype.transforms.ChromaScale.html)
- [torchaudio.prototype.transforms.ChromaSpectrogram](https://pytorch.org/audio/main/generated/torchaudio.prototype.transforms.ChromaSpectrogram.html)
- [torchaudio.prototype.pipelines.VGGISH](https://pytorch.org/audio/main/generated/torchaudio.prototype.pipelines.VGGISH.html)

**New recipes for training models**

Recipes for Audio-visual ASR, multi-channel DNN beamforming and TCPGen context-biasing were added.

Please refer to the recipes
- <https://github.com/pytorch/audio/tree/release/2.1/examples/avsr>
- <https://github.com/pytorch/audio/tree/release/2.1/examples/dnn_beamformer>
- <https://github.com/pytorch/audio/tree/release/2.1/examples/asr/librispeech_conformer_rnnt_biasing>

**Update to FFmpeg support**

The version of supported FFmpeg libraries was updated. TorchAudio v2.1 works with FFmpeg 6, 5 and 4.4. The support for 4.3, 4.2 and 4.1 are dropped.

Please refer to <https://pytorch.org/audio/2.1/installation.html#optional-dependencies> for the detail of the new FFmpeg integration mechanism.

**Update to libsox integration**

TorchAudio now depends on libsox installed separately from torchaudio. Sox I/O backend no longer supports file-like objects. (This is supported by FFmpeg backend and soundfile.)

Please refer to <https://pytorch.org/audio/2.1/installation.html#optional-dependencies> for the details.

## TorchRL

Our RLHF components make it easy to build an RLHF training loop with limited RL knowledge. TensorDict enables an easy interaction between datasets (eg, HF datasets) and RL models. The new algorithms we provide deliver a wide range of solutions for offline RL training, which is more data efficient.

Through RoboHive and IsaacGym, TorchRL now provides a built-in interface with hardware (robots), tying training at scale with policy deployment on device. Thanks to SMAC, VMAS, and PettingZoo and related MARL-oriented losses, TorchRL is now fully capable of training complex policies in multi-agent settings.

**New algorithms**
- \[BETA] We integrate some RLHF components and examples: we provide building blocks for data formatting in RL frameworks, reward model design, specific transforms that enable efficient learning (eg. KL correction) and training scripts
- \[Stable] New algorithms include Decision transformers, CQL, multi-agent losses such as MAPPO and QMixer.**New features**- \[Stable] New transforms such as Visual Cortex 1 (VC1), a foundational model for RL. 
- We widened the panel of library covered by TorchRL: 
  - \[Beta] IsaacGym, a powerful GPU-based simulator that allows interaction and rendering of thousands of vectorized environments by NVIDIA.
  - \[Stable] PettingZoo, a multi-agent library by the Farama Foundation.
  - \[Stable] SMAC-v2, the new Starcraft Multi-agent simulator
  - \[Stable] RoboHive, a collection of environments/tasks simulated with the MuJoCo physics engine.
  
**Performance improvements**

We provide faster data collection through refactoring and integration of SB3 and Gym asynchronous environments execution. We also made our value functions faster to execute.

## TorchRec

**\[Prototype] Zero Collision / Managed Collision Embedding Bags**

A common constraint in Recommender Systems is the sparse id input range is larger than the number of embeddings the model can learn for a given parameter size.   To resolve this issue, the conventional solution is to hash sparse ids into the same size range as the embedding table.  This will ultimately lead to hash collisions, with multiple sparse ids sharing the same embedding space.   We have developed a performant alternative algorithm that attempts to address this problem by tracking the _N_ most common sparse ids and ensuring that they have a unique embedding representation. The module is defined [here](https://github.com/pytorch/torchrec/blob/b992eebd80e8ccfc3b96a7fd39cb072c17e8907d/torchrec/modules/mc_embedding_modules.py#L26) and an example can be found [here](https://github.com/pytorch/torchrec/blob/b992eebd80e8ccfc3b96a7fd39cb072c17e8907d/torchrec/modules/mc_embedding_modules.py#L26).

**\[Prototype] UVM Caching - Prefetch Training Pipeline**

For tables where on-device memory is insufficient to hold the entire embedding table, it is common to leverage a caching architecture where part of the embedding table is cached on device and the full embedding table is on host memory (typically DDR SDRAM).   However, in practice, caching misses are common, and hurt performance due to relatively high latency of going to host memory.   Building on TorchRec’s existing data pipelining, we developed a new [_Prefetch Training Pipeline_](https://pytorch.org/torchrec/torchrec.distributed.html#torchrec.distributed.train_pipeline.PrefetchPipelinedForward) to avoid these cache misses by prefetching the relevant embeddings for upcoming batch from host memory, effectively eliminating cache misses in the forward path.

## TorchVision 
### **Transforms and augmentations**

**Major speedups**

The new transforms in `torchvision.transforms.v2` are now[ 10%-40% faster](https://github.com/pytorch/vision/issues/7497#issuecomment-1557478635) than before! This is mostly achieved thanks to 2X-4X improvements made to `v2.Resize()`, which now supports native `uint8` tensors for Bilinear and Bicubic mode. Output results are also now closer to PIL's! Check out our[ performance recommendations](https://pytorch.org/vision/stable/transforms.html#performance-considerations) to learn more.

Additionally, `torchvision` now ships with `libjpeg-turbo` instead of `libjpeg`, which should significantly speed-up the jpeg decoding utilities ([`read_image`](https://pytorch.org/vision/stable/generated/torchvision.io.read_image.html#torchvision.io.read_image),[ `decode_jpeg`](https://pytorch.org/vision/stable/generated/torchvision.io.read_image.html#torchvision.io.decode_jpeg)), and avoid compatibility issues with PIL.

**CutMix and MixUp**

Long-awaited support for the `CutMix` and `MixUp` augmentations is now here! Check[ our tutorial](https://pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html#sphx-glr-auto-examples-transforms-plot-cutmix-mixup-py) to learn how to use them.

**Towards stable V2 transforms**

In the[ previous release 0.15](https://github.com/pytorch/vision/releases/tag/v0.15.1) we BETA-released a new set of transforms in `torchvision.transforms.v2` with native support for tasks like segmentation, detection, or videos. We have now stabilized the design decisions of these transforms and made further improvements in terms of speedups, usability, new transforms support, etc.

We're keeping the `torchvision.transforms.v2` and `torchvision.tv_tensors` namespaces as BETA until 0.17 out of precaution, but we do not expect disruptive API changes in the future.

Whether you’re new to Torchvision transforms, or you’re already experienced with them, we encourage you to start with[ Getting started with transforms v2](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py) in order to learn more about what can be done with the new v2 transforms.

Browse our[ main docs](https://pytorch.org/vision/stable/transforms.html#) for general information and performance tips. The available transforms and functionals are listed in the[ API reference](https://pytorch.org/vision/stable/transforms.html#v2-api-ref). Additional information and tutorials can also be found in our[ example gallery](https://pytorch.org/vision/stable/auto_examples/index.html#gallery), e.g.[ Transforms v2: End-to-end object detection/segmentation example](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html#sphx-glr-auto-examples-transforms-plot-transforms-e2e-py) or[ How to write your own v2 transforms](https://pytorch.org/vision/stable/auto_examples/transforms/plot_custom_transforms.html#sphx-glr-auto-examples-transforms-plot-custom-transforms-py).

### \[BETA] MPS support

The `nms` and roi-align kernels (`roi_align`, `roi_pool`, `ps_roi_align`, `ps_roi_pool`) now support MPS. Thanks to[ Li-Huai (Allan) Lin](https://github.com/qqaatw) for this contribution!

## TorchX

**Schedulers**
- \[Prototype] Kubernetes MCAD Scheduler: Integration for easily scheduling jobs on Multi-Cluster-Application-Dispatcher (MCAD)

- AWS Batch 

  - Add privileged option to enable running containers on EFA enabled instances with elevated networking permissions
  
### **TorchX Tracker**
- \[Prototype] MLFlow backend for TorchX Tracker: in addition to _fsspec_ based tracker, TorchX can use MLFlow instance to track metadata/experiments 

**Components**
- _dist.spmd_ component to support Single-Process-Multiple-Data style applications

**Workspace**
- Add ability to access image and workspace path from Dockerfile while building docker workspace

Release includes number of other bugfixes.

To learn more about Torchx visit <https://pytorch.org/torchx/latest/>

## TorchText and TorchData

As of September 2023 we have paused active development of TorchText and TorchData as we re-evaluate how we want to serve the needs of the community in this space.
