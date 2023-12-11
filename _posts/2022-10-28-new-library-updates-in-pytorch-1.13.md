---
layout: blog_detail
title: "New Library Updates in PyTorch 1.13"
author: Team PyTorch
featured-img: "assets/images/new-library-updates-in-pytorch-1.13-2.jpg"
---

## Summary

We are bringing a number of improvements to the current PyTorch libraries, alongside the PyTorch 1.13 [release](https://github.com/pytorch/pytorch/releases). These updates demonstrate our focus on developing common and extensible APIs across all domains to make it easier for our community to build ecosystem projects on PyTorch.

Along with **1.13**, we are releasing updates to the PyTorch Libraries, please find them below.

### TorchAudio 

#### (Beta) Hybrid Demucs Model and Pipeline

Hybrid Demucs is a music source separation model that uses both spectrogram and time domain features. It has demonstrated state-of-the-art performance in the Sony<sup>®</sup> Music DeMixing Challenge. (citation: [https://arxiv.org/abs/2111.03600](https://arxiv.org/abs/2111.03600))

The TorchAudio v0.13 release includes the following features

- MUSDB_HQ Dataset, which is used in Hybrid Demucs training ([docs](https://pytorch.org/audio/0.13.0/generated/torchaudio.datasets.MUSDB_HQ.html#torchaudio.datasets.MUSDB_HQ))
- Hybrid Demucs model architecture ([docs](https://pytorch.org/audio/0.13.0/generated/torchaudio.models.HDemucs.html#torchaudio.models.HDemucs))
- Three factory functions suitable for different sample rate ranges
- Pre-trained pipelines ([docs](https://pytorch.org/audio/0.13.0/pipelines.html#id46))
- SDR Results of pre-trained pipelines on MUSDB_HQ test set
- Tutorial that steps through music source separation using the pretrained pipeline ([docs](https://pytorch.org/audio/0.13.0/tutorials/hybrid_demucs_tutorial.html))

| Pipeline                               |  All  | Drums |  Bass  | Other | Vocals |
|----------------------------------------|-------|-------|--------|-------|--------|
| <em>HDEMUCS_HIGH_MUSDB*</em>           |  6.42 |  7.76 |   6.51 |  4.47 |   6.93 |
| <em>HDEMUCS_HIGH_MUSDB_PLUS**</em>     |  9.37 | 11.38 |  10.53 |  7.24 |   8.32 |

<p><small>* Trained on the training data of MUSDB-HQ dataset.<br/>** Trained on both training and test sets of MUSDB-HQ and 150 extra songs from an internal database that were specifically produced for Meta.</small></p>

```python
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()
sources_list = model.sources

mixture, samplerate = torchaudio.load("song.wav")
sources = model(mixture)
audios = dict(zip(sources_list, sources)
```

Special thanks to Alexandre Defossez for the guidance.

#### (Beta) Datasets and Metadata Mode for SUPERB Benchmark

TorchAudio adds support for various audio-related datasets used in downstream tasks for benchmarking self-supervised learning models. With the addition of several new datasets, there is now support for the downstream tasks in version 1 of the [SUPERB benchmark](https://superbbenchmark.org/), which can be found in the [s3prl repository](https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/docs/superb.md).

For these datasets, we also add metadata support through a `get_metadata` function, enabling faster dataset iteration or preprocessing without the need to load waveforms. The function returns the same features as `__getitem__`, except it returns the relative waveform path rather than the loaded waveform.

Datasets with metadata functionality

- LIBRISPEECH ([docs](https://pytorch.org/audio/0.13.0/generated/torchaudio.datasets.LIBRISPEECH.html#torchaudio.datasets.LIBRISPEECH))
- LibriMix ([docs](https://pytorch.org/audio/0.13.0/generated/torchaudio.datasets.LibriMix.html#torchaudio.datasets.LibriMix))
- QUESST14 ([docs](https://pytorch.org/audio/0.13.0/generated/torchaudio.datasets.QUESST14.html#torchaudio.datasets.QUESST14))
- SPEECHCOMMANDS ([docs](https://pytorch.org/audio/0.13.0/generated/torchaudio.datasets.SPEECHCOMMANDS.html#torchaudio.datasets.SPEECHCOMMANDS))
- (new) FluentSpeechCommands ([docs](https://pytorch.org/audio/0.13.0/generated/torchaudio.datasets.FluentSpeechCommands.html#torchaudio.datasets.FluentSpeechCommands))
- (new) Snips ([docs](https://pytorch.org/audio/0.13.0/generated/torchaudio.datasets.Snips.html#torchaudio.datasets.Snips))
- (new) IEMOCAP ([docs](https://pytorch.org/audio/0.13.0/generated/torchaudio.datasets.IEMOCAP.html#torchaudio.datasets.IEMOCAP))
- (new) VoxCeleb1 ([Identification](https://pytorch.org/audio/0.13.0/generated/torchaudio.datasets.VoxCeleb1Identification.html#torchaudio.datasets.VoxCeleb1Identification), [Verification](https://pytorch.org/audio/0.13.0/generated/torchaudio.datasets.VoxCeleb1Verification.html#torchaudio.datasets.VoxCeleb1Verification))

#### (Beta) Custom Language Model support in CTC Beam Search Decoding

TorchAudio released a CTC beam search decoder in release 0.12, with KenLM language model support. This release, there is added functionality for creating custom Python language models that are compatible with the decoder, using the `torchaudio.models.decoder.CTCDecoderLM` wrapper.

For more information on using a custom language model, please refer to the [documentation](https://pytorch.org/audio/0.13.0/generated/torchaudio.models.decoder.CTCDecoder.html#ctcdecoderlm) and [tutorial](https://pytorch.org/audio/0.13.0/tutorials/asr_inference_with_ctc_decoder_tutorial.html#custom-language-model).

#### (Beta) StreamWriter

torchaudio.io.StreamWriter is a class for encoding media including audio and video. This can handle a wide variety of codecs, chunk-by-chunk encoding and GPU encoding.

```python
writer = StreamWriter("example.mp4")
writer.add_audio_stream(
    sample_rate=16_000,
    num_channels=2,
)
writer.add_video_stream(
    frame_rate=30,
    height=96,
    width=128,
    format="rgb24",
)
with writer.open():
    writer.write_audio_chunk(0, audio)
    writer.write_video_chunk(1, video)
```

For more information, refer to [the documentation](https://pytorch.org/audio/0.13.0/generated/torchaudio.io.StreamWriter.html) and the following tutorials
- [StreamWriter Basic Usage](https://pytorch.org/audio/0.13.0/tutorials/streamwriter_basic_tutorial.html)
- [StreamWriter Advanced Usage](https://pytorch.org/audio/0.13.0/tutorials/streamwriter_advanced.html)
- [Hardware-Accelerated Video Decoding and Encoding](https://pytorch.org/audio/0.13.0/hw_acceleration_tutorial.html)

### TorchData

For a complete list of changes and new features, please visit [our repository’s 0.5.0 release note](https://github.com/pytorch/data/releases).

#### (Prototype) DataLoader2

`DataLoader2` was introduced in the last release to execute `DataPipe` graph, with support for dynamic sharding for multi-process/distributed data loading, multiple backend ReadingServices, and `DataPipe` graph in-place modification (e.g. shuffle control).

In this release, we further consolidated the API for `DataLoader2` and a [detailed documentation is now available here](https://pytorch.org/data/0.5/dataloader2.html). We continue to welcome early adopters and feedback, as well as potential contributors. If you are interested in trying it out, we encourage you to install the nightly version of TorchData.

#### (Beta) Data Loading from Cloud Service Providers

We extended our support to load data from additional cloud storage providers via DataPipes, now covering AWS, Google Cloud Storage, and Azure. A [tutorial is also available](https://pytorch.org/data/0.5/tutorial.html#working-with-cloud-storage-providers). We are open to feedback and feature requests.

We also performed a simple benchmark, comparing the performance of data loading from AWS S3 and attached volume on an AWS EC2 instance.

### torch::deploy (Beta)

torch::deploy is now in Beta! torch::deploy is a C++ library for Linux based operating systems that allows you to run multiple Python interpreters in a single process. You can run your existing eager PyTorch models without any changes for production inference use cases. Highlights include: 

- Existing models work out of the box–no need to modify your python code to support tracing.
- Full support for your existing Python environment including C extensions.
- No need to cross process boundaries to load balance in multi-GPU serving environments.
- Model weight can be shared between multiple Python interpreters.
- A vastly improved installation and setup process.

```Python
torch::deploy::InterpreterManager manager(4);

// access one of the 4 interpreters
auto I = manager.acquireOne();

// run infer from your_model.py
I.global("your_model", "infer")({at::randn({10, 240, 320})});
```

Learn more [here](https://github.com/pytorch/multipy).

#### (Beta) CUDA/ROCm/CPU Backends

torch::deploy now links against standard PyTorch Python distributions so all accelerators that PyTorch core supports such as CUDA and AMD/HIP work out of the box.

- Can install any device variant of PyTorch via pip/conda like normal.
- [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### (Prototype) aarch64/arm64 support

torch::deploy now has basic support for aarch64 Linux systems.

- We're looking to gather feedback on it and learn more about arm use cases for eager PyTorch models.
- Learn more / share your use case at [https://github.com/pytorch/multipy/issues/64](https://github.com/pytorch/multipy/issues/64)

### TorchEval

#### (Prototype) Introducing Native Metrics Support for PyTorch

TorchEval is a library built for users who want highly performant implementations of common metrics to evaluate machine learning models. It also provides an easy to use interface for building custom metrics with the same toolkit. Building your metrics with TorchEval makes running distributed training loops with [torch.distributed](https://pytorch.org/docs/stable/distributed.html) a breeze.

Learn more with our [docs](https://pytorch.org/torcheval), see our [examples](https://pytorch.org/torcheval/stable/metric_example.html), or check out our [GitHub repo](http://github.com/pytorch/torcheval).

### TorchMultimodal Release (Beta)

Please watch for upcoming blogs in early November that will introduce TorchMultimodal, a PyTorch domain library for training SoTA multi-task multimodal models at scale, in more details; in the meantime, play around with the library and models through our [tutorial](https://pytorch.org/tutorials/beginner/flava_finetuning_tutorial.html).

### TorchRec

#### (Prototype) Simplified Optimizer Fusion APIs

We’ve provided a simplified and more intuitive API for setting fused optimizer settings via apply_optimizer_in_backward. This new approach enables the ability to specify optimizer settings on a per-parameter basis and sharded modules will configure [FBGEMM’s TableBatchedEmbedding modules accordingly](https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops.py#L181). Additionally, this now let's TorchRec’s planner account for optimizer memory usage. This should alleviate reports of sharding jobs OOMing after using Adam using a plan generated from planner.

#### (Prototype) Simplified Sharding APIs

We’re introducing the shard API, which now allows you to shard only the embedding modules within a model, and provides an alternative to the current main entry point - DistributedModelParallel. This lets you have a finer grained control over the rest of the model, which can be useful for customized parallelization logic, and inference use cases (which may not require any parallelization on the dense layers). We’re also introducing construct_module_sharding_plan, providing a simpler interface to the TorchRec sharder.

#### (Beta) Quantized Comms

Applying [quantization or mixed precision](https://dlp-kdd.github.io/assets/pdf/a11-yang.pdf) to tensors in a collective call during model parallel training greatly improves training efficiency, with little to no effect on model quality. TorchRec now integrates with the [quantized comms library provided by FBGEMM GPU](https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/quantize_comm.py) and provides an interface to construct encoders and decoders (codecs) that surround the all_to_all, and reduce_scatter collective calls in the output_dist of a sharded module. We also allow you to construct your own codecs to apply to your sharded module. The codces provided by FBGEMM allow FP16, BF16, FP8, and INT8 compressions, and you may use different quantizations for the forward pass and backward pass.

### TorchSnapshot (Beta)

Along with PyTorch 1.13, we are releasing the beta version of TorchSnapshot, which is a performant, memory-efficient checkpointing library for PyTorch applications, designed with large, complex distributed workloads in mind. Highlights include:

- Performance: TorchSnapshot provides a fast checkpointing implementation employing various optimizations, including zero-copy serialization for most tensor types, overlapped device-to-host copy and storage I/O, parallelized storage I/O
- Memory Use: TorchSnapshot's memory usage adapts to the host's available resources, greatly reducing the chance of out-of-memory issues when saving and loading checkpoints
- Usability: Simple APIs that are consistent between distributed and non-distributed workloads

Learn more with our [tutorial](https://pytorch.org/torchsnapshot/main/getting_started.html).

### TorchVision 

We are happy to introduce torchvision v0.14 [(release note)](https://github.com/pytorch/vision/releases). This version introduces a new [model registration API](https://pytorch.org/blog/easily-list-and-initialize-models-with-new-apis-in-torchvision/) to help users retrieving and listing models and weights. It also includes new image and video classification models such as MViT, S3D, Swin Transformer V2, and MaxViT. Last but not least, we also have new primitives and augmentation such as PolynomicalLR scheduler and SimpleCopyPaste.

#### (Beta) Model Registration API

Following up on the [multi-weight support API](https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/) that was released on the previous version, we have added a new [model registration API](https://pytorch.org/blog/easily-list-and-initialize-models-with-new-apis-in-torchvision/) to help users retrieve models and weights. There are now 4 new methods under the torchvision.models module: get_model, get_model_weights, get_weight, and list_models. Here are examples of how we can use them:

```Python
import torchvision
from torchvision.models import get_model, get_model_weights, list_models


max_params = 5000000

tiny_models = []
for model_name in list_models(module=torchvision.models):
    weights_enum = get_model_weights(model_name)
    if len([w for w in weights_enum if w.meta["num_params"] <= max_params]) > 0:
        tiny_models.append(model_name)

print(tiny_models)
# ['mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mobilenet_v2', ...]

model = get_model(tiny_models[0], weights="DEFAULT")
print(sum(x.numel() for x in model.state_dict().values()))
# 2239188
```

#### (Beta) New Video Classification Models

We added two new video classification models, MViT and S3D. MViT is a state of the art video classification transformer model which has 80.757% accuracy on the Kinetics400 dataset, while S3D is a relatively small model with good accuracy for its size. These models can be used as follows:

```Python
import torch
from torchvision.models.video import *

video = torch.rand(3, 32, 800, 600)
model = mvit_v2_s(weights="DEFAULT")
# model = s3d(weights="DEFAULT")
model.eval()
prediction = model(images)
```

Here is the table showing the accuracy of the new video classification models tested in the Kinetics400 dataset.

| **Model**                      | **Acc@1** | **Acc@5** |
|--------------------------------|-----------|-----------|
| mvit_v1_b                      |    81.474 |    95.776 |
| mvit_v2_s                      |    83.196 |     96.36 |
| s3d                            |    83.582 |     96.64 |

We would like to thank Haoqi Fan, Yanghao Li, Christoph Feichtenhofer and Wan-Yen Lo for their work on [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo/) and their support during the development of the MViT model. We would like to thank Sophia Zhi for her contribution implementing the S3D model in torchvision.

#### (Stable) New Architecture and Model Variants

For Classification Models, we’ve added the Swin Transformer V2 architecture along with pre-trained weights for its tiny/small/base variants. In addition, we have added support for the MaxViT transformer. Here is an example on how to use the models:

```Python
import torch
from torchvision.models import *

image = torch.rand(1, 3, 224, 224)
model = swin_v2_t(weights="DEFAULT").eval()
# model = maxvit_t(weights="DEFAULT").eval()
prediction = model(image)
```

Here is the table showing the accuracy of the models tested on ImageNet1K dataset.

| **Model**     | **Acc@1** | **Acc@1 change over V1** | **Acc@5** | **Acc@5 change over V1** |
|---------------|-----------|--------------------------|-----------|--------------------------|
| swin_v2_t     |    82.072 |                  + 0.598 |    96.132 |                  + 0.356 |
| swin_v2_s     |    83.712 |                  + 0.516 |    96.816 |                  + 0.456 |
| swin_v2_b     |    84.112 |                  + 0.530 |    96.864 |                  + 0.224 |
| maxvit_t      |    83.700 |                    -     |    96.722 |                    -     |

We would like to thank [Ren Pang](https://github.com/ain-soph) and [Teodor Poncu](https://github.com/TeodorPoncu) for contributing the 2 models to torchvision.

### (Stable) New Primitives & Augmentations

In this release we’ve added the [SimpleCopyPaste](https://arxiv.org/abs/2012.07177) augmentation in our reference scripts and we up-streamed the PolynomialLR scheduler to PyTorch Core. We would like to thank [Lezwon Castelino](https://github.com/lezwon) and [Federico Pozzi](https://github.com/federicopozzi33) for their contributions. We are continuing our efforts to modernize TorchVision by adding more SoTA primitives, Augmentations and architectures with the help of our community. If you are interested in contributing, have a look at the following [issue](https://github.com/pytorch/vision/issues/6323).

### Torch-TensorRT

#### (Prototype) TensorRT with FX2TRT frontend

Torch-TensorRT is the PyTorch integration for TensorRT, providing high performance inference on NVIDIA GPUs. Torch-TRT allows for optimizing models directly in PyTorch for deployment providing up to 6x performance improvement. 

Torch-TRT is an AoT compiler which ingests an nn.Module or TorchScript module, optimizes compatible subgraphs in TensorRT & leaves the rest to run in PyTorch. This gives users the performance of TensorRT, but the usability and familiarity of Torch.

Torch-TensorRT is part of the PyTorch ecosystem, and was released as v1.0 in November ‘21. There are currently two distinct front-ends: Torchscript & FX. Each provides the same value proposition and underlying operation with the primary difference being the input & output formats (TS vs FX / Python).

The Torchscript front-end was included in v1.0 and should be considered stable. The FX front-end is first released in v1.2 and should be considered a Beta.

Relevant Links:

- [Github](https://github.com/pytorch/TensorRT)
- [Documentation](https://pytorch.org/TensorRT/)
- [Generic (TS) getting started guide](https://pytorch.org/TensorRT/getting_started/getting_started_with_python_api.html)
- [FX getting started guide](https://pytorch.org/TensorRT/tutorials/getting_started_with_fx_path.html)

#### (Stable)  Introducing Torch-TensorRT

Torch-TensorRT is an integration for PyTorch that leverages inference optimizations of TensorRT on NVIDIA GPUs. It takes advantage of TensorRT optimizations, such as FP16 and INT8 reduced precision, graph optimization, operation fusion, etc. while offering a fallback to native PyTorch when TensorRT does not support the model subgraphs. Currently, there are two frontend paths existing in the library that help to convert a PyTorch model to tensorRT engine. One path is through Torch Script (TS) and the other is through FX frontend. That being said, the models are traced by either TS or FX into their IR graph and then converted to TensorRT from it.

Learn more with our [tutorial](https://pytorch.org/TensorRT/).

### TorchX

TorchX 0.3 updates include a new list API, experiment tracking, elastic training and improved scheduler support. There’s also a new Multi-Objective NAS [tutorial](https://pytorch.org/tutorials/intermediate/ax_multiobjective_nas_tutorial.html) using TorchX + Ax.

#### (Prototype) List

The newly added list command and API allows you to list recently launched jobs and their statuses for a given scheduler directly from within TorchX.

- This removes the need for using secondary tools to list the jobs.
- Full programmatic access to recent jobs for integration with custom tools.

```Python
$ torchx list -s kubernetes
APP HANDLE                                                       APP STATUS
-----------------------------------------------            -----------------
kubernetes://torchx/default:train-f2nx4459p5crr   SUCCEEDED
```

Learn more with our [documentation](https://pytorch.org/torchx/main/schedulers.html#torchx.schedulers.Scheduler.list).

#### (Prototype) Tracker

TorchX Tracker is a new prototype library that provides a flexible and customizable experiment and artifact tracking interface. This allows you to track inputs and outputs for jobs across multiple steps to make it easier to use TorchX with pipelines and other external systems.

```Python
from torchx import tracker

app_run = tracker.app_run_from_env()
app_run.add_metadata(lr=lr, gamma=gamma) # hyper parameters
app_run.add_artifact("model", "storage://path/mnist_cnn.pt") # logs / checkpoints
app_run.add_source(parent_run_id, "model") # lineage
```

Example:

- [https://github.com/pytorch/torchx/tree/main/torchx/examples/apps/tracker](https://github.com/pytorch/torchx/tree/main/torchx/examples/apps/tracker)
- [https://pytorch.org/torchx/main/tracker.html](https://pytorch.org/torchx/main/tracker.html)

#### (Prototype) Elastic Training and Autoscaling

Elasticity on Ray and Kubernetes – automatic scale up of distributed training jobs when using a supported scheduler. Learn more with our [documentation](https://pytorch.org/torchx/main/components/distributed.html).

#### (Prototype) Scheduler Improvements: IBM® Spectrum LSF

Added prototype support for the IBM Spectrum LSF scheduler.

#### (Beta) AWS Batch Scheduler

The AWS Batch scheduler integration is now in beta.

- log fetching and listing jobs is now supported.
- Added configs for job priorities and queue policies
- Easily access job UI via ui_url
[https://pytorch.org/torchx/main/schedulers/aws_batch.html](https://pytorch.org/torchx/main/schedulers/aws_batch.html)

#### (Prototype) AnyPrecision Optimizer 

Drop in replacement for AdamW optimizer that reduces GPU memory, enables two main features:

- Ability to successfully train the entire model pipeline in full BFloat16.
Kahan summation ensures precision.  This can improve training throughput, especially on huge models, by reduced memory and increased computation speed.
- Ability to change the variance state to BFloat16.  This can reduce overall memory required for model training with additional speed improvements.

Find more information [here](https://github.com/pytorch/torchdistx/pull/52).
