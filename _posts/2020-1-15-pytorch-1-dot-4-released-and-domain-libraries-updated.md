---
layout: blog_detail
title: 'PyTorch 1.4 released, domain libraries updated'
author: Team PyTorch
---

Today, we’re announcing the availability of PyTorch 1.4, along with updates to the PyTorch domain libraries. These releases build on top of the announcements from [NeurIPS 2019](https://pytorch.org/blog/pytorch-adds-new-tools-and-libraries-welcomes-preferred-networks-to-its-community/), where we shared the availability of PyTorch Elastic, a new classification framework for image and video, and the addition of Preferred Networks to the PyTorch community. For those that attended the workshops at NeurIPS, the content can be found [here](https://research.fb.com/neurips-2019-expo-workshops/).

## PyTorch 1.4

The 1.4 release of PyTorch adds new capabilities, including the ability to do fine grain build level customization for PyTorch Mobile, and new experimental features including support for model parallel training and Java language bindings.

### PyTorch Mobile - Build level customization

Following the open sourcing of [PyTorch Mobile in the 1.3 release](https://pytorch.org/blog/pytorch-1-dot-3-adds-mobile-privacy-quantization-and-named-tensors/), PyTorch 1.4 adds additional mobile support including the ability to customize build scripts at a fine-grain level. This allows mobile developers to optimize library size by only including the operators used by their models and, in the process, reduce their on device footprint significantly. Initial results show that, for example, a customized MobileNetV2 is 40% to 50% smaller than the prebuilt PyTorch mobile library. You can learn more [here](https://pytorch.org/mobile/home/) about how to create your own custom builds and, as always, please engage with the community on the [PyTorch forums](https://discuss.pytorch.org/c/mobile) to provide any feedback you have.

Example code snippet for selectively compiling only the operators needed for MobileNetV2:

```python
# Dump list of operators used by MobileNetV2:
import torch, yaml
model = torch.jit.load('MobileNetV2.pt')
ops = torch.jit.export_opnames(model)
with open('MobileNetV2.yaml', 'w') as output:
    yaml.dump(ops, output)
```

```console
# Build PyTorch Android library customized for MobileNetV2:
SELECTED_OP_LIST=MobileNetV2.yaml scripts/build_pytorch_android.sh arm64-v8a

# Build PyTorch iOS library customized for MobileNetV2:
SELECTED_OP_LIST=MobileNetV2.yaml BUILD_PYTORCH_MOBILE=1 IOS_ARCH=arm64 scripts/build_ios.sh
```

### Distributed model parallel training (Experimental)

With the scale of models, such as RoBERTa, continuing to increase into the billions of parameters, model parallel training has become ever more important to help researchers push the limits. This release provides a distributed RPC framework to support distributed model parallel training. It allows for running functions remotely and referencing remote objects without copying the real data around, and provides autograd and optimizer APIs to transparently run backwards and update parameters across RPC boundaries.

To learn more about the APIs and the design of this feature, see the links below:

* [API documentation](https://pytorch.org/docs/stable/rpc.html)

For the full tutorials, see the links below:  

* [A full RPC tutorial](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)
* [Examples using model parallel training for reinforcement learning and with an LSTM](https://github.com/pytorch/examples/tree/master/distributed/rpc)

As always, you can connect with community members and discuss more on the [forums](https://discuss.pytorch.org/c/distributed/distributed-rpc).

### Java bindings (Experimental)

In addition to supporting Python and C++, this release adds experimental support for Java bindings. Based on the interface developed for Android in PyTorch Mobile, the new bindings allow you to invoke TorchScript models from any Java program. Note that the Java bindings are only available for Linux for this release, and for inference only. We expect support to expand in subsequent releases. See the code snippet below for how to use PyTorch within Java:

```java
Module mod = Module.load("demo-model.pt1");
Tensor data =
    Tensor.fromBlob(
        new int[] {1, 2, 3, 4, 5, 6}, // data
        new long[] {2, 3} // shape
        );
IValue result = mod.forward(IValue.from(data), IValue.from(3.0));
Tensor output = result.toTensor();
System.out.println("shape: " + Arrays.toString(output.shape()));
System.out.println("data: " + Arrays.toString(output.getDataAsFloatArray()));
```

Learn more about how to use PyTorch from Java [here](https://github.com/pytorch/java-demo), and see the full Javadocs API documentation [here](https://pytorch.org/javadoc/1.4.0/).

For the full 1.4 release notes, see [here](https://github.com/pytorch/pytorch/releases).

## Domain Libraries

PyTorch domain libraries like torchvision, torchtext, and torchaudio complement PyTorch with common datasets, models, and transforms. We’re excited to share new releases for all three domain libraries alongside the PyTorch 1.4 core release.

### torchvision 0.5

The improvements to torchvision 0.5 mainly focus on adding support for production deployment including quantization, TorchScript, and ONNX. Some of the highlights include:

* All models in torchvision are now torchscriptable making them easier to ship into non-Python production environments
* ResNets, MobileNet, ShuffleNet, GoogleNet and InceptionV3 now have quantized counterparts with pre-trained models, and also include scripts for quantization-aware training.
* In partnership with the Microsoft team, we’ve added ONNX support for all models including Mask R-CNN.

Learn more about torchvision 0.5 [here](https://github.com/pytorch/vision/releases).

### torchaudio 0.4

Improvements in torchaudio 0.4 focus on enhancing the currently available transformations, datasets, and backend support. Highlights include:

* SoX is now optional, and a new extensible backend dispatch mechanism exposes SoundFile as an alternative to SoX.
* The interface for datasets has been unified. This enables the addition of two large datasets: LibriSpeech and Common Voice.
* New filters such as biquad, data augmentation such as time and frequency masking, transforms such as MFCC, gain and dither, and new feature computation such as deltas, are now available.
* Transformations now support batches and are jitable.
* An interactive speech recognition demo with voice activity detection is available for experimentation.

Learn more about torchaudio 0.4 [here](https://github.com/pytorch/audio/releases).

### torchtext 0.5

torchtext 0.5 focuses mainly on improvements to the dataset loader APIs, including compatibility with core PyTorch APIs, but also adds support for unsupervised text tokenization. Highlights include:

* Added bindings for SentencePiece for unsupervised text tokenization .
* Added a new unsupervised learning dataset - enwik9.
* Made revisions to PennTreebank, WikiText103, WikiText2, IMDb to make them compatible with torch.utils.data. Those datasets are in an experimental folder and we welcome your feedback.

Learn more about torchtext 0.5 [here](https://github.com/pytorch/text/releases).

*We’d like to thank the entire PyTorch team and the community for all their contributions to this work.*

Cheers!

Team PyTorch
