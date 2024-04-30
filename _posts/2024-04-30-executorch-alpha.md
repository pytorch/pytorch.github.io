---
layout: blog_detail
title: "ExecuTorch Alpha: Taking LLMs and AI to the Edge with Our Community and Partners"
---

We are excited to announce the release of [ExecuTorch alpha](https://github.com/pytorch/executorch), focused on deploying large language models (LLMs) and large ML models to the edge, stabilizing the API surface, and improving our installation processes. It has been an exciting few months [from our 0.1 (preview) release](https://pytorch.org/blog/pytorch-edge/) in collaboration with our partners at Arm, Apple, and Qualcomm Technologies, Inc.

In this post we’ll discuss our full support for Meta’s Llama 2, early support for Meta’s Llama 3, broad model support in ExecuTorch, and highlight the important work our partners have done to move us forward.

## Large Language Models on Mobile

Mobile devices are highly constrained for compute, memory, and power. To bring LLMs to these devices, we heavily leverage quantization and other techniques to pack these models appropriately. 

ExecuTorch alpha supports 4-bit post-training quantization using GPTQ. We've provided broad device support on CPU by landing dynamic shape support and new dtypes in XNNPack. We've also made significant improvements in export and lowering, reduced memory overhead and improved runtime performance. This enables running Llama 2 7B efficiently on iPhone 15 Pro, iPhone 15 Pro Max, Samsung Galaxy S22, S23, and S24 phones and other edge devices. [Early support](https://github.com/pytorch/executorch/releases/tag/v0.2.0) for [Llama 3 8B](https://ai.meta.com/blog/meta-llama-3/) is also included. We are always improving the token/sec on various edge devices and you can visit GitHub for the [latest performance numbers](https://github.com/pytorch/executorch/blob/main/examples/models/llama2/README.md). 

We're working closely with our partners at Apple, Arm, and Qualcomm Technologies to delegate to GPU and NPU for performance through Core ML, MPS, TOSA, and Qualcomm AI Stack backends respectively.

## Supported Models

We remain committed to supporting an ever-expanding list of models with ExecuTorch. Since preview, we have significantly expanded our tested models across NLP, vision and speech, with full details [in our release notes](https://github.com/pytorch/executorch/releases/tag/v0.2.0). Although support for on-device LLMs is early, we anticipate most traditional models to function seamlessly out of the box, with delegation to XNNPACK, Core ML, MPS, TOSA, and HTP for performance. If you encounter any problems please open [a GitHub issue](https://github.com/pytorch/executorch/issues) with us.

## Productivity

Deploying performant models tuned for specific platforms often require deep visualization into the on-device runtime data to determine the right changes to make in the original PyTorch model. With ExecuTorch alpha, we provide a powerful SDK with observability throughout the process from model authoring to deployment, including delegate and hardware-level information.

The ExecuTorch SDK was enhanced to include better debugging and profiling tools. Because ExecuTorch is built on PyTorch, the debugging capabilities include the ability to map from operator nodes back to original Python source code for more efficient anomaly resolution and performance tuning for both delegated and non-delegated model instances. You can learn more about the ExecuTorch SDK [here](https://github.com/pytorch/executorch/blob/main/examples/sdk/README.md). 

## Partnerships

ExecuTorch has only been possible because of strong collaborations across Arm, Apple, and  Qualcomm Technologies. The collaboration for the initial launch of ExecuTorch continues as we support LLMs and large AI models on the edge for PyTorch. As we’ve seen with this early work for ExecuTorch alpha, there are unique challenges with these larger models and we’re excited to develop in the open.

We also want to highlight the great partnership with Google on [XNNPACK](https://github.com/google/XNNPACK) for CPU performance. The teams continue to work together upstreaming our changes and across the TensorFlow and PyTorch teams to make sure we can all support generative AI models on the edge with SOTA performance.

Lastly, our hardware partner MediaTek has been doing work enabling the Llama collection of models with ExecuTorch on their SoCs. We'll have more to share in the future.

## Alpha and Production Usage

With our alpha release, we have production-tested ExecuTorch. Meta is using ExecuTorch for hand tracking on Meta Quest 3 and a variety of models on Ray-Ban Meta Smart Glasses. In addition, we have begun the rollout of ExecuTorch with Instagram and are integrating with other Meta products. We are excited to see how ExecuTorch can be used for other edge experiences.

## Community

We are excited to see various efforts in the community to adopt or contribute to ExecuTorch. For instance, Unity recently [shared their work](https://schedule.gdconf.com/session/unity-developer-summit-drive-better-gameplay-experiences-on-user-devices-with-ai-presented-by-unity/903634) at the Game Developers Conference ([GDC](https://gdconf.com/)) on leveraging ExecuTorch and Edge IR to run PyTorch models with their neural network inference library Sentis. Leveraging ExecuTorch's hackability and extensibility, Unity introduced their own custom backend that serializes ExecuTorch’s Edge Dialect IR into Sentis’ native serialized format enabling developers to begin using PyTorch models easily in their games and apps. 

We’ve been building and innovating with ExecuTorch in the open. Our north star is to empower the community to deploy any ML model on edge devices painlessly and efficiently. Whether you are a hobbyist or this is your day job, we’d love for you to [jump in to bring your ML models to the edge](https://pytorch.org/executorch/stable/getting-started-setup.html). We are looking for your help to:

1. Use ExecuTorch to [run your LLM models locally](https://github.com/pytorch/executorch/blob/main/docs/source/llm/getting-started.md) on various deployment targets and share your feedback
2. Expand our supported models, including bug reports
3. Expand our quantization schemes
4. Help us build out delegates to GPU and NPU

To all individual contributors and early adopters of ExecuTorch, a big thank you as well. We can’t wait to have more of you [join us](https://github.com/pytorch/executorch)!