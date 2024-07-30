---
layout: blog_detail
title: "Introducing torchchat: Accelerating Local LLM Inference on Laptop, Desktop and Mobile"
author: Ali Khosh, Jesse White, Orion Reblitz-Richardson
---

Today, we’re releasing [torchchat](https://github.com/pytorch/torchchat), a library showcasing how to seamlessly and performantly run Llama 3, 3.1, and other large language models across laptop, desktop, and mobile.

In our previous blog posts, we [showed](https://pytorch.org/blog/accelerating-generative-ai-2/) how to use native PyTorch 2.0 to run LLMs with great performance using CUDA. Torchchat expands on this with more target environments, models and execution modes as well as providing important functions such as export, quantization and export in a way that’s easy to understand.

You will find the project organized into three areas:

* Python: Torchchat provides a [REST API](https://github.com/pytorch/torchchat?tab=readme-ov-file#server) that is called via a Python CLI or can be accessed via the browser
* C++: Torchchat produces a desktop-friendly binary using PyTorch's [AOTInductor](https://pytorch-dev-podcast.simplecast.com/episodes/aotinductor) backend
* Mobile devices: Torchchat uses [ExecuTorch](https://pytorch.org/executorch/stable/index.html) to export a .pte binary file for on-device inference


![torchchat schema](/assets/images/torchchat.png){:style="width:100%"}


## Performance

The following table tracks the performance of torchchat for Llama 3 for a variety of configurations.

_Numbers for Llama 3.1 are coming soon._

**Llama 3 8B Instruct on Apple MacBook Pro M1 Max 64GB**


<table class="table table-bordered">
  <tr>
   <td><strong>Mode</strong>
   </td>
   <td><strong>DType</strong>
   </td>
   <td><strong>Llama 3 8B Tokens/Sec</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="3" >Arm Compile
   </td>
   <td>float16
   </td>
   <td>5.84
   </td>
  </tr>
  <tr>
   <td>int8
   </td>
   <td>1.63
   </td>
  </tr>
  <tr>
   <td>int4
   </td>
   <td>3.99
   </td>
  </tr>
  <tr>
   <td rowspan="3" >Arm AOTI
   </td>
   <td>float16
   </td>
   <td>4.05
   </td>
  </tr>
  <tr>
   <td>int8
   </td>
   <td>1.05
   </td>
  </tr>
  <tr>
   <td>int4
   </td>
   <td>3.28
   </td>
  </tr>
  <tr>
   <td rowspan="3" >MPS Eager
   </td>
   <td>float16
   </td>
   <td>12.63
   </td>
  </tr>
  <tr>
   <td>int8
   </td>
   <td>16.9
   </td>
  </tr>
  <tr>
   <td>int4
   </td>
   <td>17.15
   </td>
  </tr>
</table>


**Llama 3 8B Instruct on Linux x86 and CUDA**

_Intel(R) Xeon(R) Platinum 8339HC CPU @ 1.80GHz with 180GB Ram + A100 (80GB)_


<table class="table table-bordered">
  <tr>
   <td>
<strong>Mode</strong>
   </td>
   <td><strong>DType</strong>
   </td>
   <td><strong>Llama 3 8B Tokens/Sec</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="3" >x86 Compile
   </td>
   <td>bfloat16
   </td>
   <td>2.76
   </td>
  </tr>
  <tr>
   <td>int8
   </td>
   <td>3.15
   </td>
  </tr>
  <tr>
   <td>int4
   </td>
   <td>5.33
   </td>
  </tr>
  <tr>
   <td rowspan="3" >CUDA Compile
   </td>
   <td>bfloat16
   </td>
   <td>83.23
   </td>
  </tr>
  <tr>
   <td>int8
   </td>
   <td>118.17
   </td>
  </tr>
  <tr>
   <td>int4
   </td>
   <td>135.16
   </td>
  </tr>
</table>


Torchchat provides exceptional performance for Llama 3 8B on mobile (iPhone and Android). We run Llama 2 7B on Samsung Galaxy S22, and S23, and on iPhone 15 Pro using 4-bit GPTQ and post training quantization (PTQ). Early work on Llama 3 8B support is included in collaboration with ExecuTorch. Many improvements were made to export speed, memory overhead, and runtime speed. Ultimately, though, we’ll be seeing even stronger performance through Core ML, MPS, and HTP in the near future. We are excited!

We encourage you to **[clone the torchchat repo and give it a spin](https://github.com/pytorch/torchchat)**, explore its capabilities, and share your feedback as we continue to empower the PyTorch community to run LLMs locally and on constrained devices. Together, let's unlock the full potential of generative AI and LLMs on any device. Please submit [issues](https://github.com/pytorch/torchat/issues) as you see them as well as in [PyTorch](https://github.com/pytorch/pytorch/issues) plus [ExecuTorch](https://github.com/pytorch/executorch/issues), since we are still iterating quickly. We’re also inviting community contributions across a broad range of areas, from additional models, target hardware support, new quantization schemes, or performance improvements.  Happy experimenting!
