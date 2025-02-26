---
layout: blog_detail
title: "Accelerating Generative AI with PyTorch: Segment Anything 2 - Fast and furious inference with low latency and fast cold starts"
---

This post is a follow-up to our [first entry in the multi-series blog focused on how to accelerate generative AI models](https://pytorch.org/blog/accelerating-generative-ai/) with pure, native PyTorch and a focus on latency and elastic scalability. We use torch.compile and torch.export to create highly optimized low latency versions of SAM2 that can be quickly scaled up on new instances.

By utilizing AOTInductor's (AOTI) ahead-of-time compilation via torch.export, reduced precision, batched prompts and GPU preprocessing we observe up to **13x improvement in p90 execution latency** and **queue times compared to regular eager mode PyTorch**.

We calculate our final results and demonstrate the improvement in a realistic deployment on auto-scaling cloud infrastructure from [Modal](https://modal.com).


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td colspan="2" >p50 execution latency
<br/>
(ms / improvement)
   </td>
   <td colspan="2" >p90 execution latency
<br/>
(ms / improvement)
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>eager float32
   </td>
   <td>AOTI float16
   </td>
   <td>eager float32
   </td>
   <td>AOTI float16
   </td>
  </tr>
  <tr>
   <td>AMG
   </td>
   <td>741
   </td>
   <td>112 (6.6x)
   </td>
   <td>1140
   </td>
   <td>176 (6.5x)
   </td>
  </tr>
  <tr>
   <td>SPS
   </td>
   <td>98
   </td>
   <td>20 (4.9x)
   </td>
   <td>130
   </td>
   <td>28 (4.6x)
   </td>
  </tr>
  <tr>
   <td>MPS
   </td>
   <td>269
   </td>
   <td>38 (7.1x)
   </td>
   <td>714
   </td>
   <td>52 (13.7x)
   </td>
  </tr>
</table>



<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td colspan="2" >p50 queue time (ms / improvement)
   </td>
   <td colspan="2" >p90 queue time (ms / improvement)
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>eager float32
   </td>
   <td>AOTI float16
   </td>
   <td>eager float32
   </td>
   <td>AOTI float16
   </td>
  </tr>
  <tr>
   <td>AMG
   </td>
   <td>201
   </td>
   <td>41 (4.9x)
   </td>
   <td>815
   </td>
   <td>327 (2.6x)
   </td>
  </tr>
  <tr>
   <td>SPS
   </td>
   <td>31
   </td>
   <td>33 (0.9x)
   </td>
   <td>441
   </td>
   <td>49 (9.0x)
   </td>
  </tr>
  <tr>
   <td>MPS
   </td>
   <td>40
   </td>
   <td>37 (1.1x)
   </td>
   <td>942
   </td>
   <td>75 (12.6x)
   </td>
  </tr>
</table>



## The Tasks

The first post focused on processing a small number of varying prompts (points of interest) per image. These points represented the center points of the ground truth masks. For this post, we'll now focus on a broader set of tasks. Single prompt segmentation (SPS), multi prompt segmentation (MPS), automatic mask generation (AMG) which generates the full set of masks for the input image without a given set of prompts. The first post focused on MPS only.

![comparison of 3 images](/assets/images/accelerating-generative-ai-2.jpg){:style="width:100%"}



The little star in the image represents a user prompt. For AMG there are no prompts and masks are filtered down heuristically from a dense grid of initial candidate prompts (guesses). For SPS and MPS user prompts are derived from the center points of AMG masks. For SPS we choose the mask with the largest area.

**Note that SAM2 uses a different backbone than SAM1. In particular, we only consider the largest and most accurate sam2.1_hiera_large backbone for this blog.**

We aggregate the scripts needed to reproduce the results in [torchao's example folder](https://github.com/pytorch/ao/tree/main/examples/sam2_amg_server) and incrementally upstream the more stable parts of the [changes to the SAM2 model in torchao](https://github.com/pytorch/ao/tree/main/torchao/_models/sam2) to the main [SAM2](https://github.com/facebookresearch/sam2) repository. So if you are interested in taking a look at the cutting-edge variant or would like to contribute experimental features, please don't hesitate to reach out to the torchao repository and team. For the more stable and latest model version, please head on over to SAM2 directly.


## Overview

We categorize the changes presented here into two. **Fast** changes constrain themselves to techniques that are not meant to affect model accuracy. **Furious** changes sacrifice some numerical accuracy for additional speed by making use of approximations such as low-precision data types. 

Approximations may slightly lower precision metrics in favor of significantly improved performance while still passing an end-to-end check based on mean intersection over union (mIoU).

To measure the performance improvements we processed 1000 images, which were selected at random from the SAM2 validation dataset. We look at the p50 and p90 latency per image. To measure accuracy we consider the mIoU. Most notably for the AMG task we also define a fail count metric. We consider a comparison failed if the **number of masks** differs. This turns out to be a fairly unstable quantity and we can see that the other tasks are not as sensitive to small numeric changes as AMG.


## The Setup

We are running the offline experiments on a regular H100 devserver, which is a fairly beefy and performant machine.

However, we try to look at these tasks with realistic constraints. In particular, we would like to emulate a server-side inference environment. That means we don't use DataLoader to hide the latency of image preprocessing or decoding routines.

For the latency calculations we include decoding, segmentation and conversion of masks to a dictionary of run-length encoded masks. Or put differently, we exclude loading the images into in-memory host bytearrays and storing the resulting dictionaries as json files on disk. This is meant to emulate a more realistic setting.

More concretely, consider the code below for the routines we include in our measurements. For any task `gen_masks` produces a batched bool Tensor bitmask that represents the corresponding object masks. We then compress this bitmask into a run length encoded (rle) format that can be used to transfer back the results from a remote server much more efficiently.


```
image_tensors = decode_img_bytes(...)
masks = gen_masks(image_tensors, ...)
rle_dicts = [rle_dict_from_masks(m) for m in masks]
```



## Optimizations


### ao: eager code optimizations

The most effective tool for this work is the PyTorch autograd profiler combined with `record_function`. To build this software, we've used the profiler repeatedly to observe the program and confirm the effectiveness of any changes. It's also important to keep in mind that the profiler itself has overhead. The more data you collect, such as stack traces, the more overhead you introduce, which might skew the collected trace. But it is excellent to find synchronization points, space between kernels and GPU kernels that take a long time.

GPU traces help you understand bottlenecks that are not necessarily easily addressed by compile. We found that AutomaticMaskGeneration in particular is dominated by the data structure used to store the masks and by the routine used to convert the masks to a run-length encoded compressed format. We also found a large part of AMG performance is dominated by the large number of masks created as a single batch. Sometimes candidate masks can be filtered down to fewer candidates earlier in the postprocessing stage by reordering operations. This in turn significantly speeds up the later operations.

In order to confirm the accuracy of our implementation we first compare without any changes in settings and using float32 precision. We see that mIoU is unchanged and the masks match perfectly when using the exact same settings. This means that these eager mode changes did not affect the accuracy of these tasks.

AMG


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU / fail count
   </td>
  </tr>
  <tr>
   <td>Baseline
   </td>
   <td>864
   </td>
   <td>1144
   </td>
   <td>4350
   </td>
   <td>reference
   </td>
  </tr>
  <tr>
   <td>AO
   </td>
   <td>693
   </td>
   <td>786
   </td>
   <td>4010
   </td>
   <td>1 / 0
   </td>
  </tr>
</table>



### ao: batching prompts

Another lossless performance optimization that we were able to apply is batching the user input prompt calculations. When optimizing for latency at batch size 1 on a server-grade GPU such as an H100 we are often left with a lot of spare memory. We can easily trade off that memory for more performance by processing more points of interest (also called user prompts) at once. Remember that SAM2 is split into two parts: First the backbone (image encoder), second the prediction and decoding of masks based on a set of user prompts / points of interest. It is the second part where we may expect a larger or even varying number of inputs and it is this second part where we apply batching.

This causes a large increase in memory, but also much better latency. The baseline generates one mask per prompt in a loop. For AMG the baseline processes 64 prompts at once and all that is needed is to change it to 1024, which is the number of candidate prompts generated. For SPS we process one prompt at a time, but it's still included below for completeness.

AMG


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU / fail count
   </td>
  </tr>
  <tr>
   <td>Baseline
   </td>
   <td>864
   </td>
   <td>1144
   </td>
   <td>4350
   </td>
   <td>reference
   </td>
  </tr>
  <tr>
   <td>AO + batching
   </td>
   <td>613
   </td>
   <td>706
   </td>
   <td>33786
   </td>
   <td>0.9999995 / 0
   </td>
  </tr>
</table>


SPS


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU
   </td>
  </tr>
  <tr>
   <td>Baseline
   </td>
   <td>116
   </td>
   <td>181
   </td>
   <td>1337
   </td>
   <td>reference
   </td>
  </tr>
  <tr>
   <td>AO
   </td>
   <td>110
   </td>
   <td>170
   </td>
   <td>1339
   </td>
   <td>1
   </td>
  </tr>
</table>


MPS


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU
   </td>
  </tr>
  <tr>
   <td>Baseline
   </td>
   <td>276
   </td>
   <td>681
   </td>
   <td>1337
   </td>
   <td>reference
   </td>
  </tr>
  <tr>
   <td>AO + batching
   </td>
   <td>126
   </td>
   <td>225
   </td>
   <td>8021
   </td>
   <td>0.9999992
   </td>
  </tr>
</table>


As a technical side note: Most notably to enable batching for MPS, and to avoid a significant manual rewrite of the code base to support multiple prompts at the same time, we used a Tensor subclass we call MapTensor. A MapTensor allows us to pass a batch of N prompts, but have it advertise a batch size of 1. Any operation is then automatically broadcast to the wrapped Tensor and propagated throughout the prediction part of the model. This works because individual prompt predictions are independent of one another. This is very similar to torch.vmap.


```
center_points_torch = to_map_tensor(center_points_torch)
center_points_label_torch = to_map_tensor(center_points_label_torch)
masks, scores, _ = mask_generator.predictor.predict(
    point_coords=center_points_torch,
    point_labels=center_points_label_torch,
    multimask_output=True,
    return_logits=False,
    return_type="torch",
)
# Unwrapping MapTensor
masks = masks.elems
scores = scores.elems
```



### fast: fullgraph compilation

Just as with our first post, we first remove GPU syncs and graph breaks to make use of fullgraph compiled model code with max-autotune kernels where appropriate. After some rewriting, we are able to compile the image encoder and the prediction of masks.

We run the experiments twice to get a sense of the overhead due to compilation. We run it once in an environment with an empty TORCHINDUCTOR_CACHE_DIR and then again while ingesting the artifacts from the previous run. In particular, auto-tuning can take a long time and happens on the first call in a pristine environment. We call the second run "warm". The first iteration is typically expected to be slow due to various other related initialization processes, but compile increases it significantly, even if an existing cache is used and the same exact shapes are fed again. Having said that, an overhead of a few seconds in a warm environment is often still stomachable on the very first call.

Most of these drawbacks can be mitigated and compiling causes a significant improvement in latency and reduction in memory.

AMG


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU / 
<br/>
fail count
   </td>
   <td>first iteration
<br/>
(ms)
   </td>
  </tr>
  <tr>
   <td>AO + batching
   </td>
   <td>613
   </td>
   <td>706
   </td>
   <td>33786
   </td>
   <td>0.9999995 / 0
   </td>
   <td>1125
   </td>
  </tr>
  <tr>
   <td>+ compile (cold)
   </td>
   <td>423
   </td>
   <td>513
   </td>
   <td>29349
   </td>
   <td>skipped
   </td>
   <td>404866
   </td>
  </tr>
  <tr>
   <td>+ compile (warm)
   </td>
   <td>439
   </td>
   <td>530
   </td>
   <td>29349
   </td>
   <td>0.994 / 190
   </td>
   <td>8544
   </td>
  </tr>
</table>


The number of masks produced per mask can vary slightly when using automatic mask segmentation. There is ambiguity in the number of masks per object the model may produce. For example, a car may be subdivided into frames, windows and doors or treated as a whole. When a modification causes the number of masks to change, we consider the comparison failed and we only calculate the mIoU on masks with an exact match. This does not apply to the other tasks. We found that the number of masks generated is very sensitive to small numerical changes. The other tasks use the same code and MPS in particular can help us further verify correctness.

SPS


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU
   </td>
   <td>first iteration
<br/>
(ms)
   </td>
  </tr>
  <tr>
   <td>AO
   </td>
   <td>110
   </td>
   <td>170
   </td>
   <td>1339
   </td>
   <td>1
   </td>
   <td>562
   </td>
  </tr>
  <tr>
   <td>+ compile (cold)
   </td>
   <td>102
   </td>
   <td>158
   </td>
   <td>1343
   </td>
   <td>skipped
   </td>
   <td>319954
   </td>
  </tr>
  <tr>
   <td>+ compile (warm)
   </td>
   <td>100
   </td>
   <td>160
   </td>
   <td>1302
   </td>
   <td>0.9999
   </td>
   <td>8947
   </td>
  </tr>
</table>


MPS


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU
   </td>
   <td>first iteration
<br/>
(ms)
   </td>
  </tr>
  <tr>
   <td>AO + batching
   </td>
   <td>126
   </td>
   <td>225
   </td>
   <td>8021
   </td>
   <td>0.9999992
   </td>
   <td>504
   </td>
  </tr>
  <tr>
   <td>+ compile (cold)
   </td>
   <td>129
   </td>
   <td>215
   </td>
   <td>8021
   </td>
   <td>skipped
   </td>
   <td>333308
   </td>
  </tr>
  <tr>
   <td>+ compile (warm)
   </td>
   <td>113
   </td>
   <td>213
   </td>
   <td>8021
   </td>
   <td>0.998
   </td>
   <td>8617
   </td>
  </tr>
</table>



### furious: TF32, float16 and GPU preprocessing

We found that using float16 is the right level of precision for a few significant subcomponents of the model. In particular, the image encoder and mask decoder weights can be converted entirely to float16. We can also use TensorFloat32 precision for the remaining float32 matrix operations. It should be possible to further reduce the precision and we may address this in a future post. We also move image preprocessing such as image normalization onto the GPU with the furious mode. We can't use GPU decoding (nvJPEG) routines, because the differences are too significant and the model suffers from significant degradation in mIoU, so image decoding still happens on the CPU.

AMG


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU / 
<br/>
fail count
   </td>
  </tr>
  <tr>
   <td>AO 
<br/>
+ batching 
<br/>
+ compile (warm)
   </td>
   <td>439
   </td>
   <td>530
   </td>
   <td>29349
   </td>
   <td>0.994 / 190
   </td>
  </tr>
  <tr>
   <td>+ furious
   </td>
   <td>165
   </td>
   <td>240
   </td>
   <td>28335
   </td>
   <td>0.978 / 306
   </td>
  </tr>
</table>


This causes a significant degradation in mIoU for the AMG task, but doesn't affect the other tasks. After an in-depth investigation, we still chalk this up to numerical instability and reordering of operations. More work is needed to further investigate this and it may not be interesting to run the AMG task in lower precision. The other tasks, however, benefit drastically in latency with minimal changes in mIoU.

SPS


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU
   </td>
  </tr>
  <tr>
   <td>AO 
<br/>
+ compile (warm)
   </td>
   <td>100
   </td>
   <td>160
   </td>
   <td>1302
   </td>
   <td>0.9999
   </td>
  </tr>
  <tr>
   <td>+ furious
   </td>
   <td>32
   </td>
   <td>63
   </td>
   <td>861
   </td>
   <td>0.9997
   </td>
  </tr>
</table>


MPS


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU
   </td>
  </tr>
  <tr>
   <td>AO 
   <br/>
+ batching
<br/>
+ compile (warm)
   </td>
   <td>113
   </td>
   <td>213
   </td>
   <td>8021
   </td>
   <td>0.998
   </td>
  </tr>
  <tr>
   <td>+ furious
   </td>
   <td>36
   </td>
   <td>64
   </td>
   <td>4222
   </td>
   <td>0.997
   </td>
  </tr>
</table>



### AOTInductor's (AOTI) ahead-of-time compilation via torch.export

When scaling elastically it often is not possible to accommodate long startup times. That means the first iteration cannot be slow, but we must quickly deliver results. This is when torch.compile's current compilation overhead can get in the way. To address this we can use AOTInductor's (AOTI) ahead-of-time compilation via torch.export. AOTI lets us compile the model on a representative input and store the resulting code in a binary that is quick to load and run.

AOTI via torch.export is a new feature and we currently can't export everything that is compilable. We've been able to export the image encoder for all tasks but have only been able to export the mask prediction for the AMG and SPS tasks due to varying prompts. torch.export also supports dynamic shapes, but we need to invest a bit more time to prepare the code for it.

AMG: AO + batching + furious


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU / 
<br/>
fail count
   </td>
   <td>first iteration
<br/>
(ms)
   </td>
  </tr>
  <tr>
   <td>+ compile (warm)
   </td>
   <td>165
   </td>
   <td>240
   </td>
   <td>28335
   </td>
   <td>0.978 / 306
   </td>
   <td>10341
   </td>
  </tr>
  <tr>
   <td>+ load export
<br/>
(cold)
   </td>
   <td>162
   </td>
   <td>233
   </td>
   <td>27927
   </td>
   <td>0.974 / 308
   </td>
   <td>906
   </td>
  </tr>
</table>


SPS: AO + furious


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU
   </td>
   <td>first iteration
<br/>
(ms)
   </td>
  </tr>
  <tr>
   <td>+ compile (warm)
   </td>
   <td>32
   </td>
   <td>63
   </td>
   <td>861
   </td>
   <td>0.9997
   </td>
   <td>7989
   </td>
  </tr>
  <tr>
   <td>+ load export
<br/>
(cold)
   </td>
   <td>35
   </td>
   <td>66
   </td>
   <td>1686
   </td>
   <td>0.9997
   </td>
   <td>763
   </td>
  </tr>
</table>


Note that loading the exported model significantly increases memory. It likely only increases peak memory utilization, because initialization really needs to be delayed before loading up an exported model to avoid having twice the weights in memory at once. This is something we could address, but the memory consumption is nowhere near the limit. We don't see an increase in the other tasks, because AMG and MPS peak memory is dominated by processing batches of masks. One way to reduce that could be to operate on masks in the rle format (or some other sparse format) earlier on, but for now, there is no reason for this given the current memory consumption and focus on latency.

MPS: AO + batching + furious


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU
   </td>
   <td>first iteration
<br/>
(ms)
   </td>
  </tr>
  <tr>
   <td>+ compile (warm)
   </td>
   <td>36
   </td>
   <td>64
   </td>
   <td>4222
   </td>
   <td>0.997
   </td>
   <td>9626
   </td>
  </tr>
  <tr>
   <td>+ load export
<br/>
(cold)
   </td>
   <td>43
   </td>
   <td>72
   </td>
   <td>3813
   </td>
   <td>0.997
   </td>
   <td>747
   </td>
  </tr>
</table>


Using export by itself doesn't seem to benefit from extensive warmup and can be run in a pristine new inductor cache directory. But again, we do not evict the CUDA cache or other caches. In the section on Modal, we are running some of these experiments in a pristine environment.

When only processing 1000 images in a new process, using export can really be worth it to save out on compile and other cold start overhead.


### bonus: More GPU preprocessing

At this point, the latency is fairly low. In particular, for the SPS and MPS tasks we are processing at around 30ms to 40ms. Let's bring back the pseudo-code from the setup section again.


```
image_tensors = decode_img_bytes(...)
masks = gen_masks(image_tensors, ...)
rle_dicts = [rle_dict_from_masks(m) for m in masks]
```


Further profiling showed that at this point `decode_img_bytes` takes about 10ms. In particular, it uses torchvision's ToTensor transform to convert from a numpy Tensor to a scaled, float32 torch.Tensor. The bytes passed to ToTensor have already been decoded and converted to an numpy ndarray. By slightly rewriting ToTensor, using torchvision's v2 API and moving the uint8 decoded smaller integer Tensor to GPU first before scaling, we can gain another 10ms in latency. Without including `decode_img_bytes` in our analysis we would have missed this opportunity that has real-world impact on server-side inference.


```
image_tensor = torch.from_numpy(image_tensor)
image_tensor = image_tensor.permute((2, 0, 1))
image_tensor = image_tensor.cuda()
image_tensor = v2.ToDtype(torch.float32, scale=True)( image_tensor)
```


Note in particular that using pinned memory to perform asynchronous data transfers doesn't apply, since the time it takes to move the Tensor into pinned memory isn't worth the gain in asynchronicity for this data movement. For future work, we might want to explore further improvements here by using more advanced direct memory transfer techniques.

AMG: AO + batching + furious


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU / 
<br/>
fail count
   </td>
   <td>first iteration
<br/>
(ms)
   </td>
  </tr>
  <tr>
   <td>+ load export
<br/>
(cold)
   </td>
   <td>162
   </td>
   <td>233
   </td>
   <td>27927
   </td>
   <td>0.974 / 308
   </td>
   <td>906
   </td>
  </tr>
  <tr>
   <td>+ load export (warm)
   </td>
   <td>157
   </td>
   <td>230
   </td>
   <td>27927
   </td>
   <td>0.974 / 308
   </td>
   <td>799
   </td>
  </tr>
  <tr>
   <td>+ load export (warm)
<br/>
+ preproc
   </td>
   <td>136
   </td>
   <td>208
   </td>
   <td>27950
   </td>
   <td>0.977 / 311
   </td>
   <td>908
   </td>
  </tr>
</table>


SPS: AO + furious


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU
   </td>
   <td>first iteration
<br/>
(ms)
   </td>
  </tr>
  <tr>
   <td>+ load export
<br/>
(cold)
   </td>
   <td>35
   </td>
   <td>66
   </td>
   <td>1686
   </td>
   <td>0.9997
   </td>
   <td>763
   </td>
  </tr>
  <tr>
   <td>+ load export (warm)
   </td>
   <td>31
   </td>
   <td>63
   </td>
   <td>1686
   </td>
   <td>0.9997
   </td>
   <td>683
   </td>
  </tr>
  <tr>
   <td>+ load export (warm)
<br/>
+ preproc
   </td>
   <td>19
   </td>
   <td>25
   </td>
   <td>1711
   </td>
   <td>0.9997
   </td>
   <td>658
   </td>
  </tr>
</table>


MPS: AO + batching + furious


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>p50 latency (ms)
   </td>
   <td>p90 latency (ms)
   </td>
   <td>memory (MiB)
   </td>
   <td>mIoU
   </td>
   <td>first iteration
<br/>
(ms)
   </td>
  </tr>
  <tr>
   <td>+ load export
<br/>
(cold)
   </td>
   <td>43
   </td>
   <td>72
   </td>
   <td>3813
   </td>
   <td>0.997
   </td>
   <td>747
   </td>
  </tr>
  <tr>
   <td>+ load export (warm)
   </td>
   <td>53
   </td>
   <td>81
   </td>
   <td>3813
   </td>
   <td>0.997
   </td>
   <td>807
   </td>
  </tr>
  <tr>
   <td>+ load export (warm)
<br/>
+ preproc
   </td>
   <td>31
   </td>
   <td>41
   </td>
   <td>3837
   </td>
   <td>0.997
   </td>
   <td>671
   </td>
  </tr>
</table>


This small change has a significant impact on the SPS and MPS task.


## Deploying on Modal

Finally, we deployed our optimized inference onto [Modal](https://modal.com), a serverless infrastructure provider, to demonstrate that the benefits of these optimizations can be realized in a more realistic deployment setting.

In particular, compilation and AOTI via torch.export requires extra work. In a naïve deployment that work might be added to every single inference execution, adding latency that dwarfs any improvements from a faster model. This is particularly challenging with elastic or autoscaling infrastructure, where replicas of our inference service need to be regularly and automatically created and destroyed.

We share a deployment script in the torchao repository ([cli_on_modal.py](https://github.com/pytorch/ao/tree/main/examples/sam2_amg_server)) to demonstrate one pattern for an elastic deployment. We build the exported models ahead of time and then upload them to [distributed storage](https://modal.com/docs/guide/volumes). Relative to eager execution, this adds a bit of extra work when replicas spin up since they need to read this data over a network, but this is far less costly than compilation or export.

We benchmarked this deployment with a large batch inference workload: sending 1000 images for concurrent processing. The deployment scales up to ten replicas on ten GPUs at peak and scales down to zero GPUs when inactive.

First, let’s look at the execution latencies. 


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td colspan="3" >p50 execution latency
<br/>
(ms / improvement)
   </td>
   <td colspan="3" >p90 execution latency
<br/>
(ms / improvement)
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>eager float32
   </td>
   <td colspan="2" >AOTI float16
   </td>
   <td>eager float32
   </td>
   <td colspan="2" >AOTI float16
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
   </td>
   <td>Modal
   </td>
   <td>Offline
   </td>
   <td>
   </td>
   <td>Modal
   </td>
   <td>Offline
   </td>
  </tr>
  <tr>
   <td>AMG
   </td>
   <td>741
   </td>
   <td>112 (6.6x)
   </td>
   <td>136 (5.4x)
   </td>
   <td>1140
   </td>
   <td>176 (6.5x)
   </td>
   <td>208 (5.5x)
   </td>
  </tr>
  <tr>
   <td>SPS
   </td>
   <td>98
   </td>
   <td>20 (4.9x)
   </td>
   <td>19 (5.2x)
   </td>
   <td>130
   </td>
   <td>28 (4.6x)
   </td>
   <td>25 (5.2x)
   </td>
  </tr>
  <tr>
   <td>MPS
   </td>
   <td>269
   </td>
   <td>38 (7.1x)
   </td>
   <td>31 (8.7x)
   </td>
   <td>714
   </td>
   <td>52 (13.7x)
   </td>
   <td>41 (17.4x)
   </td>
  </tr>
</table>


We notice that execution latencies on Modal and Offline are fairly close, especially relative to the baseline, indicating that optimizing the deployment offline was a reasonable proxy for optimizing the deployment directly.

In addition to execution latency, our batch workload has queueing time, since there are fewer replicas than there are inputs, and so some inputs have to wait in line.


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td colspan="2" >p50 queue time (ms)
   </td>
   <td colspan="2" >p90 queue time (ms)
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>eager float32
   </td>
   <td>AOTI float16
   </td>
   <td>eager float32
   </td>
   <td>AOTI float16
   </td>
  </tr>
  <tr>
   <td>AMG
   </td>
   <td>201
   </td>
   <td>41 (4.9x)
   </td>
   <td>815
   </td>
   <td>327 (2.6x)
   </td>
  </tr>
  <tr>
   <td>SPS
   </td>
   <td>31
   </td>
   <td>33 (0.9x)
   </td>
   <td>441
   </td>
   <td>49 (9.0x)
   </td>
  </tr>
  <tr>
   <td>MPS
   </td>
   <td>40
   </td>
   <td>37 (1.1x)
   </td>
   <td>942
   </td>
   <td>75 (12.6x)
   </td>
  </tr>
</table>


Even though the queueing system provided by the infrastructure is unchanged, the queue latencies also decrease when we use our optimized model – in the p90 case by a factor of 2 to 12. That’s because when we finish previous inputs faster (from reduced execution latency) we can pull our next inputs sooner (reducing their queueing time).

If you’re interested in optimizing SAM2 inference or deployments further, don’t hesitate to reach out to us at the [torchao repository](https://github.com/pytorch/ao)!


## Conclusions

We rewrote Meta's original SAM2 in pure PyTorch with little loss of accuracy and a strong focus on latency. We deployed our optimized inference onto [Modal](https://modal.com), a serverless infrastructure provider, to demonstrate that the benefits of these optimizations can be realized in a more realistic deployment setting.

By utilizing AOTInductor's (AOTI) ahead-of-time compilation via torch.export, reduced precision, batched prompts and GPU preprocessing we observe up to 13x improvement in p90 execution latency and queue times compared to regular eager mode PyTorch.

With elastic or autoscaling infrastructure, where replicas of our inference service need to be regularly and automatically created and destroyed, a naïve deployment of torch.compile can add work to inference execution that dwarfs any improvements from a faster model. By utilizing AOTInductor's (AOTI) ahead-of-time compilation via torch.export, we are able to upload exported models ahead of time and read this data over a network, which enables us to get the benefits of compilation without significantly increased work.

For more details on how to reproduce the data in this blog post, [check out the experiments folder of torchao](https://github.com/pytorch/ao/tree/main/examples/sam2_amg_server). Please don't hesitate to contact us or [open an issue](https://github.com/pytorch/ao/issues/new) if you run into any technical issues.