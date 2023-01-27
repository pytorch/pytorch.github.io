---
layout: blog_detail
title: "Performance experiments with Stable Diffusion"
author: Grigory Sizov, Michael Gschwind, Hamid Shojanazeri, Driss Guessous, Daniel Haziza, Christian Puhrsch
hidden: true
---

*This is a companion to the main blog [“Accelerated Stable Diffusion with PyTorch 2”](/blog/accelerated-stable-diffusion-2/), containing detailed information on benchmarking setup and results of individual experiments. It is mainly aimed at a hands-on reader who would want to reproduce or develop further the work we described in the main text. Please see the main text for all the context and the summary of results.*


## Appendix 1: benchmarked versions definition

Here we define precisely what we mean by “original code” and “optimized code” in the main text.

**Original code**

Lives in [https://github.com/sgrigory/stablediffusion2](https://github.com/sgrigory/stablediffusion2) on `original-benchmark` branch, specifically in [this commit](https://github.com/sgrigory/stablediffusion2/tree/cee9b9f057eeef4b481e138da9dbc4fe8ecb0cba). This is almost the same code as in [https://github.com/Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion), with minimal modifications necessary for benchmarking. In particular, the code is able to turn off xFormers attention when the environment variable `USE_XFORMERS` is set to `False.`

  

This code uses PyTorch 1.12 and [the original custom implementation of attention](https://github.com/sgrigory/stablediffusion2/blob/cee9b9f057eeef4b481e138da9dbc4fe8ecb0cba/ldm/modules/attention.py#L165-L196).

**Optimized code**

The _optimized version_ is the code living [here](https://github.com/sgrigory/stablediffusion2/tree/0f6d17cb2602302bc0f5c7dee6825e4b49a85518). It has all the optimizations we mentioned in the main text:



* `nn.MultiheadAttention` in `CrossAttention` instead of custom attention implementation 
* Compilation with `torch.compile`
* Other minor optimizations in PyTorch-related code. 

The first optimization (using `nn.MultiheadAttention` in `CrossAttention`) schematically boils down to the following pseudocode:

```
class CrossAttention(nn.Module):
    def __init__(self, ...):
        # Create matrices: Q, K, V, out_proj
        ...
    def forward(self, x, context=None, mask=None):
       # Compute out = SoftMax(Q*K/sqrt(d))V
       # Return out_proj(out)
       …
```


gets replaced with

```
class CrossAttention(nn.Module):
    def __init__(self, ...):
        self.mha = nn.MultiheadAttention(...)
    def forward(self, x, context):
	return self.mha(x, context, context)
```

See the full diff [here](https://github.com/Stability-AI/stablediffusion/compare/main...sgrigory:stablediffusion2:optimize-w-compile?expand=1#diff-db5d837c282869a3588a17885e0baec3e29bf0701af6f4f34774d7b94503f7d4R145-R188).

We have also introduced the following CLI flags:



* `--disable_math, --disable_mem_efficient, --disable_flash` to allow turning specific attention backends off
* `--compile` to turn on PyTorch compilation

The optimized version uses PyTorch 2.0.0.dev20230111+cu117

**Flags added to both code versions**

In both code versions we have added the following CLI options to `txt2img.py`. 



* `--skip_first` to use a “warm-up” iteration before starting to measure time. See the end of section “Benchmarking setup and results summary” in [the main text](/blog/accelerated-stable-diffusion-2/) on why this was necessary
* `--time_file <FILENAME> `to write runtime in seconds in text format to the specified file

**Prompts**

Now it should already be clear how to run the 5 configurations mentioned in the main text. For completeness we provide the prompts which can be used to run each of them. This assumes you have 



* installed dependencies from the original version into conda environment` ldm-original`
* installed dependencies from the optimized version into conda environment `ldm`
* downloaded model weights into` /tmp/model.ckpt`
* converted model weights to the new architecture and saved them into `/tmp/model_native_mha.ckpt`

(see [Colab](https://colab.research.google.com/drive/1cSP5HoRZCbjH55MdYiRtxC_Q0obQQ5ZD?usp=sharing) for a bash script which does that)

Prompts for 5 configurations:

```
# Run optimized with memory-efficient attention and compilation
conda activate ldm
git checkout optmize-w-compile
python scripts/txt2img.py --prompt "A photo" --seed 1 --plms --config configs/stable-diffusion/v2-inference_native_mha.yaml --ckpt /tmp/model_native_mha.ckpt --n_iter 2 --n_samples 1 --compile --skip_first

# Run optimized with memory-efficient attention
conda activate ldm
git checkout optmize-w-compile
python stable-diffusion/scripts/txt2img.py --prompt "A photo" --seed 1 --plms --config stable-diffusion/configs/stable-diffusion/v2-inference_native_mha.yaml --ckpt /tmp/model_native_mha.ckpt --n_iter 2 --n_samples 1 --skip_first

# Run optimized without memory-efficient or flash attention
conda activate ldm
git checkout optmize-w-compile
python stable-diffusion/scripts/txt2img.py --prompt "A photo" --seed 1 --plms --config stable-diffusion/configs/stable-diffusion/v2-inference_native_mha.yaml --ckpt /tmp/model_native_mha.ckpt --n_iter 2 --n_samples 1 --disable_mem_efficient --disable_flash --skip_first 

# Run original code with xFormers
conda activate ldm-original
git checkout original-benchmark
python stable-diffusion-original/scripts/txt2img.py --prompt "A photo" --seed 1 --plms --config stable-diffusion-original/configs/stable-diffusion/v2-inference.yaml --ckpt /tmp/model.ckpt --n_iter 2 --n_samples 1 --skip_first

# Run original code without xFormers
conda activate ldm-original
git checkout original-benchmark
USE_XFORMERS=False python stable-diffusion-original/scripts/txt2img.py --prompt "A photo" --seed 1 --plms --config stable-diffusion-original/configs/stable-diffusion/v2-inference.yaml --ckpt /tmp/model.ckpt --n_iter 2 --n_samples 1 --skip_first
```

## Appendix 2: per-run data

Plots with per-run benchmark data can be found [here](https://drive.google.com/drive/folders/1NWIGDBAsMakMeByQU0FmMFtyoRsUI0pF?usp=share_link). Each plot shows all the runs for a particular GPU (P100, V100, T4, A10, A100) and batch size (1, 2, or 4). The bar charts in [the main text](/blog/accelerated-stable-diffusion-2/) are obtained from this data by averaging. The file names are self-explanatory, for example “original_vs_optimized_A10_n_samples_2_n_iter_2_sd2.png” contains runs for A10 GPU, batch size 2 and number of iterations 2. 


## Appendix 3: Accelerated Stable Diffusion 1

Before the work on Stable Diffusion 2 described in the main text, we also applied similar optimizations to [Stable Diffusion 1](https://github.com/CompVis/stable-diffusion) by CompVis prior to the release of Stable Diffusion 2. The original implementation of SD1 does not integrate with xFormers yet, and so the speedup from just using the PyTorch optimized attention instead of custom implementation is significant. It should be noted that the [HuggingFace Diffusers port of SD1](https://github.com/huggingface/diffusers#stable-diffusion-is-fully-compatible-with-diffusers) allows integration with xFormers, so an interesting open question which we didn’t explore would be how the performance of SD1 with PyTorch optimized attention compares to HuggingFace SD1+xFormers. 

We benchmarked two versions of SD1, _original and optimized_:



* As the _original_ version we took the first SD release, and placed it [here](https://github.com/sgrigory/stable-diffusion/tree/original-release) with minimal modifications to simplify benchmarking. It uses PyTorch 1.11 and custom implementation of attention.
* The _optimized_ version is the code living [here](https://github.com/sgrigory/stable-diffusion/tree/9809711e6921dfae8a4c2934f8c737bd03ad32a1). It uses `nn.MultiheadAttention` in `CrossAttention` and PyTorch 2.0.0.dev20221220+cu117.

Here are the results for different GPU architectures and batch size 2:


<table class="table">
  <tr>
   <td>
<strong>Version</strong>

   </td>
   <td><strong>T4</strong>
   </td>
   <td><strong>P100</strong>
   </td>
   <td><strong>V100</strong>
   </td>
   <td><strong>A100 </strong>
   </td>
  </tr>
  <tr>
   <td>
Original SD1 (runtime in s)

   </td>
   <td>70.9
   </td>
   <td>71.5
   </td>
   <td>20.3
   </td>
   <td>14.4
   </td>
  </tr>
  <tr>
   <td>
Optimized SD1 (runtime in s)

   </td>
   <td>52.7 <strong>(-25.6%)</strong>
   </td>
   <td>57.5 <strong>(-19.5%)</strong>
   </td>
   <td>14.3 <strong>(-29.3%)</strong>
   </td>
   <td>10.4 <strong>(</strong>-<strong>27.9%)</strong>
   </td>
  </tr>
</table>


Same as for SD2, we used Meta hardware for P100, V100, A100 benchmarks. The T4 benchmark was done in Google Colab [here](https://colab.research.google.com/drive/1E83F4o6yePnXTI0vUsQTggiZabLipTCD?usp=sharing).

We didn’t apply compilation to SD1, and so didn’t include a “warm-up” iteration in these benchmarks, as we did for SD2.

Both applying `torch.compile` to SD1 and benchmarking HuggingFace version of SD1 with PyTorch 2 optimisations would be a great exercise for the reader - try it and let us know if you get interesting results.
