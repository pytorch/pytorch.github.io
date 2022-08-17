---
layout: blog_detail
title: "Fast Beam Search Decoding in PyTorch with TorchAudio and Flashlight Text"
author: PlayTorch Team
featured-img: ""
---

Beam search decoding with industry-leading speed from [Flashlight Text](https://github.com/flashlight/text) is now available with official support in [TorchAudio](https://pytorch.org/audio/0.12.0/models.decoder.html#ctcdecoder), bringing high-performance beam search and text utilities for speech and text applications built on top of PyTorch. The current integration supports CTC-style decoding, but it can be used for *any modeling setting that outputs token-level probability distributions over time steps*.

## A brief beam search refresher

In speech and language settings, *beam search* is an efficient, greedy algorithm that can convert sequences of *continuous values* (i.e. probabilities or scores) into *graphs* or *sequences* (i.e. tokens, word-pieces, words) using *optional constraints* on valid sequences (i.e. a lexicon), *optional external scoring* (i.e. an LM which scores valid sequences), and other *score adjustments* for particular sequences.

In the example that follows, we'll consider — a token set of {ϵ, a, b}, where ϵ is a special token that we can imagine denotes a space between words or a pause in speech. Graphics here and below are taken from Awni Hannun's excellent [distill.pub writeup](https://distill.pub/2017/ctc/) on CTC and beam search.

<p align="center">
  <img src="\assets\images\fast-beam-search-decoding-in-pytorch-with-torchaudio-and-flashlight-text-1.jpeg" width="100%">
</p>

With a greedy-like approach, beam search considers the next viable token given an existing sequence of tokens — in the example above, a, b, b is a valid sequence, but a, b, a is not. We *rank* each possible next token at each step of the beam search according to a scoring function. Scoring functions (s) typically looks something like:

<p align="center">
  <img src="\assets\images\fast-beam-search-decoding-in-pytorch-with-torchaudio-and-flashlight-text-2.jpeg" width="100%">
</p>

Where **ŷ** is a potential path/sequence of tokens, **x** is the input *<strong>(P(ŷ|x)</strong>* represents the model's predictions over time), and 𝛼 is a weight on the language model probability *<strong>(P(y)</strong>* the probability of the sequence under the language model). Some scoring functions add *<strong>𝜷</strong>* which adjusts a score based on the length of the predicted sequence **|ŷ|**. This particular scoring function is used in [FAIR's prior work](https://arxiv.org/pdf/1911.08460.pdf) on end-to-end ASR, and there are many variations on scoring functions which can vary across application areas.

Given a particular sequence, to assess the next viable token in that sequence (perhaps constrained by a set of allowed words or sequences, such as a lexicon of words), the beam search algorithm scores the sequence with each candidate token added, and sorts token candidates based on those scores. For efficiency and since the number of paths is exponential in the token set size, the *<strong>top-k</strong>* highest-scoring candidates are kept — *<strong>k</strong>* represents the *<strong>beam size</strong>*.

<p align="center">
  <img src="\assets\images\fast-beam-search-decoding-in-pytorch-with-torchaudio-and-flashlight-text-3.jpeg" width="100%">
</p>
<p align="center">There are many other nuances with how beam search can progress: similar hypothesis sequences can be “merged”, for instance.
</p>

The scoring function can be further augmented to up/down-weight token insertion or long or short words. Scoring with *stronger external language* models, while incurring computational cost, can also significantly improve performance; this is frequently referred to as *LM fusion*. There are many other knobs to tune for decoding — these are documented in [TorchAudio’s documentation](https://pytorch.org/audio/0.12.0/models.decoder.html#ctcdecoder) and explored further in [TorchAudio’s ASR Inference tutorial](https://pytorch.org/audio/0.12.0/tutorials/asr_inference_with_ctc_decoder_tutorial.html#beam-search-decoder-parameters). Since decoding is quite efficient, parameters can be easily swept and tuned.

Beam search has been used in ASR extensively over the years in far too many works to cite, and in strong, recent results and systems including [wav2vec 2.0](https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf) and [NVIDIA's NeMo](https://developer.nvidia.com/nvidia-nemo).

## Why beam search?

Beam search remains a fast competitor to heavier-weight decoding approaches such as [RNN-Transducer](https://arxiv.org/pdf/1211.3711.pdf) that Google has invested in putting [on-device](https://ai.googleblog.com/2019/03/an-all-neural-on-device-speech.html) and has shown strong results with on [common benchmarks](https://arxiv.org/pdf/2010.10504.pdf). Autoregressive text models at scale can benefit from beam search as well. Among other things, fast beam search gives:

- A flexible performance/latency tradeoff — by adjusting beam size and the external LM, users can sacrifice latency for accuracy or pay for more accurate results with a small latency cost. Decoding with no external LM can improve results at very little performance cost.
- Portability without retraining — existing neural models can benefit from multiple decoding setups and plug-and-play with external LMs without training or fine-tuning.
- A compelling complexity/accuracy tradeoff — adding beam search to an existing modeling pipeline incurs little additional complexity and can improve performance.

## Performance Benchmarks

Today's most commonly-used beam search decoding libraries today that support external language model integration include Kensho's [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode), NVIDIA's [NeMo toolkit](https://github.com/NVIDIA/NeMo/tree/stable/scripts/asr_language_modeling). We benchmark the TorchAudio + Flashlight decoder against them with a *wav2vec 2.0* base model trained on 100 hours of audio evaluated on [LibriSpeech](https://www.openslr.org/12) dev-other with the official [KenLM](https://github.com/kpu/kenlm/) 3-gram LM. Benchmarks were run on Intel E5-2698 CPUs on a single thread. All computation was in-memory — KenLM memory mapping was disabled as it wasn't widely supported.

When benchmarking, we measure the *time-to-WER (word error rate)* — because of subtle differences in the implementation of decoding algorithms and the complex relationships between parameters and decoding speed, some hyperparameters differed across runs. To fairly assess performance, we first sweep for parameters that achieve a baseline WER, minimizing beam size if possible.

<p align="center">
  <img src="\assets\images\fast-beam-search-decoding-in-pytorch-with-torchaudio-and-flashlight-text-4.jpeg" width="100%">
</p>

## TorchAudio API and Usage

TorchAudio provides a Python API for CTC beam search decoding, with support for the following:

- lexicon and lexicon-free decoding
- KenLM n-gram language model integration
- character and word-piece decoding
- sample pretrained LibriSpeech KenLM models and corresponding lexicon and token files
- various customizable beam search parameters (beam size, pruning threshold, LM weight...)

To set up the decoder, use the factory function torchaudio.models.decoder.ctc_decoder

```python
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
files = download_pretrained_files("librispeech-4-gram")
decoder = ctc_decoder(
   lexicon=files.lexicon,
   tokens=files.tokens,
   lm=files.lm,
   nbest=1,
   ... additional optional customizable args ...
)
```

Given emissions of shape *(batch, time, num_tokens)*, the decoder will compute and return a List of batch Lists, each consisting of the nbest hypotheses corresponding to the emissions. Each hypothesis can be further broken down into tokens, words (if a lexicon is provided), score, and timesteps components.

```python
emissions = acoustic_model(waveforms)  # (B, T, N)
batch_hypotheses = decoder(emissions)  # List[List[CTCHypothesis]]

# transcript for a lexicon decoder
transcripts = [" ".join(hypo[0].words) for hypo in batch_hypotheses]

# transcript for a lexicon free decoder, splitting by sil token
batch_tokens = [decoder.idxs_to_tokens(hypo[0].tokens) for hypo in batch_hypotheses]
transcripts = ["".join(tokens) for tokens in batch_tokens]
```

Please refer to the [documentation](https://pytorch.org/audio/stable/models.decoder.html#ctcdecoder) for more API details, and the tutorial ([ASR Inference Decoding](https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html)) or sample [inference script](https://github.com/pytorch/audio/tree/main/examples/asr/librispeech_ctc_decoder) for more usage examples.

## Upcoming Improvements

**Full NNLM support** — decoding with large neural language models (e.g. transformers) remains somewhat unexplored at scale. Already supported in Flashlight, we'll add support in TorchAudio, allowing users to use custom decoder-compatible LMs. Custom word level language models are already available in the nightly TorchAudio build, and will be released in TorchAudio 0.13.

**Autoregressive/seq2seq decoding** — Flashlight Text also supports [sequence-to-sequence (seq2seq) decoding](https://github.com/flashlight/text/blob/main/flashlight/lib/text/decoder/LexiconSeq2SeqDecoder.h) for autoregressive models, which we hope to add bindings for and add to TorchAudio and TorchText with efficient GPU implementations as well.

**Better build support** — to benefit from improvements in Flashlight Text, TorchAudio will directly submodule Flashlight Text to make upstreaming modifications and improvements easier. This is already in effect in the nightly TorchAudio build, and will be released in TorchAudio 0.13.