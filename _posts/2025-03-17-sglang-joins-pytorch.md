---
layout: blog_detail
title: "SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine"
author: "SGLang Team"
hidden: true
---


![sglang logo](/assets/images/sglang-join-pytorch/fg1.png){:style="max-width:400px; display: block; margin-left: auto; margin-right: auto"}


We’re thrilled to announce that the SGLang project has been integrated into the PyTorch ecosystem! This integration ensures that SGLang aligns with PyTorch’s standards and practices, providing developers with a reliable and community-supported framework for fast and flexible serving of LLMs.

To view the PyTorch Ecosystem, see the [PyTorch Landscape](https://landscape.pytorch.org/) and learn more about how projects can [join the PyTorch Ecosystem](https://github.com/pytorch-fdn/ecosystem). 


## About SGLang

SGLang is a fast-serving engine for large language models and vision language models. It makes the interaction with models faster and more controllable by co-designing the backend runtime and frontend language.

The core features include:

* Fast Backend Runtime: Provides efficient serving with RadixAttention for prefix caching, zero-overhead CPU scheduler, continuous batching, token attention (paged attention), speculative decoding, tensor parallelism, chunked prefill, structured outputs, and quantization (FP8/INT4/AWQ/GPTQ).
* Flexible Frontend Language: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
* Extensive Model Support: Supports a wide range of generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), with easy extensibility for integrating new models.
* Active Community: SGLang is open source and backed by an active community with industry adoption.

SGLang is famous for its fast speed. It can often significantly outperform other state-of-the-art frameworks in terms of serving throughput and latency. You can learn more about the underlying techniques from the past release blog posts: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).

SGLang has been widely adopted by leading industry companies and frontier research labs. For example, xAI uses SGLang to serve its flagship model, [Grok 3](https://grok.com/), which is currently the best model according to the Chatbot Arena leaderboard. Microsoft Azure uses SGLang to serve [DeepSeek R1](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/running-deepseek-r1-on-a-single-ndv5-mi300x-vm/4372726) on AMD GPUs, which is currently the best open source model.


## Serving DeepSeek Models

You can easily launch a Docker container to serve a DeepSeek model with the following command:

```
# Pull the latest image
docker pull lmsysorg/sglang:latest

# Launch a server
docker run --gpus all --shm-size 32g -p 30000:30000 -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host --network=host --privileged lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000
```

Then you can query the server with the OpenAI-compatible API

```
import openai
client = openai.Client(base_url=f"http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
```

The server launch command above works for 8xH200. You can find detailed instructions for other hardware (MI300X, H100, A100, H20, L40S) at https://docs.sglang.ai/references/deepseek.html.

SGLang integrates DeepSeek-specific optimizations, such as MLA throughput optimizations, MLA-optimized kernels, data-parallel attention, multi-token prediction, and DeepGemm, making it the top choice for serving DeepSeek models by dozens of [companies](https://x.com/lmsysorg/status/1887262321636221412), including AMD, NVIDIA, and many cloud providers. The team is actively working on integrating more optimizations following the 2025 H1 roadmap below.


## Serving Llama Models

Similarly, you can launch the server for a Llama 3.1 text model with:

```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct
```

Or a Llama 3.2 multimodal model with:

```
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.2-11B-Vision-Instruct  --chat-template=llama_3_vision
```


## Roadmap

This year, the SGLang team will continue to push the boundaries of system efficiency. You can find the roadmap of 2025H1 [here](https://github.com/sgl-project/sglang/issues/4042). The focus is

- Throughput-oriented large-scale deployment similar to the DeepSeek inference system
- Long context optimizations
- Low latency speculative decoding
- Reinforcement learning training framework integration
- Kernel optimizations

## Community

SGLang has been deployed to large-scale production, generating trillions of tokens every day. It has an active community with over three hundred contributors on GitHub. It is supported by the following institutions: AMD, Atlas Cloud, Baseten, Cursor, DataCrunch, Etched, Hyperbolic, iFlytek, Jam & Tea Studios, LinkedIn, LMSYS, Meituan, Nebius, Novita AI, NVIDIA, RunPod, Stanford, UC Berkeley, UCLA, xAI, and 01.AI. 


![logos](/assets/images/sglang-join-pytorch/fg2.png){:style="width:100%;"}



## Conclusion

We’re excited to welcome SGLang to the PyTorch ecosystem. SGLang accelerates the serving of large language and vision language models. It’s widely adopted by industry, powering the large-scale online serving of frontier models like Grok and DeepSeek.

We invite you to explore the [SGLang GitHub repo](https://github.com/sgl-project/sglang/tree/main), join the [community on Slack](https://slack.mindee.com/), and reach out to [contact@sglang.ai](mailto:contact@sglang.ai) for inquiries or collaboration opportunities. Together, we can make powerful AI models accessible to everyone.