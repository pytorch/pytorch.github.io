---
layout: blog_detail
title: "vLLM Joins PyTorch Ecosystem: Easy, Fast, and Cheap LLM Serving for Everyone"
author: vLLM Team
hidden: true
---

![vllm logo](/assets/images/vllm.png){:style="width:100%;display: block;max-width:400px; margin-left:auto; margin-right:auto;"}

We’re thrilled to announce that the [vLLM project](https://github.com/vllm-project/vllm) has become a PyTorch ecosystem project, and joined the PyTorch ecosystem family!

Running large language models (LLMs) is both resource-intensive and complex, especially as these models scale to hundreds of billions of parameters. That’s where vLLM comes in — a high-throughput, memory-efficient inference and serving engine designed for LLMs.

Originally built around the innovative [PagedAttention algorithm](https://arxiv.org/abs/2309.06180), vLLM has grown into a comprehensive, state-of-the-art inference engine. A thriving community is also continuously adding new features and optimizations to vLLM, including the following:



* [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
* [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)
* [Llumnix: Dynamic Scheduling for Large Language Model Serving](https://arxiv.org/abs/2406.03243)
* [CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving](https://blog.vllm.ai/2024/07/25/lfai-perf.html#:~:text=CacheGen%3A%20KV%20Cache%20Compression%20and%20Streaming%20for%20Fast%20Large%20Language%20Model%20Serving)
* [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://blog.vllm.ai/2024/07/25/lfai-perf.html#:~:text=vAttention%3A%20Dynamic%20Memory%20Management%20for%20Serving%20LLMs%20without%20PagedAttention)
* [Andes: Defining and Enhancing Quality-of-Experience in LLM-Based Text Streaming Services](https://blog.vllm.ai/2024/07/25/lfai-perf.html#:~:text=Andes%3A%20Defining%20and%20Enhancing%20Quality%2Dof%2DExperience%20in%20LLM%2DBased%20Text%20Streaming%20Services)
* [SGLang: Efficient Execution of Structured Language Model Programs](https://blog.vllm.ai/2024/07/25/lfai-perf.html#:~:text=SGLang%3A%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs)

Since its release, vLLM has garnered significant attention, achieving over 31,000 GitHub stars—a testament to its popularity and thriving community. This milestone marks an exciting chapter for vLLM as we continue to empower developers and researchers with cutting-edge tools for efficient and scalable AI deployment. Welcome to the next era of LLM inference!

vLLM has always had a strong connection with the PyTorch project. It is deeply integrated into PyTorch, leveraging it as a unified interface to support a wide array of hardware backends. These include NVIDIA GPUs, AMD GPUs, Google Cloud TPUs, Intel GPUs, Intel CPUs, Intel Gaudi HPUs, and AWS Neuron, among others. This tight coupling with PyTorch ensures seamless compatibility and performance optimization across diverse hardware platforms.

Do you know you can experience the power of vLLM right from your phone? During this year’s Amazon Prime Day, vLLM played a crucial role in [delivering lightning-fast responses to millions of users](https://aws.amazon.com/cn/blogs/machine-learning/scaling-rufus-the-amazon-generative-ai-powered-conversational-shopping-assistant-with-over-80000-aws-inferentia-and-aws-trainium-chips-for-prime-day/). Across three regions, over 80,000 Trainium and Inferentia chips powered an average of 3 million tokens per minute, all while maintaining a P99 latency of less than 1 second for the first response. That means when customers opened the Amazon app and chatted with Rufus, they were seamlessly interacting with vLLM in action!

vLLM also collaborates tightly with leading model vendors to ensure support for popular models. This includes tight integration with Meta LLAMA, Mistral, QWen, and DeepSeek models, plus many others. One particularly memorable milestone was the [release of LLAMA 3.1 (405B)](https://ai.meta.com/blog/meta-llama-3-1/). As the launching partner, vLLM was the first to enable running this very large model, showcasing vLLM’s capability to handle the most complex and resource-intensive language models.

To install vLLM, simply run:


```
pip install vllm
```


vLLM is designed for both researchers and production-grade serving.

To run vLLM as an OpenAI API compatible server, just use the Huggingface model ID:


```
vllm serve meta-llama/Llama-3.1-8B
```


To run vLLM as a simple function:


```
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
   "Hello, my name is",
   "The president of the United States is",
   "The capital of France is",
   "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="meta-llama/Llama-3.1-8B")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
   prompt = output.prompt
   generated_text = output.outputs[0].text
   print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


Open-source innovation is part of the vLLM’s DNA. Born out of a Berkeley academic project, it follows the legacy of other pioneering open-source initiatives such as BSD, which revolutionized operating systems in the 1980s. Other innovations from the same organization include [Apache Spark](https://github.com/apache/spark) and [Ray](https://github.com/ray-project/ray), now the standard for big data and AI systems. In the Gen AI era, vLLM serves as a platform dedicated to democratizing AI inference.

The vLLM team remains steadfast in its mission to keep the project “of the community, by the community, and for the community.” Collaboration and inclusivity lie at the heart of everything we do.

If you have collaboration requests or inquiries, feel free to reach out at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu). To join the active and growing vLLM community, explore our [GitHub repository](https://github.com/vllm-project/vllm) or connect with us on the [vLLM Slack](https://slack.vllm.ai). Together, we can push the boundaries of AI innovation and make it accessible to all.