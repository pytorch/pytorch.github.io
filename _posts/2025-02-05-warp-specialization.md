---
layout: blog_detail
title: "Enabling advanced GPU features in PyTorch - Warp Specialization"
author: "Meta and NVIDIA" 
---

**Meta**: Hongtao Yu, Manman Ren, Bert Maher, Shane Nay    
**NVIDIA**: Gustav Zhu, Shuhao Jiang

Over the past few months, we have been working on enabling advanced GPU features for PyTorch and Triton users through the Triton compiler. One of our key goals has been to introduce warp specialization support on NVIDIA Hopper GPUs. Today, we are thrilled to announce that our efforts have resulted in the rollout of fully automated Triton warp specialization, now available to users in the upcoming release of Triton [3.2](https://github.com/triton-lang/triton/tree/release/3.2.x), which will ship with PyTorch 2.6. PyTorch users can leverage this feature by [implementing user-defined Triton kernels](https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html). This work leveraged an initial implementation of warp specialization in Triton by NVIDIA and we look forward to further development with the community in the future.

Warp specialization (WS) is a GPU programming technique where warps (a group of 32 threads on NVIDIA GPUs) within a threadblock are assigned distinct roles or tasks. This approach optimizes performance by enabling efficient execution of workloads that require task differentiation or cooperative processing. It enhances kernel performance by leveraging an asynchronous execution model, where different parts of the kernel are managed by separate hardware units. Data communication between these units, facilitated via shared memory on the NVIDIA H100, is highly efficient. Compared to a uniform warp approach, warp specialization allows the hardware multitasking warp scheduler to operate more effectively, maximizing resource utilization and overall performance.

Using GEMM as an example, a typical uniform warp approach on the H100 GPU involves 8 warps per thread block collectively computing a tile of the output tensor. These 8 warps are divided into two warp groups (WG), with each group cooperatively computing half of the tile using efficient warp-group-level MMA (WGMMA) instructions, as illustrated in Figure 1.


![Figure 1. GEMM K-loop Body with Uniform Warps](/assets/images/warp-specialization/fg1.jpg){:style="width:100%"}

Figure 1. GEMM K-loop Body with Uniform Warps

The implementation is clean, easy to understand, and generally performs well, thanks to an elegant software pipeliner. The pipeliner's purpose is to enhance instruction-level parallelism by executing non-dependent operations on different hardware units. For instance, load operations from the next loop iteration can be executed simultaneously with WGMMA operations in the current iteration. However, this approach relies heavily on the compiler to craft an instruction sequence that ensures load and WGMMA instructions are issued at precisely the right time. While this is relatively straightforward for GEMM, which involves a limited number of operations, it becomes significantly more challenging for more complex kernels, such as flash attention.

On the other hand, warp specialization addresses programming challenges by separating operations intended to run simultaneously on different hardware units into distinct warps, synchronizing them efficiently using low-cost barriers in shared memory. This allows each warp to have its own instruction sequence, enabling instructions to be issued and executed continuously without being interrupted by other operations, thanks to the multi-way warp scheduler. An illustration of a warp-specialized GEMM can be seen in Figure 2.


![Figure 2. GEMM K-loop Body with Specialized Warps](/assets/images/warp-specialization/fg2.jpg){:style="width:100%"}

Figure 2. GEMM K-loop Body with Specialized Warps


## How to enable WS

To enable warp specialization, users simply need to specify two autotune flags: num_consumer_groups and num_buffers_warp_spec. For example, a warp-specialized GEMM implementation might look as shown below.  Users can enable warp specialization by setting a non-zero value for num_consumer_groups, which defines the number of consumer warp groups. There is no corresponding flag to set the number of producer warp groups, as currently only one producer is supported. The num_buffers_warp_spec flag specifies the number of buffers the producer warp group will use to communicate with the consumer warp groups. A working example of a warp-specialized kernel is provided in the persistent GEMM [tutorial](https://github.com/triton-lang/triton/blob/6771065cb3137f7e64454cc047b9b74d577cbf7f/python/tutorials/09-persistent-matmul.py#L620).

```
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=4,
            num_consumer_groups=2,
            num_buffers_warp_spec=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_persistent_ws_kernel(
   a_ptr, b_ptr, c_ptr, M, N, K,
   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
   pid = tl.program_id(axis=0)
   num_pid_m = tl.cdiv(M, BLOCK_M)
   num_pid_n = tl.cdiv(N, BLOCK_N)
   pid_m = pid // num_pid_m
   pid_n = pid % num_pid_n
   offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
   offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
   offs_k = tl.arange(0, BLOCK_K)
   a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
   b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
   acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
   for k in range(0, tl.cdiv(K, BLOCK_K)):
       a = tl.load(a_ptrs)
       b = tl.load(b_ptrs)
       acc += tl.dot(a, b)
       a_ptrs += BLOCK_K * stride_ak
       b_ptrs += BLOCK_K * stride_bk
   c = acc.to(tl.float16)
   c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
   tl.store(c_ptrs, c)
```


## Under the Hood

Warp specialization uses a set of hierarchical compiler transformations and IR changes to translate a user's non-warp-specialized kernel into warp-specialized machine code. These include:



* **Task Partitioning**: The entire kernel is automatically divided into asynchronous tasks based on predefined heuristics. The compiler determines how to utilize one producer warp group and a user-specified number of consumer warp groups to execute the kernel. It assigns task IDs to specific anchor operations, which then influence the task assignments for remaining operations through asynchronous task ID propagation and dependency analysis. Since shared memory is the most efficient method for data transfer between warp groups across all supported platforms, the compiler optimizes task partitions to minimize register spills to shared memory, ensuring efficient execution.
* **Data Partitioning for Multiple Consumer Groups**: Efficiently partitioning data among multiple consumer groups is key to optimizing workload distribution. On the H100 GPU, the compiler, by default, attempts to partition the input tensor `A` along the `M` dimension, allowing each consumer group to compute half of the output tensor independently. This strategy, known as [cooperative partitioning](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#warp-specialization), maximizes efficiency under most conditions. However, if this split leads to inefficiencies—such as producing a workload smaller than the native WGMMA instruction size—the compiler dynamically adjusts and partitions along the `N` dimension instead.
* **Dataflow Pipelining**: The compiler creates cyclic shared memory buffers to pipeline dataflows across multiple-dimensional loops. Warp-specialized pipelining supports complex control flow. For example, our warp-specialized persistent GEMM kernel uses a doubly-nested loop, allowing the producer to begin fetching data for the next output tile while the consumer is finishing the compute for the prior tile.
* **Communication Operations**`: `We introduced four high-level Triton GPU IR (TTGIR) communication operations`—ProducerAcquireOp, ProducerCommitOp, ConsumerWaitOp, `and` ConsumerReleaseOp—`to manage pipelined dataflows. These support both TMA and non-TMA memory operations.
* **Code Partitioning**: Each async task is outlined into its own standalone code region, guarded by warp group ID checks. Control dependencies are duplicated as needed. 
* **TTGIR to LLVM/PTX Materialization**: TTGIR communication operations are materialized into corresponding LLVM/PTX barrier operations.


## Performance

The [warp specialization release](https://github.com/triton-lang/triton/pull/5622) introduces a range of Triton compiler transformations that collectively convert user code into warp-specialized kernels. This feature has been applied to several key kernels, including Flash Attention and FP8 row-wise GEMM, resulting in significant performance gains of 10% to 15%. Below, we highlight the latest performance metrics for these high-impact kernels.


![bar chart](/assets/images/warp-specialization/fg3.png){:style="width:100%"}




![bar chart](/assets/images/warp-specialization/fg4.png){:style="width:100%"}



## Future Work

Looking ahead, we plan to further enhance Triton's warp specialization support by introducing new features such as Ping-Pong scheduling, expanded buffer sharing support, improved transparent handling for TMA, refined partitioning heuristics for upcoming NVIDIA hardware.