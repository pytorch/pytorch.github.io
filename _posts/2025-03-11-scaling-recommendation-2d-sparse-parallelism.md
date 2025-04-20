---
layout: blog_detail
title: "Scaling Recommendation Systems Training to Thousands of GPUs with 2D Sparse Parallelism"
author: "PyTorch Team at Meta: Chunzhi Yang, Rich Zhu, Zain Huda, Liangbei Xu, Xin Zhang, Jiyan Yang, Dennis van der Staay, Wang Zhou, Jin Fang, Jade Nie, Yuxi Hu"
---

At Meta, recommendation systems are the cornerstone of delivering relevant and personalized ads to billions of users globally. Through technologies like PyTorch's TorchRec, we've successfully developed solutions that enable model training across hundreds of GPUs. While these systems have served us well, recent research on scaling laws has revealed a compelling opportunity: we can achieve significantly better model performance by training dramatically larger neural networks.

However, this insight presents us with a new challenge. Our current training infrastructure, though highly optimized for hundreds of GPUs, cannot efficiently scale to the thousands of GPUs needed to train these larger models. The leap from hundreds to thousands of GPUs introduces complex technical challenges, particularly around handling sparse operations in recommendation models. These challenges require fundamentally new approaches to distributed training, which we address with a novel parallelization strategy.

**To address these issues, we introduced 2D embedding parallel, a novel parallelism strategy that overcomes the sparse scaling challenges inherent in training large recommendation models across thousands of GPUs. This is available today in TorchRec through the DMPCollection API.** This approach combines two complementary parallelization techniques: data parallelism for the sparse components of the model, and model parallelism for the embedding tables, leveraging TorchRec's robust sharding capabilities. By strategically integrating these techniques, we've created a solution that scales to thousands of GPUs and now powers Meta's largest recommendation model training runs.

**What are the sparse scaling challenges?**

We identified three key challenges that prevented us from naively scaling our model to thousands of GPUs:

* **Imbalancing and straggler issue:** with more GPUs it’s harder to achieve balanced sharding, some ranks can have much heavier workload for embedding computations, which can slow down the entire training. 
* **Communication across nodes:** As training jobs utilize an increased number of GPUs, the all-to-all communication bandwidth can drop under certain network topologies which can increase communication latency significantly.  
* **Memory overhead:** The memory used by input features is often negligible, however, as we use thousands of GPUs, we can introduce larger input features and the memory requirements can become significant.

With 2D embedding parallel, we can describe our new parallelism scheme like this, in this example we have 2 model replicas (Replica 1: GPU1/GPU3, Replica 2: GPU2/GPU4)


![Flow diagram](/assets/images/scaling-recommendation-2d-sparse-parallelism/fg1.png){:style="width:100%"}

***Figure 1: Layout illustration of 2D Sparse Parallelism***

With 2D sparse parallelism we address these challenges, instead of sharding tables across all ranks, we first evenly divide all ranks into several parallel groups:



1. Within each group, we use model parallel for the embedding tables, such as column-wise/row-wise sharding. At scale, for our largest tables, we have also developed a grid sharding, which shards embedding tables on the row and column dimension. 
2. Across groups, we do data parallel, such that each rank in a group has its corresponding replica rank in the other groups (replica rank means storing the same embedding table shards). 
    1. After each group has completed its own backward pass, we all reduce the embedding table weights across the replicas to keep them synchronized.

## Our production solution

TorchRec is our library to build the sparse part of the recommendation models in native PyTorch. With the traditional API being DistributedModelParallel which applies model parallel to the embedding tables. We introduce a new API alongside it, known as DMPCollection, which serves as the main entry point for enabling 2D parallel on TorchRec models. We designed it to be as easy of a change as applying FSDP/DDP is. 

To understand what DMPCollection does, we have to understand what DistributedModelParallel (DMP) does first:



1. Create embedding tables, known as EmbeddingBagCollection and EmbeddingCollections.
2. Generate a sharding plan with respect to GPU topology, embedding tables, memory available, input data, and more.
3. Wrap model with DMP and the associated sharding plan passed in.
4. DMP initializes and shards the embedding tables in accordance with the sharding plan.
5. On a train step, DMP takes an input batch, communicates it to the appropriate GPUs containing the embedding table shard of interest, looks up the value, and returns it back to the GPU that requested it. This is all done on the global process group, with some exceptions for special sharding (such as table row wise sharding)

DistributedModelParallel was built for model parallel with many parts working under the assumption of sharding and working around the global world size. We need to change these parts in a way where we can introduce additional dimensions of parallelism without losing the optimizations and feature set of TorchRec.

DMPCollection changes a few key parts to enable 2D parallel in an extensible way,



* Generate sharding plans for the smaller sharding group once, once passed in we communicate to the appropriate ranks across the global group and remap the ranks to fit the new sharding group ranks.
* Create two new NCCL process groups, known as sharding and replica process groups. The sharding process group is passed into sharding and train step components of TorchRec. The replica process group is used for the weight and optimizer state synchronization, the all reduce call happens over this process group.
    * The sub NCCL process groups allow us to efficiently communicate only between the ranks that are relevant for a particular comm. Each rank will have two associated process groups.

To the user, the change is very simple, while taking away all the complexity around applying the parallelism strategies to the model. 

## How do we create these sharding and replication groups?

These process groups are one of the keys to DMPCollection’s performant implementation. From our earlier diagram, we showed a simple 2x2 GPU setup, however, at scale, how do we assign which ranks are part of a given sharding group and what are their replica ranks across the sharding groups? 

Consider the following setup with 2 nodes, each with 4 GPUs. The sharding and replication groups under 2D parallel will be,


<table>
  <tr>
   <td>

<table class="table table-bordered">
  <tr>
   <td>Sharding Group
   </td>
   <td>Sharding Ranks
   </td>
  </tr>
  <tr>
   <td>0
   </td>
   <td>0, 2, 4, 6
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>1, 3, 5, 7
   </td>
  </tr>
</table>


   </td>
   <td>

<table class="table table-bordered">
  <tr>
   <td>Replication Group
   </td>
   <td>Replication Ranks
   </td>
  </tr>
  <tr>
   <td>0
   </td>
   <td>0, 1
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>2, 3
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>4, 5
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>6, 7
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>


We use the following formulation,



1. Divide all trainers into G sharding groups, each with L trainers
    1. Groups, G, is determined by G = T / L, where T is total number of trainers
2. For each group, G, we assigned non-contiguous trainer ranks based on the group it’s in, following, 
    2. [i, G+i, 2G+i, ..., (L - 1) G+i], where* i = 0 to G-1*
3. From the groups, G, we can create the replication group, which is every G continuous ranks
    3. (0 to G-1, G to 2* G - 1) each continuous set stores the duplicate embedding table shards.

This means our sharding groups, G, are of size L, which can be known as the number of ranks to apply model parallel across. This, in turn, gives us replica groups, each of size G, which are the ranks we data parallel across. 

In DMPCollection, we’re able to create these process groups efficiently with the use of DeviceMesh, we create the entire GPU topology in a 2x2 matrix, with each row representing the group of sharding ranks and each column representing the corresponding replica ranks,

```
create peer matrix
num_groups = global_world_size // sharding_group_size
for each group_rank in num_groups:
	peers = [num_groups * rank + group_rank for rank in range(sharding_group_size)]
	add peer to peer matrix

initalize DeviceMesh with two dimensions (shard, replicate)
slice DeviceMesh on shard for sharding process group
slide DeviceMesh on replicate for replica process group
```

With our DeviceMesh approach, should we want to change the topology or provide further flexibility in the future, we can easily extend our creation logic to any form of topologies and even extend for further dimensions of parallelism if needed.

## Performance of 2D parallel

Our rank partitioning strategy optimizes communication patterns by strategically placing model replica ranks for each shard within the same compute node. This architecture provides significant performance benefits for the weight synchronization operation. After the backward pass, we perform all-reduce operations to synchronize model weights—which is an expensive process given the large parameter counts we have to communicate and sync—with our setup of placing replicas on the same node we leverage intra node’s high-bandwidth over-relying on slower inter-node bandwidth.

The effect of this design choice on the other communication collectives generally improves the latencies. The improvement stems from two factors. 



1. By sharding the embedding tables over a reduced number of ranks and conducting communications for the model within the smaller group, we achieve a lower all-to-all latency.
2. With the replication in 2D parallel, our embedding lookup latency on a rank reduces, we can reduce the local batch size to 1/Nth of the equivalent global batch size, where N is the number of model replicas. 

A production model trace exemplifies these two factors, here we run the 2D parallel job on 1024 GPUs, with a sharding group size of 256 GPUs.

![State diagram](/assets/images/scaling-recommendation-2d-sparse-parallelism/fg2.png){:style="width:100%"}

***Figure 2: Comparing latencies between non 2D parallel and 2D parallel workloads***

There are two key levers users have to tune to maximize performance for their workloads: 



1. The size of the model sharding group relative to the global world size. The global world size divided by the sharding group size represents the number of model replicas we will have. 
    1. To maximize performance, users can look to scale up their model up to 8x, this scaling factor maintains the intra-host all reduce. 
        1. For further scaling, the all reduce would have to happen over inter host. From our experiments, we did not see an obvious performance regression and in fact note advantages of an inter host all reduce. We can change our sharding and replica topology to inter host all reduce, which can help us introduce fault tolerance strategies should a particular host go down.
2. Frequency of all reduce synchronization, DMPCollection comes with a sync() call, which can be tuned to be called every N training steps, performing a sort of local SGD training. With scale, reducing the frequency of synchronization can bring significant gains to performance.

## Future Work

Readers should note that 2D sparse parallel training differs from non-parallelized training because we synchronize the embedding table weights rather than the gradients. This approach is made possible by TorchRec's use of FBGEMM, which provides optimized kernels under the hood. One of FBGEMM's key optimizations is the fusion of the optimizer in the backward pass. Instead of fully materializing the embedding table gradients—which would consume significant memory—they are passed directly to the optimizer update. Attempting to materialize and synchronize these gradients would create substantial overhead, making that approach impractical.

Our exploration revealed that to achieve training results comparable to the baseline, we synchronize optimizer states on a delayed schedule, with the timing dependent on the number of sharding/replica groups (ie: for Adagrad we update the momentum behind by one sync step). This approach also enables users to implement local SGD or semi-synchronized training strategies, which can achieve convergence and potentially produce better loss curves than the baseline.

We thank you for reading our post! This is an exciting direction we have come across that we hope to develop further to maximize performance of recommendation systems and push the state of the art.

<style>
@media screen and (min-width: 768px) {
    article.pytorch-article ul, article.pytorch-article ol {
        padding-left: 3.5rem;
    }
}
ol {
  list-style-type: decimal; /* 1, 2, 3 */
}

ol ol {
  list-style-type: lower-alpha; /* a, b, c */
}

ol ol ol {
  list-style-type: lower-roman; /* i, ii, iii */
}


</style>