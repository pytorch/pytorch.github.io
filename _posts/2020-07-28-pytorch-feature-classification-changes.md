---
layout: blog_detail
title: 'PyTorch feature classification changes'
author: Team PyTorch
---

Traditionally features in PyTorch were classified as either stable or experimental with an implicit third option of testing bleeding edge features by building master or through installing nightly builds (available via prebuilt whls). This has, in a few cases, caused some confusion around the level of readiness, commitment to the feature and backward compatibility that can be expected from a user perspective. Moving forward, we’d like to better classify the 3 types of features as well as define explicitly here what each mean from a user perspective.

# New Feature Designations

We will continue to have three designations for features but, as mentioned, with a few changes: Stable, Beta (previously Experimental) and Prototype (previously Nightlies). Below is a brief description of each and a comment on the backward compatibility expected:

## Stable
Nothing changes here. A stable feature means that the user value-add is or has been proven, the API isn’t expected to change, the feature is performant and all documentation exists to support end user adoption.

*Level of commitment*: We expect to maintain these features long term and generally there should be no major performance limitations, gaps in documentation and we also expect to maintain backwards compatibility (although breaking changes can happen and notice will be given one release ahead of time).

## Beta
We previously called these features ‘Experimental’ and we found that this created confusion amongst some of the users. In the case of a Beta level features, the value add, similar to a Stable feature, has been proven (e.g. pruning is a commonly used technique for reducing the number of parameters in NN models, independent of the implementation details of our particular choices) and the feature generally works and is documented. This feature is tagged as Beta because the API may change based on user feedback, because the performance needs to improve or because coverage across operators is not yet complete.

*Level of commitment*: We are committing to seeing the feature through to the Stable classification. We are however not committing to Backwards Compatibility. Users can depend on us providing a solution for problems in this area going forward, but the APIs and performance characteristics of this feature may change.

<div class="text-center">
  <img src="{{ site.url }}/assets/images/install-matrix.png" width="100%">
</div>

## Prototype
Previously these were features that were known about by developers who paid close attention to RFCs and to features that land in master. These features are part of the release and are available as part of binary distributions like PyPI or Conda. We would like to get high bandwidth partner feedback ahead of a real release in order to gauge utility and any changes we need to make to the UX. For each prototype feature, a pointer to draft docs or other instructions will be provided.

*Level of commitment*: We are committing to gathering high bandwidth feedback only. Based on this feedback and potential further engagement between community members, we as a community will decide if we want to upgrade the level of commitment or to fail fast. Additionally, while some of these features might be more speculative (e.g. new Frontend APIs), others have obvious utility (e.g. model optimization) but may be in a state where gathering feedback outside of high bandwidth channels is not practical, e.g. the feature may be in an earlier state, may be moving fast (PRs are landing too quickly to catch a major release) and/or generally active development is underway.

# What changes for current features?

First and foremost, you can find these designations on [pytorch.org/docs](http://pytorch.org/docs). We will also be linking any early stage features here for clarity.

Additionally, the following features will be reclassified under this new rubric:

1. [High Level Autograd APIs](https://pytorch.org/docs/stable/autograd.html#functional-higher-level-api): Beta (was Experimental)
2. [Eager Mode Quantization](https://pytorch.org/docs/stable/quantization.html): Beta (was Experimental)
3. [Named Tensors](https://pytorch.org/docs/stable/named_tensor.html): Prototype (was Experimental)
4. [TorchScript/RPC](https://pytorch.org/docs/stable/rpc.html#rpc): Prototype (was Experimental)
5. [Channels Last Memory Layout](https://pytorch.org/docs/stable/tensor_attributes.html#torch-memory-format): Beta (was Experimental)
6. [Custom C++ Classes](https://pytorch.org/docs/stable/jit.html?highlight=experimental): Beta (was Experimental)
7. [PyTorch Mobile](https://pytorch.org/mobile/home/): Beta (was Experimental)
8. [Java Bindings](https://pytorch.org/docs/stable/index.html): Beta (was Experimental)
9. [Torch.Sparse](https://pytorch.org/docs/stable/sparse.html?highlight=experimental#): Beta (was Experimental)


Cheers,

Joe, Greg, Woo & Jessica
