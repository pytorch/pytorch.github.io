---
layout: blog_detail
title: 'PyTorch 1.5 released, new and updated APIs including C++ frontend API parity with Python'
author: Team PyTorch
---


Today, we’re announcing the availability of PyTorch 1.5, along with new and updated libraries. This release includes several major new API additions and improvements. PyTorch now includes a significant update to the C++ frontend, ‘channels last’ memory format for computer vision models, and a stable release of the distributed RPC framework used for model-parallel training. The release also has new APIs for autograd for hessians and jacobians, and an API that allows the creation of Custom C++ Classes that was inspired by pybind.

You can find the detailed release notes [here](https://github.com/pytorch/pytorch/releases).

## C++ Frontend API (Stable)

The C++ frontend API is now at parity with Python, and the features overall have been moved to ‘stable’ (previously tagged as experimental). Some of the major highlights include:

* Now with ~100% coverage and docs for C++ torch::nn module/functional, users can easily translate their model from Python API to C++ API, making the model authoring experience much smoother.
* Optimizers in C++ had deviated from the Python equivalent: C++ optimizers can’t take parameter groups as input while the Python ones can. Additionally, step function implementations were not exactly the same. With the 1.5 release, C++ optimizers will always behave the same as the Python equivalent.
* The lack of tensor multi-dim indexing API in C++ is a well-known issue and had resulted in many posts in PyTorch Github issue tracker and forum. The previous workaround was to use a combination of `narrow` / `select` / `index_select` / `masked_select`, which was clunky and error-prone compared to the Python API’s elegant `tensor[:, 0, ..., mask]` syntax. With the 1.5 release, users can use `tensor.index({Slice(), 0, "...", mask})` to achieve the same purpose.

## ‘Channels last’ memory format for Computer Vision models (Experimental)

‘Channels last’ memory layout unlocks ability to use performance efficient convolution algorithms and hardware (NVIDIA’s Tensor Cores, FBGEMM, QNNPACK). Additionally, it is designed to automatically propagate through the operators, which allows easy switching between memory layouts.

Learn more [here](https://github.com/pytorch/pytorch/wiki/Writing-memory-format-aware-operators) on how to write memory format aware operators.

## Custom C++ Classes (Experimental)

This release adds a new API, `torch::class_`, for binding custom C++ classes into TorchScript and Python simultaneously. This API is almost identical in syntax to [pybind11](https://pybind11.readthedocs.io/en/stable/). It allows users to expose their C++ class and its methods to the TorchScript type system and runtime system such that they can instantiate and manipulate arbitrary C++ objects from TorchScript and Python. An example C++ binding:

```python
template <class T>
struct MyStackClass : torch::CustomClassHolder {
  std::vector<T> stack_;
  MyStackClass(std::vector<T> init) : stack_(std::move(init)) {}

  void push(T x) {
    stack_.push_back(x);
  }
  T pop() {
    auto val = stack_.back();
    stack_.pop_back();
    return val;
  }
};

static auto testStack =
  torch::class_<MyStackClass<std::string>>("myclasses", "MyStackClass")
      .def(torch::init<std::vector<std::string>>())
      .def("push", &MyStackClass<std::string>::push)
      .def("pop", &MyStackClass<std::string>::pop)
      .def("size", [](const c10::intrusive_ptr<MyStackClass>& self) {
        return self->stack_.size();
      });
```

 Which exposes a class you can use in Python and TorchScript like so:

```python
@torch.jit.script
def do_stacks(s : torch.classes.myclasses.MyStackClass):
    s2 = torch.classes.myclasses.MyStackClass(["hi", "mom"])
    print(s2.pop()) # "mom"
    s2.push("foobar")
    return s2 # ["hi", "foobar"]
```

You can try it out in the tutorial [here](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html).


## Distributed RPC framework APIs (Now Stable)

The Distributed [RPC framework](https://pytorch.org/docs/stable/rpc.html) was launched as experimental in the 1.4 release and the proposal is to mark Distributed RPC framework as stable and no longer experimental. This work involves a lot of enhancements and bug fixes to make the distributed RPC framework more reliable and robust overall, as well as adding a couple of new features, including profiling support, using TorchScript functions in RPC, and several enhancements for ease of use. Below is an overview of the various APIs within the framework:

### RPC API
The RPC API allows users to specify functions to run and objects to be instantiated on remote nodes. These functions are transparently recorded so that gradients can backpropagate through remote nodes using Distributed Autograd.

### Distributed Autograd
Distributed Autograd connects the autograd graph across several nodes and allows gradients to flow through during the backwards pass. Gradients are accumulated into a context (as opposed to the .grad field as with Autograd) and users must specify their model’s forward pass under a with `dist_autograd.context()` manager in order to ensure that all RPC communication is recorded properly. Currently, only FAST mode is implemented (see [here](https://pytorch.org/docs/stable/rpc/distributed_autograd.html#distributed-autograd-design) for the difference between FAST and SMART modes).

### Distributed Optimizer
The distributed optimizer creates RRefs to optimizers on each worker with parameters that require gradients, and then uses the RPC API to run the optimizer remotely. The user must collect all remote parameters and wrap them in an `RRef`, as this is required input to the distributed optimizer. The user must also specify the distributed autograd `context_id` so that the optimizer knows in which context to look for gradients.

Learn more about distributed RPC framework APIs [here](https://pytorch.org/docs/stable/rpc.html).

## New High level autograd API (Experimental)

PyTorch 1.5 brings new functions including jacobian, hessian, jvp, vjp, hvp and vhp to the `torch.autograd.functional` submodule. This feature builds on the current API and allows the user to easily perform these functions.

Detailed design discussion on GitHub can be found [here](https://github.com/pytorch/pytorch/issues/30632).

## Python 2 no longer supported

Starting PyTorch 1.5.0, we will no longer support Python 2, specifically version 2.7. Going forward support for Python will be limited to Python 3, specifically Python 3.5, 3.6, 3.7 and 3.8 (first enabled in PyTorch 1.4.0).


*We’d like to thank the entire PyTorch team and the community for all their contributions to this work.*

Cheers!

Team PyTorch
