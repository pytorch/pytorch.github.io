---
layout: mobile
title: iOS
permalink: /mobile/ios/
background-class: mobile-background
body-class: mobile
order: 2
published: true
---

# iOS

To get started with PyTorch on iOS, we recommend exploring the following [HelloWorld](https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld).

## Quickstart with a Hello World Example

HelloWorld is a simple image classification application that demonstrates how to use PyTorch C++ libraries on iOS. The code is written in Swift and uses Objective-C as a bridge.

### Model Preparation

Let's start with model preparation. If you are familiar with PyTorch, you probably should already know how to train and save your model. In case you don't, we are going to use a pre-trained image classification model - [MobileNet v2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/), which is already packaged in [TorchVision](https://pytorch.org/docs/stable/torchvision/index.html). To install it, run the command below.

> We highly recommend following the [Pytorch Github page](https://github.com/pytorch/pytorch) to set up the Python development environment on your local machine.

```shell
pip install torchvision
```

Once we have TorchVision installed successfully, let's navigate to the HelloWorld folder and run `trace_model.py`. The script contains the code of tracing and saving a [torchscript model](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) that can be run on mobile devices.

```shell
python trace_model.py
```

If everything works well, we should have our model - `model.pt` generated in the `HelloWorld` folder. Now copy the model file to our application folder `HelloWorld/model`.

> To find out more details about TorchScript, please visit [tutorials on pytorch.org](https://pytorch.org/tutorials/advanced/cpp_export.html)

### Install LibTorch via Cocoapods

The PyTorch C++ library is available in [Cocoapods](https://cocoapods.org/), to integrate it to our project, simply run

```ruby
pod install
```

Now it's time to open the `HelloWorld.xcworkspace` in XCode, select an iOS simulator and launch it (cmd + R). If everything works well, we should see a wolf picture on the simulator screen along with the prediction result.

<img src="https://github.com/pytorch/ios-demo-app/blob/master/HelloWorld/screenshot.png?raw=true" width="50%">

### Code Walkthrough

In this part, we are going to walk through the code step by step.

#### Image Loading

Let's begin with image loading.

```swift
let image = UIImage(named: "image.jpg")!
imageView.image = image
let resizedImage = image.resized(to: CGSize(width: 224, height: 224))
guard var pixelBuffer = resizedImage.normalized() else {
    return
}
```

We first load the image from our bundle and resize it to 224x224. Then we call this `normalized()` category method to normalized the pixel buffer. Let's take a closer look at the code below.

```swift
var normalizedBuffer: [Float32] = [Float32](repeating: 0, count: w * h * 3)
// normalize the pixel buffer
// see https://pytorch.org/hub/pytorch_vision_resnet/ for more detail
for i in 0 ..< w * h {
    normalizedBuffer[i]             = (Float32(rawBytes[i * 4 + 0]) / 255.0 - 0.485) / 0.229 // R
    normalizedBuffer[w * h + i]     = (Float32(rawBytes[i * 4 + 1]) / 255.0 - 0.456) / 0.224 // G
    normalizedBuffer[w * h * 2 + i] = (Float32(rawBytes[i * 4 + 2]) / 255.0 - 0.406) / 0.225 // B
}
```

The code might look weird at first glance, but it’ll make sense once we understand our model. The input data is a 3-channel RGB image of shape (3 x H x W), where H and W are expected to be at least 224. The image has to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`.

####  TorchScript Module

Now that we have preprocessed our input data and we have a pre-trained TorchScript model, the next step is to use them to run predication. To do that, we'll first load our model into the application.

```swift
private lazy var module: TorchModule = {
    if let filePath = Bundle.main.path(forResource: "model", ofType: "pt"),
        let module = TorchModule(fileAtPath: filePath) {
        return module
    } else {
        fatalError("Can't find the model file!")
    }
}()
```
Note that the `TorchModule` Class is an Objective-C wrapper of `torch::jit::script::Module`.

```cpp
torch::jit::script::Module module = torch::jit::load(filePath.UTF8String);
```
Since Swift can not talk to C++ directly, we have to either use an Objective-C class as a bridge, or create a C wrapper for the C++ library. For demo purpose, we're going to wrap everything in this Objective-C class. However, we're working on providing the Swift/Objective-C API wrappers to PyTorch. Stay tuned!

#### Run Inference

Now it's time to run inference and get the results.

```swift
guard let outputs = module.predict(image: UnsafeMutableRawPointer(&pixelBuffer)) else {
    return
}
```
Again, the `predict` method is just an Objective-C wrapper. Under the hood, it calls the C++ `forward` function. Let's take a look at how it's implemented.

```cpp
at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat);
torch::autograd::AutoGradMode guard(false);
auto outputTensor = _impl.forward({tensor}).toTensor();
float* floatBuffer = outputTensor.data_ptr<float>();
```
The C++ function `torch::from_blob` will create an input tensor from the pixel buffer. Note that the shape of the tensor is `{1,3,224,224}` which represents `NxCxWxH` as we discussed in the above section.

```cpp
torch::autograd::AutoGradMode guard(false);
at::AutoNonVariableTypeMode non_var_type_mode(true);
```
The above two lines tells the PyTorch engine to do inference only. This is because by default, PyTorch has built-in support for doing auto-differentiation, which is also known as [autograd](https://pytorch.org/docs/stable/notes/autograd.html). Since we don't do training on mobile, we can just disable the autograd mode.

Finally, we can call this `forward` function to get the output tensor and convert it to a `float` buffer.

```cpp
auto outputTensor = _impl.forward({tensor}).toTensor();
float* floatBuffer = outputTensor.data_ptr<float>();
```

### Collect Results

The output tensor is a one-dimensional float array of shape 1x1000, where each value represents the confidence that a label is predicted from the image. The code below sorts the array and retrieves the top three results.

```swift
let zippedResults = zip(labels.indices, outputs)
let sortedResults = zippedResults.sorted { $0.1.floatValue > $1.1.floatValue }.prefix(3)
```

### PyTorch Demo App

For more complex use cases, we recommend to check out the [PyTorch demo application](https://github.com/pytorch/ios-demo-app). The demo app contains two showcases. A camera app that runs a quantized model to predict the images coming from device’s rear-facing camera in real time. And a text-based app that uses a text classififcation model to predict the topic from the input string.

## Build PyTorch iOS Libraries from Source

To track the latest updates for iOS, you can build the PyTorch iOS libraries from the source code.

```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```

> Make sure you have `cmake` and Python installed correctly on your local machine. We recommend following the [Pytorch Github page](https://github.com/pytorch/pytorch) to set up the Python development environment

### Build LibTorch for iOS Simulators

Open terminal and navigate to the PyTorch root directory. Run the following command

```
BUILD_PYTORCH_MOBILE=1 IOS_PLATFORM=SIMULATOR ./scripts/build_ios.sh
```
After the build succeeds, all static libraries and header files will be generated under `build_ios/install`

### Build LibTorch for arm64 Devices

Open terminal and navigate to the PyTorch root directory. Run the following command

```
BUILD_PYTORCH_MOBILE=1 IOS_ARCH=arm64 ./scripts/build_ios.sh
```
After the build succeeds, all static libraries and header files will be generated under `build_ios/install`

### XCode Setup

Open your project in XCode, copy all the static libraries as well as header files to your project. Navigate to the project settings, set the value **Header Search Paths** to the path of header files you just copied.

In the build settings, search for **other linker flags**.  Add a custom linker flag below

```
-force_load $(PROJECT_DIR)/${path-to-libtorch.a}
```
 Finally, disable bitcode for your target by selecting the Build Settings, searching for **Enable Bitcode**, and set the value to **No**.

### API Docs

Currently, the iOS framework uses the Pytorch C++ front-end APIs directly. The C++ document can be found [here](https://pytorch.org/cppdocs/). To learn more about it, we recommend exploring the [C++ front-end tutorials](https://pytorch.org/tutorials/advanced/cpp_frontend.html) on PyTorch webpage. In the meantime, we're working on providing the Swift/Objective-C API wrappers to PyTorch.


### Custom Build

Starting from 1.4.0, PyTorch supports custom build. You can now build the PyTorch library that only contains the operators needed by your model. To do that, follow the steps below

1\. Verify your PyTorch version is 1.4.0 or above. You can do that by checking the value of `torch.__version__`.

2\. To dump the operators in your model, say `MobileNetV2`, run the following lines of Python code:

```python
import torch, yaml
model = torch.jit.load('MobileNetV2.pt')
ops = torch.jit.export_opnames(model)
with open('MobileNetV2.yaml', 'w') as output:
    yaml.dump(ops, output)
```
In the snippet above, you first need to load the ScriptModule. Then, use `export_opnames` to return a list of operator names of the ScriptModule and its submodules. Lastly, save the result in a yaml file.

3\. To run the iOS build script locally with the prepared yaml list of operators, pass in the yaml file generate from the last step into the environment variable `SELECTED_OP_LIST`. Also in the arguments, specify `BUILD_PYTORCH_MOBILE=1` as well as the platform/architechture type. Take the arm64 build for example, the command should be:

```
SELECTED_OP_LIST=MobileNetV2.yaml BUILD_PYTORCH_MOBILE=1 IOS_ARCH=arm64 ./scripts/build_ios.sh
```
4\. After the build succeeds, you can integrate the result libraries to your project by following the [XCode Setup](#xcode-setup) section above.

5\. The last step is to add a single line of C++ code before running `forward`. This is because by default JIT will do some optimizations on operators (fusion for example), which might break the consistency with the ops we dumped from the model.

```cpp
torch::jit::GraphOptimizerEnabledGuard guard(false);
```

## Issues and Contribution

If you have any questions or want to contribute to PyTorch, please feel free to drop issues or open a pull request to get in touch.

<!-- Do not remove the below script -->

<script page-id="ios" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
