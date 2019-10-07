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

To get started with PyTorch on iOS, we recommend exploring the following a HelloWorld example on Github [Hello Word](https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld). 

## Quickstart with a HelloWorld example

HelloWorld is a simple image classification application that demonstrates how to use PyTorch C++ libraries on iOS. The code is written in Swift and uses an Objective-C class as a bridging header.

Before we jump into details, we highly recommend following the Pytorch Github page to set up the Python development environment on your local machine. 

### Model preparation

Let's start with model preparation. If you are familiar with PyTorch, you probably should already know how to train and save your model. In case you don't, we are going to use a pre-trained image classification model(Resnet18), which is packaged in [TorchVision](https://pytorch.org/docs/stable/torchvision/index.html). To install TorchVision, run the command below.

```shell
pip install torchvision
```

Once we have TorchVision installed successfully, let's navigate to the HelloWorld folder and run a python script to generate our model. The `trace_model.py` contains the code of tracing and saving a [torchscript model](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) that can be run on mobile devices. Run the command below to get our model

```shell
python trace_model.py
```

If everything works well, we should have our model - `model.pt` generated in the same folder. Now copy the model file to our application folder `HelloWorld/model`.

### Install PyTorch C++ libraries via Cocoapods

The PyTorch C++ library is available in [Cocoapods](https://cocoapods.org/), to integrate it to our project, we can run 

```ruby
pod install
```
Now it's time to open the `HelloWorld.xcworkspace` in XCode, select an iOS simulator and hit the build and run button (cmd + R). If everything works well, we should see a wolf picture on the simulator screen along with the prediction result.

### Code Walkthrough

In this part, we are going to walk through the code step by step. The `ViewController.swift` contains most of the code.

#### Image loading

Let's begin with image loading.

```swift
let image = UIImage(named: "image.jpg")!
imageView.image = image
let resizedImage = image.resized(to: CGSize(width: 224, height: 224))
guard var pixelBuffer = resizedImage.normalized() else {
    return
}
```

We first load an image from the bundle and resize it to 224x224, which is the size of the input tensor. Then we call this `normalized()` category method on UIImage to get normalized pixel data from the image. Let's take a look at the code below.

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
The input data of our model is a 3-channel RGB image of shape (3 x H x W), where H and W are expected to be at least 224. The image have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

#### Init JIT interpreter

Now that we have preprocessed our input data and we have a pre-trained TorchScript model, the next step is to use the model and the data to run the predication. To do that, we'll first load our model into the application.

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
The TorchModule Class is an Objective-C wrapper for the C++ class `torch::jit::script::Module`. 

```cpp
torch::jit::script::Module module = torch::jit::load(filePath.UTF8String);
```

#### Run Inference

Now it's time to run the inference and get the result. We pass in the pixel buffer object as a raw pointer to the `predict` method and get the result from it.

```swift
guard let outputs = module.predict(image: UnsafeMutableRawPointer(&pixelBuffer)) else {
    return
}
```
Again, the `predict` method on the `module` is an Objective-C method. Under the hood, it calls the C++ version of predict which is `forward`

```cpp
 auto outputTensor = _impl.forward({inputTensor}).toTensor();
```

### Collect results

The output tensor is a one-dimensional float array of shape 1x1000, where each value represents the confidence that a label is predicted from the image. The code below sorts the array and retrieves the top three results.

```swift
let zippedResults = zip(labels.indices, outputs)
let sortedResults = zippedResults.sorted { $0.1.floatValue > $1.1.floatValue }.prefix(3)
```

### PyTorch demo app

For more complex use cases, we recommend to check out the PyTorch demo application. 

The demo app contains two showcases. A camera app that runs a quantized model to predict the images coming from device’s rear-facing camera in real time. And a text-based app that uses a self-trained NLP model to predict the topic from the input string.

## Build PyTorch iOS libraries from source

To track the latest progress on mobile, we can always build the PyTorch iOS libraries from the source. Follow the steps below.

### Setup local Python development environment

Follow the PyTorch Github page to set up the Python environment. Make sure you have `cmake` and Python installed correctly on your local machine.

### Build LibTorch.a for iOS simulator

Open terminal and navigate to the PyTorch root directory. Run the following command

```
BUILD_PYTORCH_MOBILE=1 IOS_PLATFORM=SIMULATOR ./scripts/build_ios.sh
```
After the build succeed, all static libraries and header files will be generated under `build_ios/install`

### Build LibTorch.a for arm64 devices

Open terminal and navigate to the PyTorch root directory. Run the following command

```
BUILD_PYTORCH_MOBILE=1 IOS_ARCH=arm64 ./scripts/build_ios.sh
```
After the build succeed, all static libraries and header files will be generated under `build_ios/install`

### XCode setup

Open your project in XCode, copy all the static libraries as well as header files to your project. Navigate to the project settings, set the value **Header Search Paths** to the path of header files you just copied in the first step.

In the build settings, search for **other linker flags**.  Add a custom linker flag below 

```
-force_load $(PROJECT_DIR)/path-to-libtorch.a
```
 Finally, disable bitcode for your target by selecting the Build Settings, searching for **Enable Bitcode**, and set the value to **No**.

## API Docs

Currently, the iOS framework uses raw Pytorch C++ APIs directly. The C++ document can be found here https://pytorch.org/cppdocs/. To learn how to use them, we recommend exploring the [C++ front-end tutorials](https://pytorch.org/tutorials/advanced/cpp_frontend.html) on PyTorch webpage. In the meantime, we're working on providing the Swift/Objective-C API wrappers to PyTorch.

## Issues and Contribution

If you have any questions or want to contribute to PyTorch, please feel free to drop issues or open a pull request to get in touch.

<!-- Do not remove the below script -->

<script page-id="ios" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
