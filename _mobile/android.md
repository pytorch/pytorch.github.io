---
layout: mobile
title: Android
permalink: /mobile/android/
background-class: mobile-background
body-class: mobile
order: 3
published: true
---

# Android

## Quickstart with a HelloWorld Example

[HelloWorld](https://github.com/pytorch/android-demo-app/tree/master/HelloWorldApp) is a simple image classification application that demonstrates how to use PyTorch Android API.
This application runs TorchScript serialized TorchVision pretrained resnet18 model on static image which is packaged inside the app as android asset.

#### 1. Model Preparation

Let’s start with model preparation. If you are familiar with PyTorch, you probably should already know how to train and save your model. In case you don’t, we are going to use a pre-trained image classification model(Resnet18), which is packaged in [TorchVision](https://pytorch.org/docs/stable/torchvision/index.html).
To install it, run the command below:
```
pip install torchvision
```

To serialize the model you can use python [script](https://github.com/pytorch/android-demo-app/blob/master/HelloWorldApp/trace_model.py) in the root folder of HelloWorld app:
```
import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("app/src/main/assets/model.pt")
```
If everything works well, we should have our model - `model.pt` generated in the assets folder of android application. 
That will be packaged inside android application as `asset` and can be used on the device.

More details about TorchScript you can find in [tutorials on pytorch.org](https://pytorch.org/docs/stable/jit.html) 

#### 2. Cloning from github
```
git clone https://github.com/pytorch/android-demo-app.git
cd HelloWorldApp
```
If [Android SDK]() and [Android NDK]() are already installed you can install this application to the connected android device or emulator with:
```
./gradlew installDebug
```

We recommend you to open this project in [Android Studio](https://developer.android.com/studio),
in that case you will be able to install Android NDK and Android SDK using Android Studio UI.

#### 3. Gradle dependencies

Pytorch android is added to the HelloWorld as [gradle dependencies](https://github.com/pytorch/android-demo-app/blob/master/HelloWorldApp/app/build.gradle#L28-L29) in build.gradle:

```
repositories {
    jcenter()
}

dependencies {
    implementation 'org.pytorch:pytorch_android:1.3.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.3.0'
}
```
Where `org.pytorch:pytorch_android` is the main dependency with PyTorch Android API, including libtorch native library for all 4 android abis (armeabi-v7a, arm64-v8a, x86, x86_64).
Further in this doc you can find how to rebuild it only for specific list of android abis. 

`org.pytorch:pytorch_android_torchvision` - additional library with utility functions for converting `android.media.Image` and `android.graphics.Bitmap` to tensors.

#### 4. Reading image from Android Asset

All the logic happens in [`org.pytorch.helloworld.MainActivity`](https://github.com/pytorch/android-demo-app/blob/master/HelloWorldApp/app/src/main/java/org/pytorch/helloworld/MainActivity.java#L31-L69).
As a first step we read `image.jpg` to `android.graphics.Bitmap` using the standard Android API. 
```
Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open("image.jpg"));
```

#### 5. Loading TorchScript Module
```
Module module = Module.load(assetFilePath(this, "model.pt"));
```
`org.pytorch.Module` represents `torch::jit::script::Module` that can be loaded with `load` method specifying file path to the serialized to file model.

#### 6. Preparing Input
```
Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
```
`org.pytorch.torchvision.TensorImageUtils` is part of `org.pytorch:pytorch_android_torchvision` library.
The `TensorImageUtils#bitmapToFloat32Tensor` method creates tensors in the [torchvision format](https://pytorch.org/docs/stable/torchvision/models.html) using `android.graphics.Bitmap` as a source.

> All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. 
> The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`

`inputTensor`'s shape is `1x3xHxW`, where `H` and `W` are bitmap height and width appropriately. 

#### 7. Run Inference
 
```
Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
float[] scores = outputTensor.getDataAsFloatArray();
```

`org.pytorch.Module.forward` method runs loaded module's `forward` method and gets result as `org.pytorch.Tensor` outputTensor with shape `1x1000`.

#### 8. Processing results
Its content is retrieved using `org.pytorch.Tensor.getDataAsFloatArray()` method that returns java array of floats with scores for every image net class.
 
After that we just find index with maximum score and retrieve predicted class name from `ImageNetClasses.IMAGENET_CLASSES` array that contains all ImageNet classes.

```
float maxScore = -Float.MAX_VALUE;
int maxScoreIdx = -1;
for (int i = 0; i < scores.length; i++) {
  if (scores[i] > maxScore) {
    maxScore = scores[i];
    maxScoreIdx = i;
  }
}
String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];
```
 
In the following sections you can find detailed explanations of PyTorch Android API, code walk through for a bigger [demo application](https://github.com/pytorch/android-demo-app/tree/master/PyTorchDemoApp), 
implementation details of the API, how to customize and build it from source.

## PyTorch Demo Application

We have also created another more complex PyTorch Android demo application that does image classification from camera output and text classification in the [same github repo](https://github.com/pytorch/android-demo-app/tree/master/PyTorchDemoApp).

To get device camera output it uses [Android CameraX API](https://developer.android.com/training/camerax
).
All the logic that works with CameraX is separated to [`org.pytorch.demo.vision.AbstractCameraXActivity`](https://github.com/pytorch/android-demo-app/blob/master/PyTorchDemoApp/app/src/main/java/org/pytorch/demo/vision/AbstractCameraXActivity.java) class.


```
void setupCameraX() {
    final PreviewConfig previewConfig = new PreviewConfig.Builder().build();
    final Preview preview = new Preview(previewConfig);
    preview.setOnPreviewOutputUpdateListener(output -> mTextureView.setSurfaceTexture(output.getSurfaceTexture()));

    final ImageAnalysisConfig imageAnalysisConfig =
        new ImageAnalysisConfig.Builder()
            .setTargetResolution(new Size(224, 224))
            .setCallbackHandler(mBackgroundHandler)
            .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            .build();
    final ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
    imageAnalysis.setAnalyzer(
        (image, rotationDegrees) -> {
          analyzeImage(image, rotationDegrees);
        });

    CameraX.bindToLifecycle(this, preview, imageAnalysis);
  }

  void analyzeImage(android.media.Image, int rotationDegrees)
```

Where the `analyzeImage` method process the camera output, `android.media.Image`.

It uses the aforementioned [`TensorImageUtils.imageYUV420CenterCropToFloat32Tensor`](https://github.com/pytorch/pytorch/blob/master/android/pytorch_android_torchvision/src/main/java/org/pytorch/torchvision/TensorImageUtils.java#L90) method to convert `android.media.Image` in `YUV420` format to input tensor.

After getting predicted scores from the model it finds top K classes with the highest scores and shows on the UI.

## Building PyTorch Android from Source

In some cases you might want to use a local build of pytorch android, for example you may build custom libtorch binary with another set of operators or to make local changes.

For this you can use `./scripts/build_pytorch_android.sh` script.
```
git clone https://github.com/pytorch/pytorch.git
cd pytorch
sh ./scripts/build_pytorch_android.sh
```

The workflow contains several steps:

1\. Build libtorch for android for all 4 android abis (armeabi-v7a, arm64-v8a, x86, x86_64)

2\. Create symbolic links to the results of those builds:
`android/pytorch_android/src/main/jniLibs/${abi}` to the directory with output libraries
`android/pytorch_android/src/main/cpp/libtorch_include/${abi}` to the directory with headers. These directories are used to build `libpytorch.so` library that will be loaded on android device.
 
3\. And finally run `gradle` in `android/pytorch_android` directory with task `assembleRelease`

Script requires that Android SDK, Android NDK and gradle are installed.
They are specified as environment variables:

`ANDROID_HOME` - path to [Android SDK](https://developer.android.com/studio/command-line/sdkmanager.html)

`ANDROID_NDK` - path to [Android NDK](https://developer.android.com/studio/projects/install-ndk)

`GRADLE_HOME` - path to [gradle](https://gradle.org/releases/)


After successful build you should see the result as aar file:

```
$ find pytorch_android/build/ -type f -name *aar
pytorch_android/build/outputs/aar/pytorch_android.aar
pytorch_android_torchvision/build/outputs/aar/pytorch_android.aar
libs/fbjni_local/build/outputs/aar/pytorch_android_fbjni.aar
```

It can be used directly in android projects, as a gradle dependency:
```
allprojects {
    repositories {
        flatDir {
            dirs 'libs'
        }
    }
}

android {
    ...
    packagingOptions {
        pickFirst "**/libfbjni.so"
    }
    ...
}

dependencies {
    implementation(name:'pytorch_android', ext:'aar')
    implementation(name:'pytorch_android_torchvision', ext:'aar')
    implementation(name:'pytorch_android_fbjni', ext:'aar')
}
```

At the moment for the case of using aar files directly we need additional configuration due to packaging specific (`libfbjni.so` is packaged in both `pytorch_android_fbjni.aar` and `pytorch_android.aar`).
```
packagingOptions {
    pickFirst "**/libfbjni.so"
}
```

## API Details

Main part of java API includes 3 classes:
```
org.pytorch.Module
org.pytorch.IValue
org.pytorch.Tensor
```

If the reader is familiar with PyTorch Python API, we can think of `org.pytorch.Tensor` representing `torch.tensor`, `org.pytorch.Module` representing `torch.Module`, and `org.pytorch.IValue` representing the value of the TorchScript variable, supporting all its [types](https://pytorch.org/docs/stable/jit.html#types).

### [`org.pytorch.Tensor`]((https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/java/org/pytorch/Tensor.java))

Tensor supports dtypes `uint8, int8, float32, int32, float64, int64`.
Tensor holds data in DirectByteBuffer of proper type with native bit order. 

To create a Tensor user can use one of the factory methods:
```
Tensor fromBlobUnsigned(ByteBuffer data, long[] shape)
Tensor fromBlobUnsigned(byte[] data, long[] shape)

Tensor fromBlob(ByteBuffer data, long[] shape)
Tensor fromBlob(byte[] data, long[] shape)

Tensor fromBlob(FloatBuffer data, long[] shape)
Tensor fromBlob(float[] data, long[] shape)

Tensor fromBlob(IntBuffer data, long[] shape)
Tensor fromBlob(int[] data, long[] shape)

Tensor fromBlob(DoubleBuffer data, long[] shape)
Tensor fromBlob(double[] data, long[] shape)

Tensor fromBlob(LongBuffer data, long[] shape)
Tensor fromBlob(long[] data, long[] shape)
```
Where the first parameter `long[] shape` is shape of the Tensor as array of longs.

Content of the Tensor can be provided either as (a) java array  or (b) as `java.nio.DirectByteBuffer` of proper type with native bit order.

In case of (a) proper `DirectByteBuffer` will be created internally. (b) case has an advantage that user can keep the reference to DirectByteBuffer and change its content in future for the next run, avoiding allocation of DirectByteBuffer for repeated runs.

Java’s primitive type byte is signed and java does not have unsigned 8 bit type. For dtype=uint8 java API uses java primitive `byte` that will be reinterpreted as uint8 on native side. On java side unsigned value of byte can be read as `byte & 0xFF`.

#### Tensor Content Layout

Tensor content is represented as a one dimensional array (buffer),
where the first element has all zero indexes T\[0, ... 0\].

Lets assume tensor shape is {d<sub>0</sub>, ... d<sub>n-1</sub>} and d<sub>n-1</sub> > 0.

The second element will be T\[0, ... 1\] and the last one T\[d<sub>0</sub>-1, ... d<sub>n-1</sub> - 1\]

Tensor has methods to check its dtype:
```
int dtype()
```
That returns one of the `DType` enum element:
```
DType.UINT8
DType.INT8
DType.INT32
DType.FLOAT32
DType.INT64
DType.FLOAT64
```

The data of Tensor can be read as a java array:
```
byte[] getDataAsUnsignedByteArray()
byte[] getDataAsByteArray()
int[] getDataAsIntArray()
long[] getDataAsLongArray()
float[] getDataAsFloatArray()
double[] getDataAsDoubleArray() 
```
These methods throw `IllegalStateException` if called for inappropriate dtype.

### [`org.pytorch.IValue`](https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/java/org/pytorch/IValue.java)

IValue represents a TorchScript variable that can be one of the supported (by TorchScript) [types](https://pytorch.org/docs/stable/jit.html#types). 
`IValue` is a tagged union. For every supported type it has a factory method, method to check the type and a getter method to retrieve a value.
Getters throw `IllegalStateException` if called for inappropriate type.

### [`org.pytorch.Module`](https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/java/org/pytorch/Module.java)

Module is a wrapper of torch.jit.ScriptModule (`torch::jit::script::Module` in PyTorch C++ API) which can be constructed with the factory method `load` providing absolute path to the file with serialized TorchScript.
```
IValue IValue.runMethod(String methodName, IValue... inputs)
```
for running a particular method of the script module.
```
IValue IValue.forward(IValue... inputs)
```
Shortcut to run 'forward' method.

```
IValue IValue.destroy()
```
Explicitly destructs native (C++) part of the Module, `torch::jit::script::Module`.

As fbjni library destructs native part automatically when current `org.pytorch.Module` instance will be collected by Java GC, the instance will not leak if this method is not called, but timing of deletion and the thread will be at the whim of the Java GC. If you want to control the thread and timing of the destructor, you should call this method explicitly.

<!-- Do not remove the below script -->

<script page-id="android" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
