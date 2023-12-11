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

Let’s start with model preparation. If you are familiar with PyTorch, you probably should already know how to train and save your model. In case you don’t, we are going to use a pre-trained image classification model ([MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)).
To install it, run the command below:
```
pip install torchvision
```

To serialize the model you can use python [script](https://github.com/pytorch/android-demo-app/blob/master/HelloWorldApp/trace_model.py) in the root folder of HelloWorld app:
```
import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("app/src/main/assets/model.ptl")

```
If everything works well, we should have our model - `model.ptl` generated in the assets folder of android application.
That will be packaged inside android application as `asset` and can be used on the device.

More details about TorchScript you can find in [tutorials on pytorch.org](https://pytorch.org/docs/stable/jit.html)

#### 2. Cloning from github
```
git clone https://github.com/pytorch/android-demo-app.git
cd HelloWorldApp
```
If [Android SDK](https://developer.android.com/studio/index.html#command-tools) and [Android NDK](https://developer.android.com/ndk/downloads) are already installed you can install this application to the connected android device or emulator with:
```
./gradlew installDebug
```

We recommend you to open this project in [Android Studio 3.5.1+](https://developer.android.com/studio). At the moment PyTorch Android and demo applications use [android gradle plugin of version 3.5.0](https://developer.android.com/studio/releases/gradle-plugin#3-5-0), which is supported only by Android Studio version 3.5.1 and higher.
Using Android Studio you will be able to install Android NDK and Android SDK with Android Studio UI.

#### 3. Gradle dependencies

Pytorch android is added to the HelloWorld as [gradle dependencies](https://github.com/pytorch/android-demo-app/blob/master/HelloWorldApp/app/build.gradle#L28-L29) in build.gradle:

```
repositories {
    jcenter()
}

dependencies {
    implementation 'org.pytorch:pytorch_android_lite:1.9.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.9.0'
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

#### 5. Loading Mobile Module
```
Module module = Module.load(assetFilePath(this, "model.ptl"));
```
`org.pytorch.Module` represents `torch::jit::mobile::Module` that can be loaded with `load` method specifying file path to the serialized to file model.

#### 6. Preparing Input
```
Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
```
`org.pytorch.torchvision.TensorImageUtils` is part of `org.pytorch:pytorch_android_torchvision` library.
The `TensorImageUtils#bitmapToFloat32Tensor` method creates tensors in the [torchvision format](https://pytorch.org/vision/stable/models.html) using `android.graphics.Bitmap` as a source.

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

#### Language Processing Example

Another example is natural language processing, based on an LSTM model, trained on a reddit comments dataset.
The logic happens in [`TextClassificattionActivity`](https://github.com/pytorch/android-demo-app/blob/master/PyTorchDemoApp/app/src/main/java/org/pytorch/demo/nlp/TextClassificationActivity.java).

Result class names are packaged inside the TorchScript model and initialized just after initial module initialization.
The module has a `get_classes` method that returns `List[str]`, which can be called using method `Module.runMethod(methodName)`:
```
    mModule = Module.load(moduleFileAbsoluteFilePath);
    IValue getClassesOutput = mModule.runMethod("get_classes");
```
The returned `IValue` can be converted to java array of `IValue` using `IValue.toList()` and processed to an array of strings using `IValue.toStr()`:
```
    IValue[] classesListIValue = getClassesOutput.toList();
    String[] moduleClasses = new String[classesListIValue.length];
    int i = 0;
    for (IValue iv : classesListIValue) {
      moduleClasses[i++] = iv.toStr();
    }
```

Entered text is converted to java array of bytes with `UTF-8` encoding. `Tensor.fromBlobUnsigned` creates tensor of `dtype=uint8` from that array of bytes.
```
    byte[] bytes = text.getBytes(Charset.forName("UTF-8"));
    final long[] shape = new long[]{1, bytes.length};
    final Tensor inputTensor = Tensor.fromBlobUnsigned(bytes, shape);
```

Running inference of the model is similar to previous examples:
```
Tensor outputTensor = mModule.forward(IValue.from(inputTensor)).toTensor()
```

After that, the code processes the output, finding classes with the highest scores.

## More PyTorch Android Demo Apps

### D2go

[D2Go](https://github.com/pytorch/android-demo-app/tree/master/D2Go) demonstrates a Python script that creates the much lighter and much faster Facebook [D2Go](https://github.com/facebookresearch/d2go) model that is powered by PyTorch 1.8, torchvision 0.9, and Detectron2 with built-in SOTA networks for mobile, and an Android app that uses it to detect objects from pictures in your photos, taken with camera, or with live camera. This demo app also shows how to use the native pre-built torchvision-ops library.

### Image Segmentation

[Image Segmentation](https://github.com/pytorch/android-demo-app/tree/master/ImageSegmentation) demonstrates a Python script that converts the PyTorch [DeepLabV3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) model and an Android app that uses the model to segment images.

### Object Detection

[Object Detection](https://github.com/pytorch/android-demo-app/tree/master/ObjectDetection) demonstrates how to convert the popular [YOLOv5](https://pytorch.org/hub/ultralytics_yolov5/) model and use it in an Android app that detects objects from pictures in your photos, taken with camera, or with live camera.

### Neural Machine Translation

[Neural Machine Translation](https://github.com/pytorch/android-demo-app/tree/master/Seq2SeqNMT) demonstrates how to convert a sequence-to-sequence neural machine translation model trained with the code in the [PyTorch NMT tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) and use the model in an Android app to do French-English translation.

### Question Answering

[Question Answering](https://github.com/pytorch/android-demo-app/tree/master/QuestionAnswering) demonstrates how to convert a powerful transformer QA model and use the model in an Android app to answer questions about PyTorch Mobile and more.

### Vision Transformer

[Vision Transformer](https://github.com/pytorch/android-demo-app/tree/master/ViT4MNIST) demonstrates how to use Facebook's latest Vision Transformer [DeiT](https://github.com/facebookresearch/deit) model to do image classification, and how convert another Vision Transformer model and use it in an Android app to perform handwritten digit recognition.

### Speech recognition

[Speech Recognition](https://github.com/pytorch/android-demo-app/tree/master/SpeechRecognition) demonstrates how to convert Facebook AI's wav2vec 2.0, one of the leading models in speech recognition, to TorchScript and how to use the scripted model in an Android app to perform speech recognition.

### Video Classification

[TorchVideo](https://github.com/pytorch/android-demo-app/tree/master/TorchVideo) demonstrates how to use a pre-trained video classification model, available at the newly released [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo), on Android to see video classification results, updated per second while the video plays, on tested videos, videos from the Photos library, or even real-time videos.


## PyTorch Android Tutorial and Recipes

### [Image Segmentation DeepLabV3 on Android](https://pytorch.org/tutorials/beginner/deeplabv3_on_android.html)

A comprehensive step-by-step tutorial on how to prepare and run the PyTorch DeepLabV3 image segmentation model on Android.

### [PyTorch Mobile Performance Recipes](https://pytorch.org/tutorials/recipes/mobile_perf.html)

List of recipes for performance optimizations for using PyTorch on Mobile.

### [Making Android Native Application That Uses PyTorch Android Prebuilt Libraries](https://pytorch.org/tutorials/recipes/android_native_app_with_custom_op.html)

Learn how to make Android application from the scratch that uses LibTorch C++ API and uses TorchScript model with custom C++ operator.

### [Fuse Modules recipe](https://pytorch.org/tutorials/recipes/fuse.html)

Learn how to fuse a list of PyTorch modules into a single module to reduce the model size before quantization.

### [Quantization for Mobile Recipe](https://pytorch.org/tutorials/recipes/quantization.html)

Learn how to reduce the model size and make it run faster without losing much on accuracy.

### [Script and Optimize for Mobile](https://pytorch.org/tutorials/recipes/script_optimized.html)

Learn how to convert the model to TorchScipt and (optional) optimize it for mobile apps.

### [Model Preparation for Android Recipe](https://pytorch.org/tutorials/recipes/model_preparation_android.html)

Learn how to add the model in an Android project and use the PyTorch library for Android.

## Building PyTorch Android from Source

In some cases you might want to use a local build of PyTorch android, for example you may build custom LibTorch binary with another set of operators or to make local changes, or try out the latest PyTorch code.

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
`android/pytorch_android/src/main/cpp/libtorch_include/${abi}` to the directory with headers. These directories are used to build `libpytorch_jni.so` library, as part of the `pytorch_android-release.aar` bundle, that will be loaded on android device.

3\. And finally run `gradle` in `android/pytorch_android` directory with task `assembleRelease`

Script requires that Android SDK, Android NDK, Java SDK, and gradle are installed.
They are specified as environment variables:

`ANDROID_HOME` - path to [Android SDK](https://developer.android.com/studio/command-line/sdkmanager.html)

`ANDROID_NDK` - path to [Android NDK](https://developer.android.com/studio/projects/install-ndk). It's recommended to use NDK 21.x.

`GRADLE_HOME` - path to [gradle](https://gradle.org/releases/)

`JAVA_HOME` - path to [JAVA JDK](https://www.oracle.com/java/technologies/javase-downloads.html#javasejdk)


After successful build, you should see the result as aar file:

```
$ find android -type f -name *aar
android/pytorch_android/build/outputs/aar/pytorch_android-release.aar
android/pytorch_android_torchvision/build/outputs/aar/pytorch_android_torchvision-release.aar
```

## Using the PyTorch Android Libraries Built from Source or Nightly

First add the two aar files built above, or downloaded from the nightly built PyTorch Android repos at [here](https://oss.sonatype.org/#nexus-search;quick~pytorch_android) and [here](https://oss.sonatype.org/#nexus-search;quick~torchvision_android), to the Android project's `lib` folder, then add in the project's app `build.gradle` file:
```
allprojects {
    repositories {
        flatDir {
            dirs 'libs'
        }
    }
}

dependencies {

    // if using the libraries built from source
    implementation(name:'pytorch_android-release', ext:'aar')
    implementation(name:'pytorch_android_torchvision-release', ext:'aar')

    // if using the nightly built libraries downloaded above, for example the 1.8.0-snapshot on Jan. 21, 2021
    // implementation(name:'pytorch_android-1.8.0-20210121.092759-172', ext:'aar')
    // implementation(name:'pytorch_android_torchvision-1.8.0-20210121.092817-173', ext:'aar')

    ...
    implementation 'com.android.support:appcompat-v7:28.0.0'
    implementation 'com.facebook.fbjni:fbjni-java-only:0.0.3'
}
```

Also we have to add all transitive dependencies of our aars. As `pytorch_android` depends on `com.android.support:appcompat-v7:28.0.0` or `androidx.appcompat:appcompat:1.2.0`, we need to one of them. (In case of using maven dependencies they are added automatically from `pom.xml`).

## Using the Nightly PyTorch Android Libraries

Other than using the aar files built from source or downloaded from the links in the previous section, you can also use the nightly built Android PyTorch and TorchVision libraries by adding in your app `build.gradle` file the maven url and the nightly libraries implementation as follows:

```
repositories {
    maven {
        url "https://oss.sonatype.org/content/repositories/snapshots"
    }
}

dependencies {
    ...
    implementation 'org.pytorch:pytorch_android:1.8.0-SNAPSHOT'
    implementation 'org.pytorch:pytorch_android_torchvision:1.8.0-SNAPSHOT'
}
```

This is the easiest way to try out the latest PyTorch code and the Android libraries, if you do not need to make any local changes. But be aware you may need to build the model used on mobile in the latest PyTorch - using either the latest PyTorch code or a quick nightly install with commands like `pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html` - to avoid possible model version mismatch errors when running the model on mobile.

## Custom Build

To reduce the size of binaries you can do custom build of PyTorch Android with only set of operators required by your model.
This includes two steps: preparing the list of operators from your model, rebuilding pytorch android with specified list.

1\. Verify your PyTorch version is 1.4.0 or above. You can do that by checking the value of `torch.__version__`.

2\. Preparation of the list of operators

List of operators of your serialized torchscript model can be prepared in yaml format using python api function `torch.jit.export_opnames()`.
To dump the operators in your model, say `MobileNetV2`, run the following lines of Python code:
```
# Dump list of operators used by MobileNetV2:
import torch, yaml
model = torch.jit.load('MobileNetV2.pt')
ops = torch.jit.export_opnames(model)
with open('MobileNetV2.yaml', 'w') as output:
    yaml.dump(ops, output)
```
3\. Building PyTorch Android with prepared operators list.

To build PyTorch Android with the prepared yaml list of operators, specify it in the environment variable `SELECTED_OP_LIST`. Also in the arguments, specify which Android ABIs it should build; by default it builds all 4 Android ABIs.

```
# Build PyTorch Android library customized for MobileNetV2:
SELECTED_OP_LIST=MobileNetV2.yaml scripts/build_pytorch_android.sh arm64-v8a
```

After successful build you can integrate the result aar files to your android gradle project, following the steps from previous section of this tutorial (Building PyTorch Android from Source).

## Use PyTorch JIT interpreter

PyTorch JIT interpreter is the default interpreter before 1.9 (a version of our PyTorch interpreter that is not as size-efficient). It will still be supported in 1.9, and can be used via `build.gradle`:
```
repositories {
    jcenter()
}

dependencies {
    implementation 'org.pytorch:pytorch_android:1.9.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.9.0'
}
```


## Android Tutorials

Watch the following [video](https://youtu.be/5Lxuu16_28o) as PyTorch Partner Engineer Brad Heintz walks through steps for setting up the PyTorch Runtime for Android projects:

[![PyTorch Mobile Runtime for Android](https://i.ytimg.com/vi/O_2KBhkIvnc/maxresdefault.jpg){:height="75%" width="75%"}](https://youtu.be/5Lxuu16_28o "PyTorch Mobile Runtime for Android")

The corresponding code can be found [here](https://github.com/pytorch/workshops/tree/master/PTMobileWalkthruAndroid).

Checkout our [Mobile Performance Recipes](https://pytorch.org/tutorials/recipes/mobile_perf.html) which cover how to optimize your model and check if optimizations helped via benchmarking.

In addition, follow this recipe to learn how to [make Native Android Application that use PyTorch prebuilt libraries](https://pytorch.org/tutorials/recipes/android_native_app_with_custom_op.html).

## API Docs

You can find more details about the PyTorch Android API in the [Javadoc](https://pytorch.org/javadoc/).

<!-- Do not remove the below script -->

<script page-id="android" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
