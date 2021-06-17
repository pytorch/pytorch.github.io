# An Overview of the PyTorch Mobile Demo Apps

PyTorch Mobile provides a runtime environment to execute state-of-the-art machine learning models on mobile devices. Latency is reduced, privacy preserved, and models can run on mobile devices anytime, anywhere.

In this blog post, we provide a quick overview of 10 currently available PyTorch Mobile powered demo apps running various state-of-the-art PyTorch 1.9 machine learning models spanning images, video, audio and text.

It’s never been easier to deploy a state-of-the-art ML model to a phone. You don’t need any domain knowledge in Machine Learning and we hope one of the below examples resonates enough with you to be the starting point for your next project.

<div class="text-center">
  <img src="{{ site.url }}/assets/images/mobile_app_code.png" width="100%">
</div>

## Computer Vision
### Image Classification
This app demonstrates how to use PyTorch C++ libraries on iOS and Android to classify a static image with the MobileNetv2/3 model.


<img src="{{ site.url }}/assets/images/github_logo_32.png"> [iOS ##1](https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld) [iOS ##2](https://github.com/pytorch/workshops/tree/master/PTMobileWalkthruIOS) [Android ##1](https://github.com/pytorch/android-demo-app/tree/master/HelloWorldApp) [Android ##2](https://github.com/pytorch/workshops/tree/master/PTMobileWalkthruAndroid)

<img src="{{ site.url }}/assets/images/screencast.png"> [iOS](https://youtu.be/amTepUIR93k) [Android](https://youtu.be/5Lxuu16_28o)


### Live Image Classification
This app demonstrates how to run a quantized MobileNetV2 and Resnet18 models to classify images in real time with an iOS and Android device camera.

<img src="{{ site.url }}/assets/images/github_logo_32.png"> [iOS](https://github.com/pytorch/ios-demo-app/tree/master/PyTorchDemo) [Android](https://github.com/pytorch/android-demo-app/tree/master/PyTorchDemoApp)


### Image Segmentation
This app demonstrates how to use the PyTorch DeepLabV3 model to segment images. The updated app for PyTorch 1.9 also demonstrates how to create the model using the Mobile Interpreter and load the model with the LiteModuleLoader API.

<img src="{{ site.url }}/assets/images/github_logo_32.png"> [iOS](https://github.com/pytorch/ios-demo-app/tree/master/ImageSegmentation) [Android](https://github.com/pytorch/android-demo-app/tree/master/ImageSegmentation)

<img src="{{ site.url }}/assets/images/tutorial.png"> [iOS](https://pytorch.org/tutorials/beginner/deeplabv3_on_ios.html) [Android](https://github.com/pytorch/android-demo-app/tree/master/ImageSegmentation)


### Vision Transformer for Handwritten Digit Recognition
This app demonstrates how to use Facebook's latest optimized Vision Transformer DeiT model to do image classification and handwritten digit recognition.

<img src="{{ site.url }}/assets/images/github_logo_32.png"> [iOS](https://github.com/pytorch/ios-demo-app/tree/master/ViT4MNIST) [Android](https://github.com/pytorch/android-demo-app/tree/master/ViT4MNIST)

<img src="{{ site.url }}/assets/images/screencast.png"> [Android](https://drive.google.com/file/d/11L5mIjrLn7B7VdwjQl5vJv3ZVK4hcYut/view?usp=sharing)


### Object Detection
This app demonstrates how to convert the popular YOLOv5 model and use it on an iOS app that detects objects from pictures in your photos, taken with camera, or with live camera.

<img src="{{ site.url }}/assets/images/github_logo_32.png"> [iOS](https://github.com/pytorch/ios-demo-app/tree/master/ObjectDetection) [Android](https://github.com/pytorch/android-demo-app/tree/master/ObjectDetection)

<img src="{{ site.url }}/assets/images/screencast.png"> [iOS](https://drive.google.com/file/d/1pIDrUDnCD5uF-mIz8nbSlZcXxPlRBKhl/view) [Android](https://drive.google.com/file/d/1-5AoRONUqZPZByM-fy0m7r8Ct11OnlIT/view)


### D2Go
This app demonstrates how to create and use a much lighter and faster Facebook D2Go model to detect objects from pictures in your photos, taken with camera, or with live camera.

<img src="{{ site.url }}/assets/images/github_logo_32.png"> [iOS](https://github.com/pytorch/ios-demo-app/tree/master/D2Go) [Android](https://github.com/pytorch/android-demo-app/tree/master/D2Go)

<img src="{{ site.url }}/assets/images/screencast.png"> [iOS](https://drive.google.com/file/d/1GO2Ykfv5ut2Mfoc06Y3QUTFkS7407YA4/view) [Android](https://drive.google.com/file/d/18-2hLc-7JAKtd1q00X-5pHQCAdyJg7dZ/view?usp=sharing)


## Video
### Video Classification
This app demonstrates how to use a pre-trained PyTorchVideo model to perform video classification on tested videos, videos from the Photos library, or even real-time videos.

<img src="{{ site.url }}/assets/images/github_logo_32.png"> [iOS](https://github.com/pytorch/ios-demo-app/tree/master/TorchVideo) [Android](https://github.com/pytorch/android-demo-app/tree/master/TorchVideo)

<img src="{{ site.url }}/assets/images/screencast.png"> [iOS](https://drive.google.com/file/d/1ijb4UIuF2VQiab4xfAsBwrQXCInvb9wd/view) [Android](https://drive.google.com/file/d/193tkZgt5Rlk7u-EQPcvkoFtmOQ14-zCC/view)

## Natural Language Processing
### Text Classification
This app demonstrates how to use a pre-trained Reddit model to perform text classification.

<img src="{{ site.url }}/assets/images/github_logo_32.png"> [iOS](https://github.com/pytorch/ios-demo-app/tree/master/PyTorchDemo) [Android](https://github.com/pytorch/android-demo-app/tree/master/PyTorchDemoApp)

### Machine Translation
This app demonstrates how to convert a sequence-to-sequence neural machine translation model trained with the code in the PyTorch NMT tutorial for french to english translation.

<img src="{{ site.url }}/assets/images/github_logo_32.png"> [iOS](https://github.com/pytorch/ios-demo-app/tree/master/Seq2SeqNMT) [Android](https://github.com/pytorch/android-demo-app/tree/master/Seq2SeqNMT)

<img src="{{ site.url }}/assets/images/screencast.png"> [iOS](https://drive.google.com/file/d/17Edk-yAyfzijHPR_2ZDAIX7VY-TkQnLf/view) [Android](https://drive.google.com/file/d/110KN3Pa9DprkBWnzj8Ppa8KMymhmBI61/view?usp=sharing)


### Question Answering
This app demonstrates how to use the DistilBERT Hugging Face transformer model to answer questions about Pytorch Mobile itself.

<img src="{{ site.url }}/assets/images/github_logo_32.png"> [iOS](https://github.com/pytorch/ios-demo-app/tree/master/QuestionAnswering) [Android](https://github.com/pytorch/android-demo-app/tree/master/QuestionAnswering)

<img src="{{ site.url }}/assets/images/screencast.png"> [iOS](https://drive.google.com/file/d/1QIB3yoP4I3zUU0bLCpvUqPV5Kv8f8JvB/view) [Android](https://drive.google.com/file/d/10hwGNFo5tylalKwut_CWFPJmV7JRdDKF/view?usp=sharing)

## Audio
### Speech Recognition
This app demonstrates how to convert Facebook AI's torchaudio-powered wav2vec 2.0, one of the leading models in speech recognition to TorchScript before deploying it.

<img src="{{ site.url }}/assets/images/github_logo_32.png"> [iOS](https://github.com/pytorch/ios-demo-app/tree/master/SpeechRecognition) [Android](https://github.com/pytorch/android-demo-app/tree/master/SpeechRecognition)


We really hope one of these demo apps stood out for you. For the full list make sure to visit the [iOS](https://github.com/pytorch/ios-demo-app) and [Android](https://github.com/pytorch/android-demo-app) demo app repos.
