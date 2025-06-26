---
layout: blog_detail
title: "Introducing the PlayTorch app: Rapidly Create Mobile AI Experiences"
author: PlayTorch Team
featured-img: "/assets/videos/PlayTorch-video.mp4"
---

<p align="center">
  <video width="100%" controls>
    <source src="/assets/videos/PlayTorch-video.mp4" type="video/mp4">
  </video>
</p>

In December, we announced PyTorch Live, a toolkit for building AI-powered mobile prototypes in minutes. The initial release included a command-line interface to set up a development environment and an SDK for building AI-powered experiences in React Native. Today, we're excited to share that PyTorch Live will now be known as PlayTorch. This new release provides an improved and simplified developer experience. PlayTorch development is independent from the PyTorch project and the PlayTorch code repository is moving into the Meta Research GitHub organization.

## A New Workflow: The PlayTorch App

The PlayTorch team is excited to announce that we have partnered with [Expo](https://expo.dev) to change the way AI powered mobile experiences are built. Our new release simplifies the process of building mobile AI experiences by eliminating the need for a complicated development environment. You will now be able to build cross platform AI powered prototypes from the very browser you are using to read this blog.

In order to make this happen, we are releasing the [PlayTorch app](https://playtorch.dev/) which is able to run AI-powered experiences built in the [Expo Snack](https://snack.expo.dev/@playtorch/playtorch-starter?supportedPlatforms=my-device) web based code editor.

<p align="center">
  <img src="/assets/images/2022-7-15-introducing-the-playtorch-app-1.gif" width="100%">
</p>

The PlayTorch app can be downloaded from the Apple App Store and Google Play Store. With the app installed, you can head over to [playtorch.dev/snack](https://playtorch.dev/snack) and write the code for your AI-powered PlayTorch Snack. When you want to try what you’ve built, you can use the PlayTorch app’s QR code scanner to scan the QR code on the Snack page and load the code to your device.

NOTE: PlayTorch Snacks will not work in the Expo Go app.

## More to Explore in the PlayTorch App

### AI Demos

The PlayTorch app comes with several examples of how you can build AI powered experiences with a variety of different machine learning models from object detection to natural language processing. See what can be built with the PlayTorch SDK and be inspired to make something of your own as you play with the examples.

<p align="center">
  <img src="/assets/images/2022-7-15-introducing-the-playtorch-app-2.jpg" width="100%">
</p>

### Sharing Your Creations

Any PlayTorch Snack that you run in the PlayTorch app can be shared with others in an instant. When they open the link on their device, the PlayTorch app will instantly load what you’ve built from the cloud so they can experience it first hand.

<p align="center">
  <img src="/assets/images/2022-7-15-introducing-the-playtorch-app-3.jpg" width="100%">
</p>

When you have something you want to share, let us know on [Discord](https://discord.gg/sQkXTqEt33) or [Twitter](https://twitter.com/PlayTorch) or embed the PlayTorch Snack on your own webpage.

## SDK Overhaul

We learned a lot from the community after our initial launch in December and have been hard at work over the past several months to make the PlayTorch SDK (formerly known as PyTorch Live) simple, performant, and robust. In our initial version, the SDK relied on config files to define how a model ingested and output data.

Today, we are happy to announce the next version of our SDK can handle data processing in JavaScript for your prototypes with the new PlayTorch API that leverages the JavaScript Interface (JSI) to directly call C++ code. Not only have we completely redone the way you can interact with models, but we have also greatly expanded the variety of supported model architectures.

## A New Data Processing API for Prototyping

With this JSI API, we now allow users direct access to tensors (data format for machine learning). Instead of only having access to predefined transformations, you can now manipulate tensors however you would like for your prototypes.

<p align="center">
  <img src="/assets/images/2022-7-15-introducing-the-playtorch-app-4.gif" width="100%">
</p>

No more switching back and forth between code and config. You will now be able to write everything in JavaScript and have access to all of the type annotations and autocomplete features available to you in those languages.

Check out our [tutorials](https://playtorch.dev/tutorials) to see the new Data Processing API in action, take a deeper dive in the [API docs](https://playtorch.dev/docs/api/core/), or inspect the code yourself on [GitHub](https://github.com/facebookresearch/playtorch).

### Expanded Use Cases

With the new version of the SDK, we have added support for several cutting edge models.

<p align="center">
  <img src="/assets/images/2022-7-15-introducing-the-playtorch-app-5.jpg" width="100%">
</p>

Image-to-image transformations are now supported thanks to our robust JSI API, so you can see what your world would look like if it were an anime.

<p align="center">
  <img src="/assets/images/2022-7-15-introducing-the-playtorch-app-6.jpg" width="100%">
</p>

Translate French to English with an AI powered translator using the Seq2Seq model.

<p align="center">
  <img src="/assets/images/2022-7-15-introducing-the-playtorch-app-7.jpg" width="100%">
</p>

Use DeepLab V3 to segment images!

## Start Playing

If you want to start creating AI experiences yourself, head over to [playtorch.dev](https://playtorch.dev) and try out our [tutorials](https://playtorch.dev/tutorials/). Each tutorial will guide you through building a simple AI powered experience that you can instantly run on your phone and share with others.

## How to Get Support

Join us on [Discord](https://discord.gg/sQkXTqEt33), collaborate with us on [GitHub](https://github.com/facebookresearch/playtorch), or follow us on [Twitter](https://twitter.com/playtorch). Got questions or feedback? We’d love to hear from you!
