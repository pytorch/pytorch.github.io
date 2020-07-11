---
layout: blog_detail
title: 'How Disney uses PyTorch for animated character recognition'
author: Miquel Àngel Farré, Anthony Accardo, Marc Junyent, Monica Alfaro, Cesc Guitart
---

## Disney's Content Genome
The long and incremental evolution of the media industry, from a traditional broadcast and home video model, to more readily-accessible digital content, has accelerated the use of machine learning and artificial intelligence (AI). Advancing the implementation of these technologies is critical for a company like Disney that has produced nearly a century of content, as it allows for new consumer experiences and enables new applications for illustrators and writers to create the highest-quality content.

In response to this industry shift, Disney's Content Genome was built by a team of R&D engineers and information scientists within Disney's Direct-to-Consumer & International Organization (DTCI) to power new applications for digital product innovation, and create a comprehensive digital archive. The platform populates knowledge graphs with content metadata, which powers AI applications across search, personalization, and production operations, all of which are critical components of digital video platforms. This metadata improves tools that are used to produce content; powers user experiences through recommendation engines, digital navigation and discovery; and enables business intelligence and analytics.

In order to bring the Content Genome to life, a significant investment in manual and automated content annotation, computer vision and machine learning techniques was necessary - and PyTorch helped us meet this challenge.

<div class="text-center">
  <img src="{{ site.url }}/assets/images/Disney_DTCI_Logo.png" width="100%">
</div>

## Tagging Disney Content
As a first step toward powering the Content Genome, DTCI engineers created the first automated tagging pipeline. Tagging content is an important component of DTCI's use of supervised learning, which is regularly employed in custom use cases that require specific detection. Tagging is also the only way to identify a lot of highly contextual story and character information from structured data, like storylines, character archetypes or motivations.

This automated tagging pipeline was equipped with face detection and recognition modules based on traditional machine learning approaches and performed well enough to recognize real human faces from characters in shows and feature films. Part of this success relied on the combination of machine learning methods, like HOG+SVM, and the DTCI knowledge graph, which specifically defines relations between particular entities of Disney content, such as the link between a specific episode of a series and the subset of locations or characters that appear in that episode. This initial success in facial recognition was then extended to the classification of other entities, such as locations.

Recent advancements in deep learning helped extend our models beyond facial recognition, largely due to pre-trained models based on new architectures that can be fine-tuned to create custom models aligned to our intellectual property. PyTorch allows us to have state-of-the-art pre-trained models accessible as a starting point to fulfill our needs expediently, and make our archiving process more efficient.

The next step toward more advanced facial detection and recognition, and one of the major technical challenges for Disney's content catalogue, pertains to tagging animated content. The first question we approached was: **_How do we move from facial detection in real-world content to facial detection in animated content?_**

The natural first approach was to try our live action face detection against animated content; while this worked in some cases, it wasn't a consistent solution. After some analysis, we determined that methods like HOG+SVM are robust against color, brightness or texture changes, but the models used could only match human features (i.e. two eyes, one nose and a mouth) in animated characters with human proportions. In comparison, a human brain can identify faces even when they appear on the front of a car or on an alien body - and for Disney, we need to ensure we can detect characters that fall into this category, like Lightning McQueen from *Cars* or Mike Wazowski from *Monsters, Inc.*, to build the most robust archive possible.

## Animated Face Detection
Our first experiment was to validate if the same HOG+SVM pipeline that worked for animated faces of characters with human proportions could work with animated faces of characters that are not humans.

We manually annotated a few samples using two Disney Junior animated shows, *Elena of Avalor* and *The Lion Guard*, drawing squares around faces for a couple hundred frames per show. With the manually annotated dataset, we validated that a HOG+SVM pipeline performed poorly in animated faces, specifically, human-like faces and animal-like faces, and knew we needed a technique able to learn more abstract concepts of faces.

<div class="text-center">
  <img src="{{ site.url }}/assets/images/princess-elena.png" width="100%">
</div>
*Disney Junior's Princess Elena of Avalor and magical flying creature, Migs, with manually annotated facial detection methods applied.*

<div class="text-center">
  <img src="{{ site.url }}/assets/images/disney-junior-bunga.png" width="100%">
</div>
*Disney Junior's The Lion Guard character, Bunga, demonstrates the complexity of animated, non-human facial detection.*

If we wanted to apply deep learning approaches, we would either have to collect thousands of different *faces* from animated content or apply transfer learning from another domain to animated content. The second solution had the advantage of requiring a much smaller training set.

We continued our experimentation using the samples that we had for the HOG+SVM experiment to fine tune a Faster-R CNN Object Detection architecture trained over the COCO dataset with the single goal of identifying animated faces.

Even with the small number of samples we used in this transfer learning solution, we obtained satisfactory results testing with images in the dataset. However, when we ran the detector against images that didn't contain animated faces, we often found false positive detections.

## The Relevance of Negative Samples
False positive detections are a common problem with transfer learning on custom datasets, largely due to the limited context in the training images.

In our particular case, during training, every object that appears in the image that is not the object of interest is considered as a negative sample. The background of animated content usually has flat regions and few details. Hence, the Faster-RCNN model was incorrectly learning that everything that stood out against a simple background was an animated face. For example, any text clearly in the foreground was considered a positive detection. Although our dataset had enough positive images to detect animated faces, it didn't have rich negative samples from detailed backgrounds.

We decided to increase our initial dataset with images not containing animated faces but with other objects from animated series or features.

In order to make this technically possible, we extended Torchvision's Faster-RCNN implementation to allow the load of negative images during the training process without annotations. This is a new feature that we contributed to Torchvision 0.6 with the guidance of the Torchvision core developers.

Adding negative examples in the dataset drastically reduced false positives at inference time, providing outstanding results.

## Speeding Up a Video Processing Pipeline with PyTorch
With an animated character face detector performing properly, our next goal was to accelerate our video analysis pipeline. For this task and thanks to the PyTorch team, we discovered that PyTorch is more than a framework to run neural net architectures and can be used to parallelize and speedup other tasks.

Running the new animated face detector on each frame is time-consuming and ineffective, so we combined our face detection model with other algorithms like a bounding box tracker and a shot detector (a shot is defined as the continuous sequence of frames between two edits or cuts). This allowed us to accelerate the processing, as fewer detections are required, and we can propagate the detected faces to all the frames. Furthermore, it provided us temporal information; instead of only detecting independent frames, they are contextualized as segments of the video where a character appears.

These relationships between models expose dependencies that impact our implementation of classifiers. For example, we choose the frames to send to the detection model depending on the output of the shot boundaries classifiers. Our pipeline has to take these dependencies into account and remove redundant computations to be as fast as possible.

Reading and decoding the video is also time-consuming so that's the first thing we optimized. We use a custom PyTorch IterableDataset that, in combination with PyTorch's DataLoader, allows us to read different parts of the video with parallel CPU workers.

The video is split in chunks based on its I-frames and each worker reads different chunks. This provides batches of contiguous ordered frames, although batches might not be ordered.

<div class="text-center">
  <img src="{{ site.url }}/assets/images/dtci-video-batching.png" width="100%">
</div>
*Video batching strategy*.

Even though this video reading approach is very fast, we try to do all our computations with a single video read. To do this we implemented most of our pipeline in PyTorch with GPU execution in mind. Each frame is sent only once to the GPU and there we apply all our algorithms on each batch, reducing the communication between CPU and GPU to a minimum.

We also use PyTorch to implement more traditional algorithms such as our shot detector, which doesn't use neural nets and primarily performs operations such as color space changes, histograms and singular value decompression (SVD). Using PyTorch to implement even the more traditional algorithms in our pipeline allows us to move computations to GPU with minimal cost and to easily recycle intermediate results shared between multiple algorithms.

By moving our CPU parts of the pipeline to GPU with PyTorch and speeding up our video reading with DataLoader, we were able to speed up our processing time by a factor of 10, taking full advantage of our hardware.

## The Right, Community-Driven Philosophy
PyTorch has been present in our animated character detection R&D from the initial neural net architecture experimentations to the latest efficiency improvements in our production environment.

From a discovery perspective, the well-maintained popular datasets and model architectures in Torchvision, combined with its popularity across academia, allowed us to compare state-of-the-art approaches and validate which ones better fit our needs, accelerating our R&D.

Digging into PyTorch core components such as IterableDataset, DataLoader and the common image transformations for computer vision in Torchvision, enabled us to increase data loading and algorithm efficiency in our production environment, growing our use of PyTorch from the inference or model training resource to a full pipeline optimization toolset.

We have also had the opportunity to meet the community behind PyTorch, which encouraged us to propose changes to the framework, hold discussions to find together the best approach for the community and eventually make the agreed solution part of the framework. This guidance was key for us to contribute. It was also paramount to understand that each addition to the framework is examined from different perspectives and to ensure it is the correct move in terms of performance and functionality.
