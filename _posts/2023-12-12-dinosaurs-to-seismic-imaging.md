---
layout: blog_detail
title: "From PyTorch Conference 2023: From Dinosaurs to Seismic Imaging with Intel"
author: Ramya Ravi, Susan Kahler at Intel
---

![Dinosaur fossil](/assets/images/hunting-dinosaurs-with-intel-ai-fig1.jpeg){:style="width:100%;"}


## Lightning Talk 1: Seismic Data to Subsurface Models with OpenFWI 

Speaker: Benjamin Consolvo, AI Software Engineering Manager, Intel, [LinkedIn](https://linkedin.com/in/bconsolvo)

### Session Overview

In this session, Ben begins with an overview of seismic imaging and full waveform inversion (FWI). Seismic imaging and FWI helps us to explore land for important subsurface minerals necessary for human thriving. To find those crucial subsurface minerals, we need to image the subsurface with a high degree of accuracy at a low cost, which involves two main challenges. He explains the solutions for those challenges using AI, which are summarized below.


<table class="table table-bordered">
  <tr>
   <td><strong>Challenges</strong>
   </td>
   <td><strong>Solutions using AI</strong>
   </td>
  </tr>
  <tr>
   <td>Traditional physics based FWI requires an accurate starting model.
   </td>
   <td>Data-driven deep learning solutions do not require an accurate starting model.
   </td>
  </tr>
  <tr>
   <td>GPUs are typically used for fine-tuning neural networks but are often unavailable and expensive.
   </td>
   <td>CPUs are highly available, inexpensive, and viable for AI fine-tuning. The new 4<sup>th</sup> Gen Intel® Xeon® Scalable processor has the built-in AI accelerator engine called Intel® AMX (Intel® Advanced Matrix Extensions) that helps to accelerate AI training and inference performance. 
   </td>
  </tr>
</table>


Next, he shows the wave propagation for the subsurface model and corresponding seismic shot gathers. In his example, the shot gathers are synthetically generated time-sampled records of sounds recordings from a shot (like a dynamite explosion or vibroseis truck) recorded by geophones spread across a large area. For this application, the training data consists of a pair of subsurface model image and seismic shot gather images, where the model from the shot gather is predicted. 


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td><strong>Number of Seismic Shot Images</strong>
   </td>
   <td><strong>Number of subsurface model images</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Train</strong>
   </td>
   <td>120,000
   </td>
   <td>24,000
   </td>
  </tr>
  <tr>
   <td><strong>Test</strong>
   </td>
   <td>25,000
   </td>
   <td>5,000
   </td>
  </tr>
  <tr>
   <td><strong>Validation</strong>
   </td>
   <td>5,000
   </td>
   <td>1,000
   </td>
  </tr>
</table>


In this application, the algorithm used during training was InversionNET (encoder-decoder convolutional neural network). Check out the implementation details for InversionNET architecture in [Deng et al. (2021)](https://arxiv.org/abs/2111.02926). 

He then shows the results:



1. Prediction versus ground truth model after one epoch and at 50 epochs. After training InversionNET, the predicted model is much closer to the ground truth image. 
2. Training loss and validation loss curves decreasing over time across 50 epochs.

Finally, Ben concludes his talk by highlighting that he was able to successfully fine-tune a deep neural network without an accurate starting model to obtain subsurface model on a 4th generation Intel® Xeon® Scalable processor.

Watch the [full video recording here](https://www.youtube.com/watch?v=TPp_Zyco6X4&list=PL_lsbAsL_o2BivkGLiDfHY9VqWlaNoZ2O&index=56) and download the [presentation](https://static.sched.com/hosted_files/pytorch2023/57/20231017_Consolvo_Seismic_PyTorchConf.pdf). More details can be found in this [blog](https://medium.com/better-programming/seismic-data-to-subsurface-models-with-openfwi-bcca0218b4e8). 

### About the Speaker 

![Ben Consolvo](/assets/images/ben-consolvo.jpg){:style="max-width:220px;float:right;margin-left: 20px;"}

Ben Consolvo is an AI Solutions Engineering Manager at Intel. He has been building a team and a program around Intel’s AI technology paired with Intel’s hardware offerings. He brings a background and passion in data science, particularly in deep learning (DL) and computer vision. He has applied his skills in DL in the cybersecurity industry to automatically identify phishing websites, as well as to the oil and gas industry to identify subsurface features for geophysical imaging.

## Lightning Talk 2: Dinosaur Bone Hunt

Speaker: Bob Chesebrough, Sr Solution Architect, Intel, [LinkedIn](https://www.linkedin.com/in/robertchesebrough/)

### Session Overview

In this session, Bob starts the presentation by explaining his interest in collecting dinosaur bones and gives an overview of [Intel AI Software portfolio](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/overview.html). 

He then explains the steps to create a dinosaur site treasure map or dinosaur bone likelihood map:



1. Collect data and create training data (New Mexico aerial photos of the Morrison Formation - a famous dinosaur bone bed in the Western United States and the GPS coordinates for small bone fragments discovered)
2. Train a simple ResNet 18 model using [Intel® Extension for PyTorch](https://www.intel.com/content/www/us/en/developer/tools/oneapi/optimization-for-pytorch.html#gs.1jggir)
3. Score the model on Utah photos and create a heat map

Finally, Bob shows the results that dinosaur bones were discovered in Utah using dinosaur bone likelihood map. Go to the [GitHub repository](https://github.com/intelsoftware/jurassic) to access the code sample and try out the sample using Intel Extension for PyTorch. 

Watch the [full video recording here](https://www.youtube.com/watch?v=Q_soyAhduKk&list=PL_lsbAsL_o2BivkGLiDfHY9VqWlaNoZ2O&index=67) and download the [presentation](https://static.sched.com/hosted_files/pytorch2023/86/PyTorch_Conf_Chesebrough_2023_PPT.pdf). More details can be found in this [blog](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-ai-step-by-step-guide-for-hunting-dinosaurs.html).

### About the Speaker

![Bob Chesebrough](/assets/images/bob-chesebrough.jpg){:style="max-width:220px;float:right;margin-left: 20px;"}

Bob Chesebrough's industry experience is software development/AI solution engineering for fortune 100 companies and national laboratories for over three decades. He is also a hobbyist who has logged over 800 miles and 1000 hours in the field finding dinosaur bones. He and his sons discovered an important fossil of the only known crocodilian from the Jurassic in New Mexico, they have also discovered and logged into the museum 2000+ bones localities and described a new mass bone bed in New Mexico.