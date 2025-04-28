---
layout: blog_detail
title: 'How IBM Research Uses PyTorch and TerraTorch to Make Geospatial Computer Vision Accessible for Everyone'
hidden: true
---

Earth Observation-based analytics are becoming essential for understanding our planet — from monitoring deforestation to tracking urban development and analyzing the impacts of climate change. However, the coding and deep learning skills for applying AI models to satellite imagery and earth observation data has traditionally been a major barrier for many practitioners.

By IBM Research’s launch of TerraTorch 1.0, a PyTorch domain library for fine-tuning of Geospatial Computer Vision Foundation Models, we make geospatial AI not only more accessible but also more practical for the wider PyTorch community. Our goal: simplify the process so that any data scientist, researcher, or enthusiast can build powerful geospatial models with ease and low GPU and data processing requirements.

![globes](/assets/images/how-ibm-uses-pt-terratorch/fg1.png){:style="width:100%"}


**The power of foundation models, even with 75-95% of the input data removed, the models do a fantastic job in reconstruction of the input data - therefore learning the underlying physics of our planet in a deep, latent space**

## The Business Challenge

Our goal was to remove the technical barriers that prevent people from working with satellite imagery, weather and climate data at scale. Together with NASA, we’ve developed the Prithvi family of foundation models. Integrating the latest innovations of AI research using the clean API PyTorch provides has facilitated the job.

We wanted to create a framework that anyone can use to go from raw data to inference ready models in just a few steps.


![globes](/assets/images/how-ibm-uses-pt-terratorch/fg2.png){:style="width:100%"}


**How a weather and climate foundation model created and fine-tuned on PyTorch is used for weather forecasts**

## How IBM Research Used PyTorch

We’ve built TerraTorch on top of PyTorch, leveraging its dynamic ecosystem to integrate:



* PyTorch Lightning for clean, scalable training loops
* TorchGeo for geospatial data handling and transformations (PyTorch transforms)
For foundation models like the leading generative multimodal foundation model ['Terramind'](https://research.ibm.com/blog/terramind-esa-earth-observation-model), co-developed by IBM and ESA, and [the ‘Prithvi’ family](https://huggingface.co/ibm-nasa-geospatial), co-developed by IBM and NASA, TerraTorch has been used to fine-tune all of the downstream geospatial models for satellite imagery, weather and climate data. It includes the family of fine-tuned models that IBM has released as part of [Granite](https://huggingface.co/collections/ibm-granite/granite-geospatial-models-667dacfed21bdcf60a8bc982). In addition, other interesting foundation models and ecosystem components like Clay, SatMAE, Satlas, DeCur and DOFA are included in TerraTorch.
* Powerful and state-of-the-art vision transformers to experiment with modern neural network architectures
* TerraTorch-Iterate build on top of PyTorch, Optuna, MLFlow and Ray Tune for Hyperparameter Optimization (HPO), Neural Architecture Search (NAS) and Foundation Model Benchmarking (GeoBench), where TerraTorch became the reference implementation


![flow diagram](/assets/images/how-ibm-uses-pt-terratorch/fg5.png){:style="width:100%"}

**The fine-tuning and inference process is completely described in a single YAML config file. There, the architectural building blocks of the model (backbone, neck, decoder, head) are defined. The Model Factory assembles the model using the build-in and custom registries. In addition, the Optimizer and Data Modules are created as defined in the config. Finally, everything is passed to the Lightning Trainer, who executes the task.**


With PyTorch’s flexibility, we were able to prototype quickly, iterate on model architectures, and deploy pipelines for a range of geospatial applications — from flood and biomass detection to increasing resolution of climate data, where some of our our work became part of the [IBM Granite Geospatial Model Family](https://huggingface.co/collections/ibm-granite/granite-geospatial-models-667dacfed21bdcf60a8bc982).


![flow diagram](/assets/images/how-ibm-uses-pt-terratorch/fg3.png){:style="width:100%"}


**Architecture of the Prithvi-EO-2.0-600M foundation model which IBM Research developed together with NASA**

## Solving AI Challenges with PyTorch

PyTorch helped us to tackle three major challenges:

* Ease of experimentation: Dynamic computation graphs, automatic differentiation, full abstraction of CUDA and rich visualization tools made it simple to test different models and training strategies.
* Scalability: With DDP, FSDP, PyTorch Lightning and TorchGeo, we could train models on large-scale datasets without worrying about infrastructure.
* Community support: PyTorch - the de-facto standard in AI research - with its active community and excellent documentation made it easy to overcome hurdles and stay up to date with the latest advancements in AI research.

## A Word from IBM Research

*"PyTorch gave me the power to turn complex linear algebra and optimization problems into accessible, shareable solutions for the community. It feels empowering that we’re building and fine-tuning models for anyone curious about understanding our planet through AI."*

— Romeo Kienzler, AI Research Engineer at IBM Research Zurich, Rueschlikon


![quote](/assets/images/how-ibm-uses-pt-terratorch/fg4.png){:style="width:100%"}


## The Benefits of Using PyTorch

Using PyTorch allowed us to:



* Build a reproducible, open-source framework for fine-tuning geospatial foundation models
* Share our work with the community through easy-to-follow notebooks, TerraTorch configuration files, tutorials and model checkpoints on HuggingFace
* Rapidly iterate over foundation model architectures and deploy fine-tuned models for inference, from research to real-world client products

## Learn More

For more information about this project and to explore the code, visit:

* [GitHub Repository](https://github.com/IBM/terratorch)
* [IBM Research: Simplifying Geospatial AI with TerraTorch 1.0](https://research.ibm.com/blog/simplifying-geospatial-ai-with-terra-torch-1-0) 
* [TerraTorch PrithviEOv2 example notebooks](https://github.com/IBM/terratorch/tree/main/examples/tutorials/PrithviEOv2)
* [TerraMind example notebooks](https://github.com/IBM/terramind/tree/main/notebooks) 
* [Run TerraMind using TerraTorch on Colab](https://colab.research.google.com/github/IBM/terramind/blob/main/notebooks/terramind_v1_base_sen1floods11.ipynb)
