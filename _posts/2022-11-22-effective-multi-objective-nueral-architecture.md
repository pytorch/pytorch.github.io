---
layout: blog_detail
title: "Efficient Multi-Objective Neural Architecture Search with Ax"
author: David Eriksson, Max Balandat
featured-img: "/assets/images/MOO-NAS-blog-img2-pareto_frontier_plot.png"
---

## tl;dr

Multi-Objective Optimization in Ax enables efficient exploration of tradeoffs (e.g. between model performance and model size or latency) in Neural Architecture Search. This method has been successfully applied at Meta for a variety of products such as On-Device AI. In this post, we provide an [end-to-end](https://pytorch.org/tutorials/intermediate/ax_multiobjective_nas_tutorial.html) tutorial that allows you to try it out yourself.

## Introduction

Neural networks continue to grow in both size and complexity. Developing state-of-the-art architectures is often a cumbersome and time-consuming process that requires both domain expertise and large engineering efforts. In an attempt to overcome these challenges, several Neural Architecture Search (NAS) approaches have been proposed to automatically design well-performing architectures without requiring a human in-the-loop.

Despite being very sample-inefficient, naïve approaches like random search and grid search are still popular for both hyperparameter optimization and NAS (a [study](https://hal.archives-ouvertes.fr/hal-02447823/document) conducted at NeurIPS 2019 and ICLR 2020 found that 80% of NeurIPS papers and 88% of ICLR papers tuned their ML model hyperparameters using manual tuning, random search, or grid search). But as models are often time-consuming to train and may require large amounts of computational resources, minimizing the number of configurations that are evaluated is important.

[Ax](https://ax.dev/) is a general tool for black-box optimization that allows users to explore large search spaces in a sample-efficient manner using [state-of-the art algorithms such as Bayesian Optimization](http://proceedings.mlr.press/v133/turner21a/turner21a.pdf). At Meta, Ax is used in a variety of domains, including hyperparameter tuning, NAS, identifying optimal product settings through large-scale A/B testing, infrastructure optimization, and designing cutting-edge AR/VR hardware.

In many NAS applications, there is a natural tradeoff between multiple metrics of interest. For instance, when deploying models on-device we may want to maximize model performance (e.g., accuracy), while simultaneously minimizing competing metrics such as power consumption, inference latency, or model size, in order to satisfy deployment constraints. In many cases, we have been able to reduce computational requirements or latency of predictions substantially by accepting a small degradation in model performance (in some cases we were able to both increase accuracy and reduce latency!). Principled methods for exploring such tradeoffs efficiently are key enablers of [Sustainable AI](https://arxiv.org/abs/2111.00364).

At Meta, we have successfully used [multi-objective Bayesian NAS](https://research.facebook.com/blog/2021/07/optimizing-model-accuracy-and-latency-using-bayesian-multi-objective-neural-architecture-search/) in Ax to explore such tradeoffs. Our methodology is being used routinely for optimizing AR/VR on-device ML models. Beyond NAS applications, we have also developed [MORBO](https://arxiv.org/pdf/2109.10964.pdf) which is a method for high-dimensional multi-objective optimization that can be used to optimize optical systems for augmented reality (AR).

## Fully automated Multi-Objective NAS with Ax

Ax’s Scheduler allows running experiments asynchronously in a closed-loop fashion by continuously deploying trials to an external system, polling for results, leveraging the fetched data to generate more trials, and repeating the process until a stopping condition is met. No human intervention or oversight is required. Features of the Scheduler include:

- Customizability of parallelism, failure tolerance, and many other settings;

- A large selection of state-of-the-art optimization algorithms;

- Saving in-progress experiments (to a SQL DB or json) and resuming an experiment from storage;

- Easy extensibility to new backends for running trial evaluations remotely.

The following illustration from the [Ax scheduler tutorial](https://ax.dev/tutorials/scheduler.html) summarizes how the scheduler interacts with any external system used to run trial evaluations:

<!-- image goes here  -->

<p align="center">
<img src="/assets/images/MOO-NAS-blog-img1-ax_scheduler_illustration.png" width="90%">
</p>

To run automated NAS with the Scheduler, the main things we need to do are:

- Define a [Runner](https://github.com/facebook/Ax/blob/main/ax/core/runner.py#L21), which is responsible for sending off a model with a particular architecture to be trained on a platform of our choice (like Kubernetes, or maybe just a Docker image on our local machine). In the tutorial below, we use TorchX for handling deployment of training jobs.

- Define a [Metric](https://github.com/facebook/Ax/blob/main/ax/core/metric.py#L21), which is responsible for fetching the objective metrics (such as accuracy, model size, latency) from the training job. In our tutorial, we use Tensorboard to log data, and so can use the Tensorboard metrics that come bundled with Ax.

## Tutorial

In our tutorial we show how to use Ax to run multi-objective NAS for a simple neural network model on the popular MNIST dataset. While the underlying methodology can be used for more complicated models and larger datasets, we opt for a tutorial that is easily runnable end-to-end on a laptop in less than an hour. In our example, we will tune the widths of two hidden layers, the learning rate, the dropout probability, the batch size, and the number of training epochs. The goal is to trade off performance (accuracy on the validation set) and model size (the number of model parameters) using [multi-objective Bayesian optimization](https://proceedings.neurips.cc/paper/2021/file/11704817e347269b7254e744b5e22dac-Paper.pdf).

The tutorial makes use of the following PyTorch libraries:

- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) (specifying the model and training loop)

- [TorchX](https://github.com/pytorch/torchx) (for running training jobs remotely / asynchronously)

- [BoTorch](https://github.com/pytorch/botorch) (the Bayesian optimization library that powers Ax’s algorithms)

The complete runnable example is available as a **[PyTorch Tutorial](https://pytorch.org/tutorials/intermediate/ax_multiobjective_nas_tutorial.html)**.

### Results

The final results from the NAS optimization performed in the tutorial can be seen in the tradeoff plot below. Here, each point corresponds to the result of a trial, with the color representing its iteration number, and the star indicating the reference point defined by the thresholds we imposed on the objectives. We see that our method was able to successfully explore the trade-offs between validation accuracy and number of parameters and found both large models with high validation accuracy as well as small models with lower validation accuracy. Depending on the performance requirements and model size constraints, the decision maker can now choose which model to use or analyze further.

<p align="center">
<img src="/assets/images/MOO-NAS-blog-img2-pareto_frontier_plot.png" width="100%">
</p>

### Visualizations

Ax provides a number of visualizations that make it possible to analyze and understand the results of an experiment. Here, we will focus on the performance of the Gaussian process models that model the unknown objectives, which are used to help us discover promising configurations faster. Ax makes it easy to better understand how accurate these models are and how they perform on unseen data via leave-one-out cross-validation. In the figures below, we see that the model fits look quite good - predictions are close to the actual outcomes, and predictive 95% confidence intervals cover the actual outcomes well. Additionally, we observe that the model size `(num_params)` metric is much easier to model than the validation accuracy `(val_acc)` metric.

<!-- another image  -->

<style>

    .cross-validation-container{
        display:flex; 
        flex-direction:row; 
    }

</style>

<div class="cross-validation-container">
<p align="center">
<img src="/assets/images/MOO-NAS-blog-img3-cv_plot_val_acc.png" width="100%">
</p>

<p align="center">
<img src="/assets/images/MOO-NAS-blog-img4-cv_plot_num_params.png" width="100%">
</p>
</div>

## Takeaways

- We showed how to run a fully automated multi-objective Neural Architecture Search using Ax.

- Using the Ax Scheduler, we were able to run the optimization automatically in a fully asynchronous fashion - this can be done locally (as done in the tutorial) or by deploying trials remotely to a cluster (simply by changing the TorchX scheduler configuration).

- The state-of-the-art multi-objective Bayesian optimization algorithms available in Ax allowed us to efficiently explore the tradeoffs between validation accuracy and model size.

## Advanced Functionality

Ax has a number of other advanced capabilities that we did not discuss in our tutorial. Among these are the following:

### Early Stopping

When evaluating a new candidate configuration, partial learning curves are typically available while the NN training job is running. We can use the information contained in the partial curves to identify under-performing trials to stop early in order to free up computational resources for more promising candidates. While not demonstrated in the above tutorial, Ax supports early stopping out-of-the-box.

### High-dimensional search spaces

In our tutorial, we used Bayesian optimization with a standard Gaussian process in order to keep the runtime low. However, these models typically scale to only about 10-20 tunable parameters. Our new SAASBO method ([paper](https://proceedings.mlr.press/v161/eriksson21a/eriksson21a.pdf), [Ax tutorial](https://ax.dev/tutorials/saasbo.html), [BoTorch tutorial](https://botorch.org/tutorials/saasbo)) is very sample-efficient and enables tuning hundreds of parameters. SAASBO can easily be enabled by passing `use_saasbo=True` to `choose_generation_strategy`.

## Acknowledgements

We thank the TorchX team (in particular Kiuk Chung and Tristan Rice) for their help with integrating TorchX with Ax, and the Adaptive Experimentation team @ Meta for their contributions to Ax and BoTorch.

## References

[D. Eriksson, P. Chuang, S. Daulton, M. Balandat. Optimizing model accuracy and latency using Bayesian multi-objective neural architecture search. Meta Research blog, July 2021.](https://research.facebook.com/blog/2021/07/optimizing-model-accuracy-and-latency-using-bayesian-multi-objective-neural-architecture-search/)
