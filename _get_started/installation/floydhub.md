# Using PyTorch with FloydHub
{:.no_toc}

To gain the full experience of what PyTorch has to offer, a machine with at least one dedicated NVIDIA GPU is necessary. While it is not always practical to have your own machine with these specifications, there are our cloud based solutions to allow you to test and use PyTorch's full features.

FloydHub [provides](https://www.floydhub.com/){:target="_blank"}:

* dedicated, pre-built [machine learning virtual machines](https://www.floydhub.com/product/build){:target="_blank"}, complete with PyTorch.
* a [machine learning service](https://www.floydhub.com/product/train) with a robust Python SDK to help you train and deploy PyTorch models at cloud scale.

## FloydHub Primer
{: #floydhub-primer}

In order to use FloydHub, you need to set up an [FloydHub account](https://www.floydhub.com/signup/){:target="_blank"}, if you do not have one already. If you have a Google or Github account, you can speed up this step by signing from one of those. You will also verify your identity by providing billing information. The billing information is necessary because while FloydHub does provide free usage credits and free services, you may need or want higher-end services as well.

Once you are logged in, you will be brought to your [FloydHub web dashboard](https://www.floydhub.com/){:target="_blank"} where you will create your first project.  You can even learn more about FloydHub through a set of [simple video tutorials](https://youtu.be/FcsjqQ2QdLQ){:target="_blank"}.

### FloydHub Environments

FloydHub provides no setup required, pre-configured environment to help you build your deep learning projects. [FloydHub Deep Learning Environments](https://docs.floydhub.com/guides/environments/){:target="_blank"} are a set of Ubuntu-based virtual machines that allow you to build and run [PyTorch machine learning based applications](https://docs.floydhub.com/guides/pytorch/).

## FloydHub Workspace: GPU-based Virtual Machines
{: #floydhub-workspace}

Generally, you will be using [FloydHub Workspace](https://www.floydhub.com/product/build){:target="_blank"} to begin with PyTorch. FlodyHub Workspace is an interactive environment (Jupyter Lab) for developing and running code. You can run Jupyter notebooks, Python scripts and much more. All the files and data in your workspace will be preserved after shutdown.

By clicking on the below button, you will create a new FloydHub Workspace with the notebook tutorial: [WHAT IS TORCH.NN REALLY?](https://pytorch.org/tutorials/beginner/nn_tutorial.html) created by Jeremy Howard and the [fast.ai](https://www.fast.ai/) Team.

[![Run on FH](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run)

## FloydHub Job: Machine Learning Service
{: #floydhub-job}

The [FloydHub Job experience](https://www.floydhub.com/product/train) is a cloud-based service you can use to accelerate your end-to-end machine learning workflows, from training to production. FloydHub allows you to easily move from training PyTorch models on your local machine to scaling out to the cloud. Using [`floyd-cli` or Python SDK](https://github.com/floydhub/floyd-cli), you can leverage the service’s advanced functionality for hyperparameter tuning, run history tracking, and production-scale model deployments.

See [this repository](https://github.com/floydhub/mnist) to learn how to run your first PyTorch MNIST example as a FloydHub Job.

## Installing PyTorch From Scratch
{: #floydhub-from-scratch}

You may prefer to start with a bare virtual machine to install PyTorch. Once you have connected to your virtual machine, setting up PyTorch is the same as [setting up locally](get-started) for your operating system of choice.
