# Using PyTorch with Azure
{:.no_toc}

To gain the full experience of what PyTorch has to offer, a machine with at least one dedicated NVIDIA GPU is necessary. While it is not always practical to have your own machine with these specifications, there are our cloud based solutions to allow you to test and use PyTorch's full features.

Azure [provides](https://azure.microsoft.com/en-us/services/machine-learning-services/){:target="_blank"}:

* a [machine learning service](https://azure.microsoft.com/en-us/services/machine-learning/) with a robust Python SDK to help you train and deploy PyTorch models at cloud scale.
* dedicated, pre-built [machine learning virtual machines](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/){:target="_blank"}, complete with PyTorch.
* bare Linux and Windows virtual machines for you to do a custom install of PyTorch.

## PyTorch Enterprise on Azure
{: #pytorch-enterprise-on-azure}

Microsoft is one of the founding members and also the inaugural participant of the [PyTorch Enterprise Support Program](https://pytorch.org/enterprise-support-program). Microsoft offers PyTorch Enterprise on Azure as a part of Microsoft [Premier](https://www.microsoft.com/en-us/msservices/premier-support) and [Unified](https://www.microsoft.com/en-us/msservices/unified-support-solutions?activetab=pivot1:primaryr4) Support. The PyTorch Enterprise support service includes long-term support to selected versions of PyTorch for up to 2 years, prioritized troubleshooting, and the latest integration with [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/) and other PyTorch add-ons including ONNX Runtime for faster inference. 

To learn more and get started with PyTorch Enterprise on Microsoft Azure, [visit here](https://azure.microsoft.com/en-us/develop/pytorch/).

For documentation, [visit here](https://docs.microsoft.com/en-us/azure/pytorch-enterprise/).

## Azure Primer
{: #microsoft-azure-primer}

In order to use Azure, you need to set up an [Azure account](https://azure.microsoft.com/en-us/free/){:target="_blank"}, if you do not have one already. You will use a Microsoft-recognized email address and password. You will also verify your identity by providing contact and billing information. The billing information is necessary because while Azure does provide free usage credits and free services, you may need or want higher-end services as well.

Once you are logged in, you will be brought to your [Azure portal](https://portal.azure.com/){:target="_blank"}.  You can even learn more about Azure through a set of [simple video tutorials](https://azure.microsoft.com/en-us/get-started/video/){:target="_blank"}.

## Azure Machine Learning Service
{: #microsoft-azure-machine-learning-service}

The [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) is a cloud-based service you can use to accelerate your end-to-end machine learning workflows, from training to production. Azure Machine Learning allows you to easily move from training PyTorch models on your local machine to scaling out to the cloud. Using Azure ML’s CLI or Python SDK, you can leverage the service’s advanced functionality for distributed training, hyperparameter tuning, run history tracking, and production-scale model deployments.

See [the documentation](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-pytorch) to learn how to use PyTorch with Azure Machine Learning.

## Pre-Configured Data Science Virtual Machines
{: #microsoft-azure-virtual-machines}

Azure [provides](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/){:target="_blank"} [pre-configured](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/){:target="_blank"} data learning and machine learning virtual machines. PyTorch are available on many of these - for example here is the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro){:target="_blank"} for how to setup an Azure virtual machine on Ubuntu Linux.

### GPU-based Virtual Machines

Microsoft has various virtual machine types and pricing options, with both [Linux](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/){:target="_blank"} and [Windows](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/windows/){:target="_blank"}, all of which are configured for specific use cases. For PyTorch, it is highly recommended that you use the [GPU optimized](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/sizes-gpu){:target="_blank"}, virtual machines. They are tailored for the high compute needs of machine learning.

The expense of your virtual machine is directly correlated to the number of GPUs that it contains. The NC6 virtual machine is, for example, one of the smallest, cheapest virtual machines and can actually be suitable for many use cases.

## Installing PyTorch From Scratch
{: #microsoft-azure-from-scratch}

You may prefer to start with a bare virtual machine to install PyTorch. Once you have connected to your virtual machine, setting up PyTorch is the same as [setting up locally](/get-started) for your operating system of choice.
