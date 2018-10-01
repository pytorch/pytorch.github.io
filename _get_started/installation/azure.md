# Using PyTorch with Azure
{:.no_toc}

To gain the full experience of what PyTorch has to offer, a machine with at least one dedicated NVIDIA GPU is necessary. While it is not always practical to have your own machine with these specifications, there are our cloud based solutions to allow you to test and use PyTorch's full features.

Azure [provides](https://azure.microsoft.com/en-us/services/machine-learning-services/){:target="_blank"}:

* dedicated, pre-built [machine learning virtual machines](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/){:target="_blank"}, complete with PyTorch.
* bare Linux and Windows virtual machines for you to do a custom install of PyTorch.
* [notebooks](https://notebooks.azure.com/pytorch){:target="_blank"} to help you learn PyTorch and machine learning.
* a [machine learning service](https://azure.microsoft.com /en-us/services/machine-learning-services/ ) with a robust Python SDK to help you train and deploy PyTorch models at cloud scale.

## Azure Primer
{: #microsoft-azure-primer}

In order to use Azure, you need to set up an [Azure account](https://azure.microsoft.com/en-us/free/){:target="_blank"}, if you do not have one already. You will use a Microsoft-recognized email address and password. You will also verify your identity by providing contact and billing information. The billing information is necessary because while Azure does provide free usage credits and free services, you may need or want higher-end services as well.

Once you are logged in, you will be brought to your [Azure portal](https://portal.azure.com/){:target="_blank"}.  You can even learn more about Azure through a set of [simple video tutorials](https://azure.microsoft.com/en-us/get-started/video/){:target="_blank"}.

### GPU-based Virtual Machines

Generally, you will be using [Azure Virtual Machines](https://azure.microsoft.com/en-us/services/virtual-machines/){:target="_blank"}) to begin with PyTorch. Microsoft has various virtual machine types and pricing options, with both [Linux](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/){:target="_blank"} and [Windows](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/windows/){:target="_blank"}, all of which are configured for specific use cases. For PyTorch, it is highly recommended that you use the [GPU optimized](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/sizes-gpu){:target="_blank"}, virtual machines. They are tailored for the high compute needs of machine learning.

The expense of your virtual machine is directly correlated to the number of GPUs that it contains. The NC6 virtual machine is, for example, one of the smallest, cheapest virtual machines and can actually be suitable for many use cases.

## Azure Machine Learning Service
{: #microsoft-azure-machine-learning-service}

The [Azure Machine Learning Service](https://azure.microsoft.com/en-us/services/machine-learning-service/) provides a cloud-based environment you can use to develop, train, test, deploy, manage, and track machine learning models. See [the documentation](https://docs.microsoft.com/azure/machine-learning/service/how-to-train-pytorch) to learn how to use PyTorch with Azure Machine Learning.

## Pre-Configured Data Science Virtual Machines
{: #microsoft-azure-virtual-machines}

Azure [provides](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/){:target="_blank"} [pre-configured](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/){:target="_blank"} data learning and machine learning virtual machines. PyTorch are available on many of these - for example here is the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro){:target="_blank"} for how to setup an Azure virtual machine on Ubuntu Linux.

## Installing PyTorch From Scratch
{: #microsoft-azure-from-scratch}

You may prefer to start with a bare virtual machine to install PyTorch. Once you have connected to your virtual machine, setting up PyTorch is the same as [setting up locally](get-started) for your operating system of choice.