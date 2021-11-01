# Using PyTorch with AWS
{:.no_toc}

To gain the full experience of what PyTorch has to offer, a machine with at least one dedicated NVIDIA GPU is necessary. While it is not always practical to have your own machine with these specifications, there are our cloud based solutions to allow you to test and use PyTorch's full features.

AWS [provides](https://aws.amazon.com/machine-learning/amis/){:target="_blank"} both:

* Deep Learning AMIs: dedicated, pre-built machine learning instances, complete with PyTorch
* Deep Learning Base AMI: bare Linux and Windows instances for you to do a custom install of PyTorch.

## Quick Start on Deep Learning AMI
{: #aws-quick-start}

If you want to get started with a Linux AWS instance that has PyTorch already installed and that you can login into from the command-line, this step-by-step guide will help you do that.

1. Sign into your [AWS console](https://aws.amazon.com/console/). If you do not have an AWS account, see the [primer](#aws-primer) below.
1. Click on `Launch a virtual machine`.
1. Select `Deep Learning AMI (Ubuntu)`.
   > This gives you an instance with a pre-defined version of PyTorch already installed. If you wanted a bare AWS instance that required PyTorch to be installed, you could choose the `Deep Learning Base AMI (Ubuntu)`, which will have the hardware, but none of the software already available.
1. Choose a GPU compute `p3.2xlarge` instance type.
   > You can choose any of the available instances to try PyTorch, even the *free-tier*, but it is recommended for best performance that you get a *GPU compute* or *Compute optimized* instance. Other instance options include the Compute Optimized c5-series (e.g., `c5.2xlarge`) or the General Compute t2-series or t3-series (e.g., `t2.2xlarge`). It is important to note that if you choose an instance without a GPU, PyTorch will only be running in CPU compute mode, and operations may take much, much longer.
1. Click on `Review and Launch`.
1. Review the instance information and click `Launch`.
1. You will want to `Create a new key pair` if you do not have one already to use. Pick a name and download it locally via the `Download Key Pair` button.
1. Now click on `Launch Instances`. You now have a live instance to use for PyTorch. If you click on `View Instances`, you will see your running instance.
1. Take note of the `Public DNS` as this will be used to `ssh` into your instance from the command-line.
1. Open a command-line prompt
1. Ensure that your key-pair has the proper permissions, or you will not be able to log in. Type `chmod 400 path/to/downloaded/key-pair.pem`.
1. Type `ssh -i path/to/downloaded/key-pair.pem ubuntu@<Public DNS that you noted above>`. e.g., `ssh -i ~/Downloads/aws-quick-start.pem ubuntu@ec2-55-181-112-129.us-west-2.compute.amazonaws.com`. If asked to continue connection, type `yes`.
1. You should now see a prompt similar to `ubuntu@ip-100-30-20-95`. If so, you are now connected to your instance.
1. Verify that PyTorch is installed by running the [verification steps below](#quick-start-verification).
   > If you chose the `Deep Learning Base AMI (Ubuntu)` instead of the `Deep Learning AMI (Ubuntu)`, then you will need to install PyTorch. Follow the [Linux getting started instructions](/get-started) in order to install it.

### Quick Start Verification

To ensure that PyTorch was installed correctly, we can verify the installation by running sample PyTorch code. Here we will construct a randomly initialized tensor.


```python
import torch
x = torch.rand(5, 3)
print(x)
```

The output should be something similar to:

```
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```

Additionally, to check if your GPU driver and CUDA is enabled and accessible by PyTorch, run the following commands to return whether or not the CUDA driver is enabled:

```python
import torch
torch.cuda.is_available()
```

<div>
  <a href="javascript:void(0);" class="btn btn-lg btn-orange btn-demo show-screencast">Show Demo</a>
  <div class="screencast">
    <script src="https://asciinema.org/a/15dyZZvvakqbfKgfh2LByMkXz.js" id="asciicast-15dyZZvvakqbfKgfh2LByMkXz" data-speed="2" async></script>
    <a href="javascript:void(0);" class="btn btn-lg btn-orange btn-demo show-info">Hide Demo</a>
  </div>
</div>

## AWS Primer
{: #aws-primer}

Generally, you will be using Amazon Elastic Compute Cloud (or [EC2](https://aws.amazon.com/ec2/?ec2-whats-new.sort-by=item.additionalFields.postDateTime&ec2-whats-new.sort-order=desc){:target="_blank"}) to spin up your instances. Amazon has various [instance types](https://aws.amazon.com/ec2/instance-types/){:target="_blank"}, each of which are configured for specific use cases. For PyTorch, it is highly recommended that you use the accelerated computing instances that feature GPUs or custom AI/ML accelerators as they are tailored for the high compute needs of machine learning.

In order to use AWS, you need to set up an [AWS account](https://aws.amazon.com/getting-started/){:target="_blank"}, if you do not have one already. You will create a username (your email address), password and an AWS account name (since you can create multiple AWS accounts for different purposes). You will also provide contact and billing information. The billing information is important because while AWS does provide what they call “free-tier” instances, to use PyTorch you will want more powerful, paid instances.

Once you are logged in, you will be brought to your [AWS console](https://aws.amazon.com/console/){:target="_blank"}. You can even learn more about AWS through a set of [simple tutorials](https://aws.amazon.com/getting-started/tutorials/){:target="_blank"}.

### AWS Inferentia-based instances

[AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/){:target="_blank"} is a chip custom built by AWS to provide higher performance and low cost machine learning inference in the cloud. [Amazon EC2 Inf1 instances](https://aws.amazon.com/ec2/instance-types/inf1/){:target="_blank"} feature up to 16 AWS Inferentia chips, the latest second generation Intel Xeon Scalable processors, and up to 100 Gbps networking to enable high throughput and lowest cost inference in the cloud. You can use Inf1 instances with Amazon SageMaker for a fully managed workflow, or use the [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/){:target="_blank"} directly which is integrated with PyTorch.

### GPU-based instances

[Amazon EC2 P4d instances](https://aws.amazon.com/ec2/instance-types/p4/) deliver the highest performance for machine learning training on AWS. They are powered by the latest NVIDIA A100 Tensor Core GPUs and feature first in the cloud 400 Gbps instance networking. P4d instances are deployed in hyperscale clusters called EC2 UltraClusters that are comprised of more than 4,000 NVIDIA A100 GPUs, Petabit-scale non-blocking networking, and scalable low latency storage with FSx for Lustre. Each EC2 UltraCluster provides supercomputer-class performance to enable you to solve the most complex multi-node ML training tasks.

 For ML inference, AWS Inferentia-based [Inf1](https://aws.amazon.com/ec2/instance-types/inf1/) instances provide the lowest cost inference in the cloud. Additionally, [Amazon EC2 G4dn instances](https://aws.amazon.com/ec2/instance-types/g4/) featuring NVIDIA T4 GPUs are optimized for GPU-based machine learning inference and small scale training that leverage NVIDIA libraries.

### Creating and Launching an Instance

Once you decided upon your instance type, you will need to create, optionally configure and launch your instance. You can connect to your instance from the web browser or a command-line interface. Here are guides for instance launch for various platforms:

* [Linux](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html/){:target="_blank"}
* [Windows](https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/EC2_GetStarted.html){:target="_blank"}
* [Command-line](https://docs.aws.amazon.com/cli/latest/userguide/cli-using-ec2.html){:target="_blank"}

## AWS SageMaker
{: #aws-sagemaker}

With [SageMaker](https://aws.amazon.com/sagemaker) service AWS provides a fully-managed service that allows developers and data scientists to build, train, and deploy machine learning models.

See AWS documentation to learn [how to configure Amazon SageMaker with PyTorch]((https://docs.aws.amazon.com/sagemaker/latest/dg/pytorch.html)).

## Pre-Built AMIs
{: #aws-amis}

AWS provides instances (called AWS Deep Learning AMIs) pre-built with a modern version of PyTorch. The available AMIs are:

* Ubuntu
* Amazon Linux
* Windows 2016

Amazon has written a good [blog post](https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/){:target="_blank"} on getting started with pre-built AMI.

## Installing PyTorch From Scratch
{: #aws-from-scratch}

You may prefer to start with a bare instance to install PyTorch. Once you have connected to your instance, setting up PyTorch is the same as [setting up locally](/get-started) for your operating system of choice.
