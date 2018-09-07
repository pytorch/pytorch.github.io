# Pytorch Usage Details: AWS

To gain the full experience of what PyTorch has to offer, a machine with at least one dedicated NVIDA GPU is necessary. While it is not always practical to have your own machine with these specifications, there are our cloud based solutions to allow you to test and use PyTorch's full features.

AWS [provides](https://aws.amazon.com/machine-learning/amis/){:target="_blank"} both:

* dedicated, pre-built machine learning instances, complete with PyTorch
* bare Linux and Windows instances for you to do a custom install of PyTorch.

## AWS Primer

In order to use AWS, you need to set up an [AWS account](https://aws.amazon.com/getting-started/){:target="_blank"}, if you do not have one already. You will create a username (your email address), password and an AWS account name (since you can create multiple AWS accounts for different purposes). You will also provide contact and billing information. The billing information is important because while AWS does provide what they call “free-tier” instances, to use PyTorch you will want more powerful, paid instances.

Once you are logged in, you will be brought to your [AWS console](https://aws.amazon.com/console/){:target="_blank"}.  You can even learn more about AWS through a set of [simple tutorials](https://aws.amazon.com/getting-started/tutorials/){:target="_blank"}.

### GPU-based instances

Generally, you will be using Amazon Elastic Compute Cloud (or [EC2](https://aws.amazon.com/ec2/){:target="_blank"}) to spin up your instances. Amazon has various [instance types](https://aws.amazon.com/ec2/instance-types/){:target="_blank"}, each of which are configured for specific use cases. For PyTorch, it is highly recommended that you use the accelerated computing, or [p3](https://aws.amazon.com/ec2/instance-types/p3/){:target="_blank"}, instances. They are tailored for the high compute needs of machine learning.

The expense of your instance is directly correlated to the number of GPUs that it contains. The p3.2xlarge instance is the smallest, cheapest instance and can actually be suitable for many use cases.

### Creating and Launching an Instance

Once you decided upon your instance type, you will need to create, optionally configure and launch your instance. You can connect to your instance from the web browser or a command-line interface. Here are guides for instance launch for various platforms:

* [Linux](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html/){:target="_blank"}
* [Windows](https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/EC2_GetStarted.html){:target="_blank"}
* [Command-line](https://docs.aws.amazon.com/cli/latest/userguide/cli-using-ec2.html){:target="_blank"}

## Pre-Built AMIs

AWS provides instances (called AWS Deep Learning AMIs) pre-built with a modern version of PyTorch. The available AMIs are:

* Ubuntu
* Amazon Linux
* Windows 2016

Amazon has written a good [blog post](https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/){:target="_blank"} on getting started with pre-built AMI.

## Installing PyTorch From Scratch

You may prefer to start with a bare instance to install PyTorch. Once you have connected to your instance, setting up PyTorch is the same as [setting up locally](get-started) for your operating system of choice.