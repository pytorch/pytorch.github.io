# Using PyTorch with Google Cloud
{:.no_toc}

To gain the full experience of what PyTorch has to offer, a machine with at least one dedicated NVIDIA GPU is necessary. While it is not always practical to have your own machine with these specifications, there are our cloud based solutions to allow you to test and use PyTorch's full features.

Google Cloud [provides](https://cloud.google.com/products/){:target="_blank"} both:

* dedicated, pre-built [machine learning platforms](https://cloud.google.com/products/ai/){:target="_blank"}, complete with PyTorch
* bare [Linux and Windows virtual machines](https://cloud.google.com/compute/){:target="_blank"} for you to do a custom install of PyTorch.

## Google Cloud Primer
{: #google-cloud-primer}

In order to use Google Cloud, you need to set up an [Google account](https://accounts.google.com/){:target="_blank"}, if you do not have one already. You will create a username (typically an `@gmail.com` email address) and password. After words, you will be able to [try Google Cloud](https://console.cloud.google.com/freetrial){:target="_blank"}. You will also provide contact and billing information. The billing information is initially used to prove you are a real person. And then, after your trial, you can choose to upgrade to a paid account.

Once you are logged in, you will be brought to your [Google Cloud console](https://console.cloud.google.com/){:target="_blank"}.  You can even learn more about Google Cloud through a set of [simple tutorials](https://console.cloud.google.com/getting-started){:target="_blank"}.

### Cloud Deep Learning VM Image

Google Cloud provides no setup required, pre-configured virtual machines to help you build your deep learning projects. [Cloud Deep Learning VM Image](https://cloud.google.com/deep-learning-vm-image/){:target="_blank"} is a set of Debian-based virtual machines that allow you to [build and run](https://cloud.google.com/deep-learning-vm/docs/) machine PyTorch learning based applications.

### GPU-based Virtual Machines
{: #google-cloud-gpu-based-virtual-machines}

For custom virtual machines, generally you will want to use [Compute Engine Virtual Machine instances](https://cloud.google.com/compute/){:target="_blank"}), with GPU enabled, to build with PyTorch. Google has [various virtual machine types](https://console.cloud.google.com/compute/instances) and pricing options, with both [Linux](https://cloud.google.com/compute/docs/quickstart-linux){:target="_blank"} and [Windows](https://cloud.google.com/compute/docs/quickstart-windows){:target="_blank"}, all of which can be configured for specific use cases. For PyTorch, it is highly recommended that you use a [GPU-enabled](https://cloud.google.com/compute/docs/gpus/add-gpus){:target="_blank"} virtual machines. They are tailored for the high compute needs of machine learning.

The expense of your virtual machine is directly correlated to the number of GPUs that it contains. One NVIDIA Tesla P100 virtual machine, for example, can actually be suitable for many use cases.

### Deep Learning Containers

Google Cloud also offers pre-configured and optimized Deep Learning Containers. They provide a consistent environment across Google Cloud services, making it easy to scale in the cloud or shift from on-premises. You have the flexibility to deploy on Google Kubernetes Engine (GKE), AI Platform, Cloud Run, Compute Engine, Kubernetes, and Docker Swarm.

## Installing PyTorch From Scratch
{: #google-cloud-from-scratch}

You may prefer to start with a bare instance to install PyTorch. Once you have connected to your instance, setting up PyTorch is the same as [setting up locally](/get-started) for your operating system of choice.
