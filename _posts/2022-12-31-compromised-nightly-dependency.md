---
layout: blog_detail
title: "Compromised PyTorch-nightly dependency chain between December 25th and December 30th, 2022."
author:  The PyTorch Team
---

If you installed PyTorch-nightly on Linux via pip between December 25, 2022 and December 30, 2022, please uninstall it and torchtriton immediately, and use the latest nightly binaries (newer than Dec 30th 2022).

```bash
$ pip3 uninstall -y torch torchvision torchaudio torchtriton
$ pip3 cache purge
```

PyTorch-nightly Linux packages installed via pip during that time installed a dependency, torchtriton, which was compromised on the Python Package Index (PyPI) code repository and ran a malicious binary. This is what is known as a supply chain attack and directly affects dependencies for packages that are hosted on public package indices.

**NOTE:** Users of the PyTorch **stable** packages **are not** affected by this issue.


## How to check if your Python environment is affected

The following command searches for the malicious binary in the torchtriton package (`PYTHON_SITE_PACKAGES/triton/runtime/triton`) and prints out whether your current Python environment is affected or not.

```bash
python3 -c "import pathlib;import importlib.util;s=importlib.util.find_spec('triton'); affected=any(x.name == 'triton' for x in (pathlib.Path(s.submodule_search_locations[0] if s is not None else '/' ) / 'runtime').glob('*'));print('You are {}affected'.format('' if affected else 'not '))"
```

The malicious binary is executed when the triton package is imported, which requires explicit code to do and is not PyTorch’s default behavior.

## The Background

At around 4:40pm GMT on December 30 (Friday), we learned about a malicious dependency package (`torchtriton`) that was uploaded to the Python Package Index (PyPI) code repository with the same package name as the one we ship on the [PyTorch nightly package index](https://download.pytorch.org/whl/nightly). Since the [PyPI index takes precedence](https://github.com/pypa/pip/issues/8606), this malicious package was being installed instead of the version from our official repository. This design enables somebody to register a package by the same name as one that exists in a third party index, and pip will install their version by default.

This malicious package has the same name `torchtriton` but added in code that uploads sensitive data from the machine.


## What we know

torchtriton on PyPI contains a malicious triton binary which is installed at `PYTHON_SITE_PACKAGES/triton/runtime/triton`. Its SHA256 hash is listed below.

`SHA256(triton)= 2385b29489cd9e35f92c072780f903ae2e517ed422eae67246ae50a5cc738a0e`

The binary’s main function does the following:

- Get system information
  - nameservers from `/etc/resolv.conf`
  - hostname from `gethostname()`
  - current username from `getlogin()`
  - current working directory name from `getcwd()`
  - environment variables
- Read the following files
  - `/etc/hosts`
  - `/etc/passwd`
  - The first 1,000 files in `$HOME/*`
  - `$HOME/.gitconfig`
  - `$HOME/.ssh/*`
- Upload all of this information, including file contents, via encrypted DNS queries to the domain *.h4ck[.]cfd, using the DNS server wheezy[.]io

The binary’s file upload functionality is limited to files less than 99,999 bytes in size. It also uploads only the first 1,000 files in $HOME (but all files < 99,999 bytes in the .ssh directory).

## Steps taken towards mitigation

- torchtriton has been removed as a dependency for our nightly packages and replaced with pytorch-triton ([pytorch/pytorch#91539](https://github.com/pytorch/pytorch/pull/91539)) and a dummy package registered on PyPI (so that this issue doesn’t repeat)
- All nightly packages that depend on torchtriton have been removed from our package indices at https://download.pytorch.org until further notice
- We have reached out to the PyPI security team to get proper ownership of the `torchtriton` package on PyPI and to delete the malicious version



