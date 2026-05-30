# Additional Platforms Integration Guide

## Overview

The PyTorch "Additional Platforms" page allows hardware vendors and platform providers to showcase their PyTorch-compatible accelerators and computing platforms. This guide explains how third-party vendors can add their platforms to the PyTorch "Get Started" page.

## Minimum Requirements

To have your platform added to the PyTorch Additional Platforms page, please ensure:

1. **Stable PyTorch Build:** Your platform must have a stable, nightly, and publicly available PyTorch build
2. **Documentation:** Comprehensive documentation for users
3. **Support Channel:** A way for users to get help (Discord, GitHub issues, forum, etc.)
4. **Active Maintenance:** The platform should be actively maintained and updated for new PyTorch versions
5. **OS support**: Your platforms should support at least Linux system

## Directory Structure

```
pytorch.github.io/
├── _additional_platform/          # JSON configuration files
│   ├── acc1.json                 # Platform configuration
│   ├── acc2.json
│   └── ...
├── _get_started/
│   └── additional_platforms/     # Markdown documentation files
│       ├── acc1.md               # Platform installation guide
│       ├── acc2.md
│       └── ...
└── scripts/
    └── gen_additional_platforms.py  # Generation script
```

## Step-by-Step Guide

### Step 1: Create a JSON Configuration File

Create a JSON file in the `_additional_platforms/` directory. The filename should match your platform ID (e.g., `myplatform.json`).

**Location:** `_additional_platforms/{platform_id}.json`

**Required Fields:**

```json
{
  "name": "Platform Name",
  "support_channel": "https://discord.gg/vendor",
  "stable": {
    "linux": "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/platform/sdk80"
  },
  "preview": {
    "linux": "pip3 install torch torchvision --pre --index-url https://download.pytorch.org/whl/nightly/platform/sdk80"
  }
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Display name of the platform shown in the selector |
| `support_channel` | string | Yes | URL to support channel (Discord, Slack, etc.) |
| `stable` | object | Yes | Installation commands for stable releases |
| `preview` | object | No | Installation commands for preview/nightly releases |

**Installation Commands Structure:**

The `stable` and `preview` objects support the following OS keys:
- `linux` - Linux installation command
- `windows` - Windows installation command

Example with multiple OS support:

```json
{
  "stable": {
    "linux": "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/myplatform/linux",
    "windows": "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/myplatform/windows",
  }
}
```

### Step 2: Create a Markdown Documentation File

Create a Markdown file in the `_get_started/additional_platforms/` directory. The filename must match the platform ID used in the JSON file.

**Location:** `_get_started/additional_platforms/{platform_id}.md`

**Recommended Structure:**

````markdown
# Installing on {Platform Name} Platform

{Brief description of the platform and what it offers.}

## Prerequisites

### Hardware Requirements

* {Hardware requirement 1}
* {Hardware requirement 2}

### Software Requirements

* Python {version range}
* {Platform} SDK {version} or later

## Installation

### pip

Use the pip package manager to install PyTorch with {Platform} support. Select your preferred options in the selector above to get the installation command.

## Verification

To ensure that PyTorch was installed correctly with {Platform} support, run the following code:

```python
import torch
print(torch.__version__)

# Check {Platform} availability
if torch.backends.{platform_id}.is_available():
    print("{Platform} is available!")
    print(f"{Platform} devices: {torch.backends.{platform_id}.device_count()}")
else:
    print("{Platform} is not available.")

```

## Documentation

For more information, please visit the [{Platform} Documentation](https://docs.vendor.com/platform).
````

**Important Notes**: Due to the room space of the page, the markdown file should be no more than 200 lines of code.

### Step 3: Submit a Pull Request

1. Fork the [pytorch.github.io](https://github.com/pytorch/pytorch.github.io) repository
2. Add your JSON and Markdown files in the appropriate directories
3. Submit a pull request with:
   - A clear title describing the platform you're adding
   - A brief description of the platform
   - Contact information for follow-up questions
