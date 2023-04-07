---
layout: blog_detail
title: "Deprecation of CUDA 11.6 and Python 3.7 Support"
---

For the upcoming PyTorch 2.0 feature release (target March 2023), we will target CUDA 11.7 as the stable version and CUDA 11.8 as the experimental version of CUDA and Python >=3.8, &lt;=3.11. 

If you are still using or depending on CUDA 11.6 or Python 3.7 builds, we strongly recommend moving to at least CUDA 11.7 and Python 3.8, as it would be the minimum versions required for PyTorch 2.0.

**Please note that as of Feb 1, CUDA 11.6 and Python 3.7  are no longer included in the nightlies**

Please refer to the Release Compatibility Matrix for PyTorch releases:


<table>
  <tr>
   <td><strong>PyTorch Version</strong>
   </td>
   <td><strong>Python</strong>
   </td>
   <td><strong>Stable CUDA</strong>
   </td>
   <td><strong>Experimental CUDA</strong>
   </td>
  </tr>
  <tr>
   <td>2.0
   </td>
   <td>>=3.8, &lt;=3.11
   </td>
   <td>CUDA 11.7, CUDNN 8.5.0.96
   </td>
   <td>CUDA 11.8, CUDNN 8.7.0.84
   </td>
  </tr>
  <tr>
   <td>1.13
   </td>
   <td>>=3.7, &lt;=3.10
   </td>
   <td>CUDA 11.6, CUDNN 8.3.2.44
   </td>
   <td>CUDA 11.7, CUDNN 8.5.0.96
   </td>
  </tr>
  <tr>
   <td>1.12
   </td>
   <td>>=3.7, &lt;=3.10
   </td>
   <td>CUDA 11.3, CUDNN 8.3.2.44
   </td>
   <td>CUDA 11.6, CUDNN 8.3.2.44
   </td>
  </tr>
</table>


As of 2/1/2023

For more information on PyTorch releases, updated compatibility matrix and release policies, please see (and bookmark) [Readme](https://github.com/pytorch/pytorch/blob/master/RELEASE.md#release-compatibility-matrix).

