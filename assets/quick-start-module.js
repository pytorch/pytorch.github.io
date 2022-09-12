// Keys are Substrings as diplayed by navigator.platform
var supportedOperatingSystems = new Map([
  ['linux', 'linux'],
  ['mac', 'macos'],
  ['win', 'windows'],
]);

var supportedComputePlatforms = new Map([
  ['cuda10.2', new Set(['linux', 'windows'])],
  ['cuda11.x', new Set(['linux', 'windows'])],
  ['cuda11.y', new Set(['linux', 'windows'])],
  ['rocm5.x', new Set(['linux'])],
  ['accnone', new Set(['linux', 'macos', 'windows'])],
]);

var default_selected_os = getAnchorSelectedOS() || getDefaultSelectedOS();
var opts = {
  cuda: getPreferredCuda(default_selected_os),
  os: default_selected_os,
  pm: 'conda',
  language: 'python',
  ptbuild: 'stable',
};

var supportedCloudPlatforms = [
  'aws',
  'google-cloud',
  'microsoft-azure',
];

var os = $(".os > .option");
var package = $(".package > .option");
var language = $(".language > .option");
var cuda = $(".cuda > .option");
var ptbuild = $(".ptbuild > .option");

os.on("click", function() {
  selectedOption(os, this, "os");
});
package.on("click", function() {
  selectedOption(package, this, "pm");
});
language.on("click", function() {
  selectedOption(language, this, "language");
});
cuda.on("click", function() {
  selectedOption(cuda, this, "cuda");
});
ptbuild.on("click", function() {
  selectedOption(ptbuild, this, "ptbuild")
});

// Pre-select user's operating system
$(function() {
  var userOsOption = document.getElementById(opts.os);
  var userCudaOption = document.getElementById(opts.cuda);
  if (userOsOption) {
    $(userOsOption).trigger("click");
  }
  if (userCudaOption) {
    $(userCudaOption).trigger("click");
  }
});


// determine os (mac, linux, windows) based on user's platform
function getDefaultSelectedOS() {
  var platform = navigator.platform.toLowerCase();
  for (var [navPlatformSubstring, os] of supportedOperatingSystems.entries()) {
    if (platform.indexOf(navPlatformSubstring) !== -1) {
      return os;
    }
  }
  // Just return something if user platform is not in our supported map
  return supportedOperatingSystems.values().next().value;
}

// determine os based on location hash
function getAnchorSelectedOS() {
  var anchor = location.hash;
  var ANCHOR_REGEX = /^#[^ ]+$/;
  // Look for anchor in the href
  if (!ANCHOR_REGEX.test(anchor)) {
    return false;
  }
  // Look for anchor with OS in the first portion
  var testOS = anchor.slice(1).split("-")[0];
  for (var [navPlatformSubstring, os] of supportedOperatingSystems.entries()) {
    if (testOS.indexOf(navPlatformSubstring) !== -1) {
      return os;
    }
  }
  return false;
}

// determine CUDA version based on OS
function getPreferredCuda(os) {
  // Only CPU builds are currently available for MacOS
  if (os == 'macos') {
    return 'accnone';
  }
  return 'cuda10.2';
}

// Disable compute platform not supported on OS
function disableUnsupportedPlatforms(os) {
  supportedComputePlatforms.forEach( (oses, platform, arr) => {
    var element = document.getElementById(platform);
    if (element == null) {
      console.log("Failed to find element for platform " + platform);
      return;
    }
    var supported = oses.has(os);
    element.style.textDecoration = supported ? "" : "line-through";
  });
}

// Change compute versions depending on build type
function changeCUDAVersion(ptbuild) {
  var cuda_element_x = document.getElementById("cuda11.x");
  var cuda_element_y = document.getElementById("cuda11.y");
  var rocm_element = document.getElementById("rocm5.x");
  if (cuda_element == null) {
    console.log("Failed to find cuda11.x element");
    return;
  }
  if (cuda_element.childElementCount != 1) {
    console.log("Unexpected number of children for cuda11.x element");
    return;
  }
  if (rocm_element == null) {
    console.log("Failed to find rocm5.x element");
    return;
  }
  if (rocm_element.childElementCount != 1) {
    console.log("Unexpected number of children for rocm5.x element");
    return;
  }
  if (ptbuild == "preview") {
    rocm_element.children[0].textContent = "ROCm 5.2";
    cuda_element_x.children[0].textContent = "CUDA 11.6";
    cuda_element_y.children[0].textContent = "CUDA 11.7";
  } else if (ptbuild == "stable") {
    rocm_element.children[0].textContent = "ROCm 5.1.1";
    cuda_element_x.children[0].textContent = "CUDA 11.3";
    cuda_element_y.children[0].textContent = "CUDA 11.6";
  } else {
    rocm_element.children[0].textContent = "ROCm 5.1.1";
    cuda_element_x.children[0].textContent = "CUDA 11.3";
    cuda_element_y.children[0].textContent = "CUDA 11.6";
  }
}

// Change accnone name depending on OS type
function changeAccNoneName(osname) {
  var accnone_element = document.getElementById("accnone");
  if (accnone_element == null) {
    console.log("Failed to find accnone element");
    return;
  }
  if (osname == "macos") {
    accnone_element.children[0].textContent = "Default";
  } else {
    accnone_element.children[0].textContent = "CPU";
  }
}

function selectedOption(option, selection, category) {
  $(option).removeClass("selected");
  $(selection).addClass("selected");
  opts[category] = selection.id;
  if (category === "pm") {
    var elements = document.getElementsByClassName("language")[0].children;
    if (selection.id !== "libtorch" && elements["cplusplus"].classList.contains("selected")) {
      $(elements["cplusplus"]).removeClass("selected");
      $(elements["python"]).addClass("selected");
      opts["language"] = "python";
    } else if (selection.id == "libtorch") {
      for (var i = 0; i < elements.length; i++) {
        if (elements[i].id === "cplusplus") {
          $(elements[i]).addClass("selected");
          opts["language"] = "cplusplus";
        } else {
          $(elements[i]).removeClass("selected");
        }
      }
    }
  } else if (category === "language") {
    var elements = document.getElementsByClassName("package")[0].children;
    if (selection.id !== "cplusplus" && elements["libtorch"].classList.contains("selected")) {
      $(elements["libtorch"]).removeClass("selected");
      $(elements["pip"]).addClass("selected");
      opts["pm"] = "pip";
    } else if (selection.id == "cplusplus") {
      for (var i = 0; i < elements.length; i++) {
        if (elements[i].id === "libtorch") {
          $(elements[i]).addClass("selected");
          opts["pm"] = "libtorch";
        } else {
          $(elements[i]).removeClass("selected");
        }
      }
    }
  } else if (category == "ptbuild") {
    changeCUDAVersion(opts.ptbuild);
  }
  commandMessage(buildMatcher());
  if (category === "os") {
    disableUnsupportedPlatforms(opts.os);
    changeAccNoneName(opts.os);
    display(opts.os, 'installation', 'os');
  }
}

function display(selection, id, category) {
  var container = document.getElementById(id);
  // Check if there's a container to display the selection
  if (container === null) {
    return;
  }
  var elements = container.getElementsByClassName(category);
  for (var i = 0; i < elements.length; i++) {
    if (elements[i].classList.contains(selection)) {
      $(elements[i]).addClass("selected");
    } else {
      $(elements[i]).removeClass("selected");
    }
  }
}

function buildMatcher() {
  return (
    opts.ptbuild.toLowerCase() +
    "," +
    opts.pm.toLowerCase() +
    "," +
    opts.os.toLowerCase() +
    "," +
    opts.cuda.toLowerCase() +
    "," +
    opts.language.toLowerCase()
  );
}

// Cloud Partners sub-menu toggle listeners
$("[data-toggle='cloud-dropdown']").on("click", function(e) {
  if ($(this).hasClass("open")) {
    $(this).removeClass("open");
    // If you deselect a current drop-down item, don't display it's info any longer
    display(null, 'cloud', 'platform');
  } else {
    $("[data-toggle='cloud-dropdown'].open").removeClass("open");
    $(this).addClass("open");
    var cls = $(this).find(".cloud-option-body")[0].className;
    for (var i = 0; i < supportedCloudPlatforms.length; i++) {
      if (cls.includes(supportedCloudPlatforms[i])) {
        display(supportedCloudPlatforms[i], 'cloud', 'platform');
      }
    }
  }
});

function commandMessage(key) {
  var object = {"preview,pip,linux,accnone,python": "pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu", "preview,pip,linux,cuda10.2,python": "pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu102", "preview,pip,linux,cuda11.x,python": "pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116", "preview,pip,linux,cuda11.y,python": "pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu117", "preview,pip,linux,rocm5.x,python": "pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.2/", "preview,conda,linux,cuda10.2,python": "conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly", "preview,conda,linux,cuda11.x,python": "conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch-nightly", "preview,conda,linux,cuda11.y,python": "conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch-nightly -c nvidia", "preview,conda,linux,rocm5.x,python": "<b>NOTE:</b> Conda packages are not currently available for ROCm, please use pip instead<br />", "preview,conda,linux,accnone,python": "conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly", "preview,libtorch,linux,accnone,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,libtorch,linux,cuda10.2,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu102/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu102/libtorch-shared-with-deps-latest.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu102/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu102/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,libtorch,linux,cuda11.x,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu116/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu116/libtorch-shared-with-deps-latest.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu116/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu116/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,libtorch,linux,cuda11.y,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu117/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu117/libtorch-shared-with-deps-latest.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu117/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu117/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,libtorch,linux,rocm5.x,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/rocm5.2/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/rocm5.2/libtorch-shared-with-deps-latest.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/rocm5.2/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/rocm5.2/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,pip,macos,cuda10.2,python": "# CUDA is not available on MacOS, please use default package<br />pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu", "preview,pip,macos,cuda11.x,python": "# CUDA is not available on MacOS, please use default package<br />pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu", "preview,pip,macos,cuda11.y,python": "# CUDA is not available on MacOS, please use default package<br />pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu", "preview,pip,macos,rocm5.x,python": "# ROCm is not available on MacOS, please use default package<br />pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu", "preview,pip,macos,accnone,python": "# MPS acceleration is available on MacOS 12.3+<br />pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu", "preview,conda,macos,cuda10.2,python": "# CUDA is not available on MacOS, please use default package<br />conda install pytorch torchvision torchaudio -c pytorch-nightly", "preview,conda,macos,cuda11.x,python": "# CUDA is not available on MacOS, please use default package<br />conda install pytorch torchvision torchaudio -c pytorch-nightly", "preview,conda,macos,cuda11.y,python": "# CUDA is not available on MacOS, please use default package<br />conda install pytorch torchvision torchaudio  -c pytorch-nightly", "preview,conda,macos,rocm5.x,python": "# ROCm is not available on MacOS, please use default package<br />conda install pytorch torchvision torchaudio -c pytorch-nightly", "preview,conda,macos,accnone,python": "# MPS acceleration is available on MacOS 12.3+<br />conda install pytorch torchvision torchaudio -c pytorch-nightly", "preview,libtorch,macos,accnone,cplusplus": "Download here:<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip</a>", "preview,libtorch,macos,cuda10.2,cplusplus": "MacOS binaries do not support CUDA. Download default libtorch here:<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip</a>", "preview,libtorch,macos,cuda11.x,cplusplus": "MacOS binaries do not support CUDA. Download default libtorch here:<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip</a>", "preview,libtorch,macos,cuda11.y,cplusplus": "MacOS binaries do not support CUDA. Download CPU libtorch here:<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip</a>", "preview,libtorch,macos,rocm5.x,cplusplus": "ROCm is not available on MacOS. Download default libtorch here:<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip</a>", "preview,pip,windows,accnone,python": "pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu", "preview,pip,windows,cuda10.2,python": "CUDA-10.2 PyTorch builds are no longer available for Windows, please use CUDA-11.6", "preview,pip,windows,cuda11.x,python": "pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116", "preview,pip,windows,cuda11.y,python": "pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu117", "preview,pip,windows,rocm5.x,python": "<b>NOTE:</b> ROCm is not available on Windows", "preview,conda,windows,cuda10.2,python": "CUDA-10.2 PyTorch builds are no longer available for Windows, please use CUDA-11.6", "preview,conda,windows,cuda11.x,python": "conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch-nightly", "preview,conda,windows,cuda11.y,python": "conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch-nightly -c nvidia", "preview,conda,windows,rocm5.x,python": "<b>NOTE:</b> ROCm is not available on Windows", "preview,conda,windows,accnone,python": "conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly", "preview,libtorch,windows,accnone,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-debug-latest.zip</a>", "preview,libtorch,windows,cuda10.2,cplusplus": "CUDA-10.2 PyTorch builds are no longer available for Windows, please use CUDA-11.6", "preview,libtorch,windows,cuda11.x,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu116/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu116/libtorch-win-shared-with-deps-latest.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu116/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu116/libtorch-win-shared-with-deps-debug-latest.zip</a>", "preview,libtorch,windows,cuda11.y,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu117/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu117/libtorch-win-shared-with-deps-latest.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu117/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu117/libtorch-win-shared-with-deps-debug-latest.zip</a>", "preview,libtorch,windows,rocm5.x,cplusplus": "<b>NOTE:</b> ROCm is not available on Windows", "lts,pip,linux,accnone,python": "pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu", "lts,pip,linux,cuda10.2,python": "pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102", "lts,pip,linux,cuda11.x,python": "pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111", "lts,pip,linux,rocm4.x,python": "<b>NOTE:</b> ROCm is not supported in LTS", "lts,conda,linux,cuda10.2,python": "<b>NOTE:</b> Pytorch LTS version 1.8.2 is only supported for Python <= 3.8<br />conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts", "lts,conda,linux,cuda11.x,python": "<b>NOTE:</b> 'nvidia' channel is required for cudatoolkit 11.1 <br> <b>NOTE:</b> Pytorch LTS version 1.8.2 is only supported for Python <= 3.8<br />conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia", "lts,conda,linux,rocm4.x,python": "<b>NOTE:</b> ROCm is not supported in LTS", "lts,conda,linux,accnone,python": "<b>NOTE:</b> Pytorch LTS version 1.8.2 is only supported for Python <= 3.8<br />conda install pytorch torchvision torchaudio cpuonly -c pytorch-lts", "lts,libtorch,linux,accnone,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-shared-with-deps-1.8.2%2Bcpu.zip'>https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-shared-with-deps-1.8.2%2Bcpu.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcpu.zip'>https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcpu.zip</a>", "lts,libtorch,linux,cuda10.2,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-shared-with-deps-1.8.2%2Bcu102.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-shared-with-deps-1.8.2%2Bcu102.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu102.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu102.zip</a>", "lts,libtorch,linux,cuda11.x,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-shared-with-deps-1.8.2%2Bcu111.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-shared-with-deps-1.8.2%2Bcu111.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu111.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu111.zip</a>", "lts,libtorch,linux,rocm4.x,cplusplus": "<b>NOTE:</b> ROCm is not supported in LTS", "lts,pip,macos,cuda10.2,python": "# macOS is not currently supported for lts", "lts,pip,macos,cuda11.x,python": "# macOS is not currently supported for lts", "lts,pip,macos,rocm4.x,python": "# macOS is not currently supported for lts", "lts,pip,macos,accnone,python": "# macOS is not currently supported for lts", "lts,conda,macos,cuda10.2,python": "# macOS is not currently supported for lts", "lts,conda,macos,cuda11.x,python": "# macOS is not currently supported for lts", "lts,conda,macos,rocm4.x,python": "# macOS is not currently supported for lts", "lts,conda,macos,accnone,python": "# macOS is not currently supported for lts", "lts,libtorch,macos,accnone,cplusplus": "# macOS is not currently supported for lts", "lts,libtorch,macos,cuda10.2,cplusplus": "# macOS is not currently supported for lts", "lts,libtorch,macos,cuda11.x,cplusplus": "# macOS is not currently supported for lts", "lts,libtorch,macos,rocm4.x,cplusplus": "# macOS is not currently supported for lts", "lts,pip,windows,accnone,python": "pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu", "lts,pip,windows,cuda10.2,python": "pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102", "lts,pip,windows,cuda11.x,python": "pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111", "lts,pip,windows,rocm4.x,python": "<b>NOTE:</b> ROCm is not supported in LTS", "lts,conda,windows,cuda10.2,python": "<b>NOTE:</b> Pytorch LTS version 1.8.2 is only supported for Python <= 3.8<br />conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts", "lts,conda,windows,cuda11.x,python": "<b>NOTE:</b> 'conda-forge' channel is required for cudatoolkit 11.1 <br> <b>NOTE:</b> Pytorch LTS version 1.8.2 is only supported for Python <= 3.8<br />conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge", "lts,conda,windows,rocm4.x,python": "<b>NOTE:</b> ROCm is not supported in LTS", "lts,conda,windows,accnone,python": "<b>NOTE:</b> Pytorch LTS version 1.8.2 is only supported for Python <= 3.8<br />conda install pytorch torchvision torchaudio cpuonly -c pytorch-lts", "lts,libtorch,windows,accnone,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-win-shared-with-deps-1.8.2%2Bcpu.zip'>https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-win-shared-with-deps-1.8.2%2Bcpu.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-win-shared-with-deps-debug-1.8.2%2Bcpu.zip'>https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-win-shared-with-deps-debug-1.8.2%2Bcpu.zip</a>", "lts,libtorch,windows,cuda10.2,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-win-shared-with-deps-1.8.2%2Bcu102.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-win-shared-with-deps-1.8.2%2Bcu102.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-win-shared-with-deps-debug-1.8.2%2Bcu102.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-win-shared-with-deps-debug-1.8.2%2Bcu102.zip</a>", "lts,libtorch,windows,cuda11.x,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-win-shared-with-deps-1.8.2%2Bcu111.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-win-shared-with-deps-1.8.2%2Bcu111.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-win-shared-with-deps-debug-1.8.2%2Bcu111.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-win-shared-with-deps-debug-1.8.2%2Bcu111.zip</a>", "lts,libtorch,windows,rocm4.x,cplusplus": "<b>NOTE:</b> ROCm is not supported in LTS", "stable,pip,linux,accnone,python": "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu", "stable,pip,linux,cuda10.2,python": "pip3 install torch torchvision torchaudio", "stable,pip,linux,cuda11.x,python": "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113", "stable,pip,linux,cuda11.y,python": "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116", "stable,pip,linux,rocm5.x,python": "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1", "stable,conda,linux,cuda10.2,python": "conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch", "stable,conda,linux,cuda11.x,python": "conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch", "stable,conda,linux,cuda11.y,python": "<b>NOTE:</b> 'conda-forge' channel is required for cudatoolkit 11.6<br />conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge", "stable,conda,linux,rocm5.x,python": "<b>NOTE:</b> Conda packages are not currently available for ROCm, please use pip instead<br />", "stable,conda,linux,accnone,python": "conda install pytorch torchvision torchaudio cpuonly -c pytorch", "stable,libtorch,linux,accnone,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.12.1%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.12.1%2Bcpu.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip</a>", "stable,libtorch,linux,cuda10.2,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.12.1%2Bcu102.zip'>https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.12.1%2Bcu102.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu102.zip'>https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu102.zip</a>", "stable,libtorch,linux,cuda11.x,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.12.1%2Bcu113.zip'>https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.12.1%2Bcu113.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu113.zip'>https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu113.zip</a>", "stable,libtorch,linux,cuda11.y,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu116/libtorch-shared-with-deps-1.12.1%2Bcu116.zip'>https://download.pytorch.org/libtorch/cu116/libtorch-shared-with-deps-1.12.1%2Bcu116.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu116.zip'>https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu116.zip</a>", "stable,libtorch,linux,rocm5.x,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/rocm5.1.1/libtorch-shared-with-deps-1.12.1%2Brocm5.1.1.zip'>https://download.pytorch.org/libtorch/rocm5.1.1/libtorch-shared-with-deps-1.12.1%2Brocm5.1.1.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/rocm5.1.1/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Brocm5.1.1.zip'>https://download.pytorch.org/libtorch/rocm5.1.1/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Brocm5.1.1.zip</a>", "stable,pip,macos,cuda10.2,python": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed<br />pip3 install torch torchvision torchaudio", "stable,pip,macos,cuda11.x,python": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed<br />pip3 install torch torchvision torchaudio", "stable,pip,macos,cuda11.y,python": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed<br />pip3 install torch torchvision torchaudio", "stable,pip,macos,rocm5.x,python": "<b>NOTE:</b> ROCm is not available on MacOS", "stable,pip,macos,accnone,python": "pip3 install torch torchvision torchaudio", "stable,conda,macos,cuda10.2,python": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed<br />conda install pytorch torchvision torchaudio -c pytorch", "stable,conda,macos,cuda11.x,python": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed<br />conda install pytorch torchvision torchaudio -c pytorch", "stable,conda,macos,cuda11.y,python": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed<br />conda install pytorch torchvision torchaudio -c pytorch", "stable,conda,macos,rocm5.x,python": "<b>NOTE:</b> ROCm is not available on MacOS", "stable,conda,macos,accnone,python": "conda install pytorch torchvision torchaudio -c pytorch", "stable,libtorch,macos,accnone,cplusplus": "Download here:<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.12.1.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.12.1.zip</a>", "stable,libtorch,macos,cuda10.2,cplusplus": "MacOS binaries do not support CUDA. Download CPU libtorch here:<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.12.1.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.12.1.zip</a>", "stable,libtorch,macos,cuda11.x,cplusplus": "MacOS binaries do not support CUDA. Download CPU libtorch here:<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.12.1.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.12.1.zip</a>", "stable,libtorch,macos,cuda11.y,cplusplus": "MacOS binaries do not support CUDA. Download CPU libtorch here:<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.12.1.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.12.1.zip</a>", "stable,libtorch,macos,rocm5.x,cplusplus": "<b>NOTE:</b> ROCm is not available on MacOS", "stable,pip,windows,accnone,python": "pip3 install torch torchvision torchaudio", "stable,pip,windows,cuda10.2,python": "CUDA-10.2 PyTorch builds are no longer available for Windows, please use CUDA-11.6", "stable,pip,windows,cuda11.x,python": "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113", "stable,pip,windows,cuda11.y,python": "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116", "stable,pip,windows,rocm5.x,python": "<b>NOTE:</b> ROCm is not available on Windows", "stable,conda,windows,cuda10.2,python": "CUDA-10.2 PyTorch builds are no longer available for Windows, please use CUDA-11.6", "stable,conda,windows,cuda11.x,python": "conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch", "stable,conda,windows,cuda11.y,python": "<b>NOTE:</b> 'conda-forge' channel is required for cudatoolkit 11.6<br />conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge", "stable,conda,windows,rocm5.x,python": "<b>NOTE:</b> ROCm is not available on Windows", "stable,conda,windows,accnone,python": "conda install pytorch torchvision torchaudio cpuonly -c pytorch", "stable,libtorch,windows,accnone,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.12.1%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.12.1%2Bcpu.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.12.1%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.12.1%2Bcpu.zip</a>", "stable,libtorch,windows,cuda10.2,cplusplus": "CUDA-10.2 PyTorch builds are no longer available for Windows, please use CUDA-11.6", "stable,libtorch,windows,cuda11.x,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-1.12.1%2Bcu113.zip'>https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-1.12.1%2Bcu113.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-debug-1.12.1%2Bcu113.zip'>https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-debug-1.12.1%2Bcu113.zip</a>", "stable,libtorch,windows,cuda11.y,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/cu116/libtorch-win-shared-with-deps-1.12.1%2Bcu116.zip'>https://download.pytorch.org/libtorch/cu116/libtorch-win-shared-with-deps-1.12.1%2Bcu116.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/cu116/libtorch-win-shared-with-deps-debug-1.12.1%2Bcu116.zip'>https://download.pytorch.org/libtorch/cu116/libtorch-win-shared-with-deps-debug-1.12.1%2Bcu116.zip</a>", "stable,libtorch,windows,rocm5.x,cplusplus": "<b>NOTE:</b> ROCm is not available on Windows"};

  if (!object.hasOwnProperty(key)) {
    $("#command").html(
      "<pre> # Follow instructions at this URL: https://github.com/pytorch/pytorch#from-source </pre>"
    );
  } else if (key.indexOf("lts") == 0  && key.indexOf('rocm') < 0) {
    $("#command").html("<pre>" + object[key] + "</pre>");
  } else {
    $("#command").html("<pre>" + object[key] + "</pre>");
  }
}

