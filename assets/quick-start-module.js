// Keys are Substrings as diplayed by navigator.platform
var supportedOperatingSystems = new Map([
  ['linux', 'linux'],
  ['mac', 'macos'],
  ['win', 'windows'],
]);

var supportedComputePlatforms = new Map([
  ['cuda10.2', new Set(['linux', 'windows'])],
  ['cuda11.x', new Set(['linux', 'windows'])],
  ['rocm4.2', new Set(['linux'])],
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
  'alibaba',
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

// Change CUDA version depending on build type
function changeCUDAVersion(ptbuild) {
  var element = document.getElementById("cuda11.x");
  if (element == null) {
    console.log("Failed to find cuda11.x element");
    return;
  }
  if (element.childElementCount != 1) {
    console.log("Unexpected number of children for cuda11.x element");
    return;
  }
  if (ptbuild == "preview" || ptbuild == "stable") {
    element.children[0].textContent = "CUDA 11.3";
  } else {
    element.children[0].textContent = "CUDA 11.1";
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
  var object = {"preview,pip,linux,accnone,python": "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html", "preview,pip,linux,cuda10.2,python": "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html", "preview,pip,linux,cuda11.x,python": "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html", "preview,pip,linux,rocm4.2,python": "pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/rocm4.2/torch_nightly.html", "preview,conda,linux,cuda10.2,python": "conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly", "preview,conda,linux,cuda11.x,python": "conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly", "preview,conda,linux,rocm4.2,python": "<b>NOTE:</b> Conda packages are not currently available for ROCm, please use pip instead<br />", "preview,conda,linux,accnone,python": "conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly", "preview,libtorch,linux,accnone,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,libtorch,linux,cuda10.2,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu102/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu102/libtorch-shared-with-deps-latest.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu102/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu102/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,libtorch,linux,cuda11.x,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu1133/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu1133/libtorch-shared-with-deps-latest.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu113/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu113/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,libtorch,linux,rocm4.2,cplusplus": "LibTorch binaries are not available for ROCm, please build it from source", "preview,pip,macos,cuda10.2,python": "# On MacOS, we provide CPU-only packages, CUDA functionality is not provided<br />pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html", "preview,pip,macos,cuda11.x,python": "# On MacOS, we provide CPU-only packages, CUDA functionality is not provided<br />pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html", "preview,pip,macos,rocm4.2,python": "# ROCm is not available on MacOS<br />pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html", "preview,pip,macos,accnone,python": "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html", "preview,conda,macos,cuda10.2,python": "conda install pytorch torchvision torchaudio -c pytorch-nightly", "preview,conda,macos,cuda11.x,python": "conda install pytorch torchvision torchaudio -c pytorch-nightly", "preview,conda,macos,rocm4.2,python": "<b>NOTE:</b> ROCm is not available on MacOS", "preview,conda,macos,accnone,python": "conda install pytorch torchvision torchaudio -c pytorch-nightly", "preview,libtorch,macos,accnone,cplusplus": "Download here:<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip</a>", "preview,libtorch,macos,cuda10.2,cplusplus": "MacOS binaries do not support CUDA. Download CPU libtorch here:<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip</a>", "preview,libtorch,macos,cuda11.x,cplusplus": "MacOS binaries do not support CUDA. Download CPU libtorch here:<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip</a>", "preview,libtorch,macos,rocm4.2,cplusplus": "ROCm is not available on MacOS. Download CPU libtorch here:<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip</a>", "preview,pip,windows,accnone,python": "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html", "preview,pip,windows,cuda10.2,python": "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html", "preview,pip,windows,cuda11.x,python": "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html", "preview,pip,windows,rocm4.2,python": "<b>NOTE:</b> ROCm is not available on Windows", "preview,conda,windows,cuda10.2,python": "conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly", "preview,conda,windows,cuda11.x,python": "conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly", "preview,conda,windows,rocm4.2,python": "<b>NOTE:</b> ROCm is not available on Windows", "preview,conda,windows,accnone,python": "conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly", "preview,libtorch,windows,accnone,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-debug-latest.zip</a>", "preview,libtorch,windows,cuda10.2,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu102/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu102/libtorch-win-shared-with-deps-latest.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu102/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu102/libtorch-win-shared-with-deps-debug-latest.zip</a>", "preview,libtorch,windows,cuda11.x,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu113/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu113/libtorch-win-shared-with-deps-latest.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu113/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu113/libtorch-win-shared-with-deps-debug-latest.zip</a>", "preview,libtorch,windows,rocm4.2,cplusplus": "<b>NOTE:</b> ROCm is not available on Windows", "lts,pip,linux,accnone,python": "pip3 install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html", "lts,pip,linux,cuda10.2,python": "pip3 install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html", "lts,pip,linux,cuda11.x,python": "pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html", "lts,pip,linux,rocm4.2,python": "<b>NOTE:</b> ROCm is not supported in LTS", "lts,conda,linux,cuda10.2,python": "conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts", "lts,conda,linux,cuda11.x,python": "<b>NOTE:</b> 'nvidia' channel is required for cudatoolkit 11.1<br />conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia", "lts,conda,linux,rocm4.2,python": "<b>NOTE:</b> ROCm is not supported in LTS", "lts,conda,linux,accnone,python": "conda install pytorch torchvision torchaudio cpuonly -c pytorch-lts", "lts,libtorch,linux,accnone,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-shared-with-deps-1.8.2%2Bcpu.zip'>https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-shared-with-deps-1.8.2%2Bcpu.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcpu.zip'>https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcpu.zip</a>", "lts,libtorch,linux,cuda10.2,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-shared-with-deps-1.8.2%2Bcu102.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-shared-with-deps-1.8.2%2Bcu102.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu102.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu102.zip</a>", "lts,libtorch,linux,cuda11.x,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-shared-with-deps-1.8.2%2Bcu111.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-shared-with-deps-1.8.2%2Bcu111.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu111.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu111.zip</a>", "lts,libtorch,linux,rocm4.2,cplusplus": "<b>NOTE:</b> ROCm is not supported in LTS", "lts,pip,macos,cuda10.2,python": "# macOS is not currently supported for lts", "lts,pip,macos,cuda11.x,python": "# macOS is not currently supported for lts", "lts,pip,macos,rocm4.2,python": "# macOS is not currently supported for lts", "lts,pip,macos,accnone,python": "# macOS is not currently supported for lts", "lts,conda,macos,cuda10.2,python": "# macOS is not currently supported for lts", "lts,conda,macos,cuda11.x,python": "# macOS is not currently supported for lts", "lts,conda,macos,rocm4.2,python": "# macOS is not currently supported for lts", "lts,conda,macos,accnone,python": "# macOS is not currently supported for lts", "lts,libtorch,macos,accnone,cplusplus": "# macOS is not currently supported for lts", "lts,libtorch,macos,cuda10.2,cplusplus": "# macOS is not currently supported for lts", "lts,libtorch,macos,cuda11.x,cplusplus": "# macOS is not currently supported for lts", "lts,libtorch,macos,rocm4.2,cplusplus": "# macOS is not currently supported for lts", "lts,pip,windows,accnone,python": "pip3 install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html", "lts,pip,windows,cuda10.2,python": "pip3 install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html", "lts,pip,windows,cuda11.x,python": "pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html", "lts,pip,windows,rocm4.2,python": "<b>NOTE:</b> ROCm is not supported in LTS", "lts,conda,windows,cuda10.2,python": "conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts", "lts,conda,windows,cuda11.x,python": "<b>NOTE:</b> 'conda-forge' channel is required for cudatoolkit 11.1<br />conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge", "lts,conda,windows,rocm4.2,python": "<b>NOTE:</b> ROCm is not supported in LTS", "lts,conda,windows,accnone,python": "conda install pytorch torchvision torchaudio cpuonly -c pytorch-lts", "lts,libtorch,windows,accnone,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-win-shared-with-deps-1.8.2%2Bcpu.zip'>https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-win-shared-with-deps-1.8.2%2Bcpu.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-win-shared-with-deps-debug-1.8.2%2Bcpu.zip'>https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-win-shared-with-deps-debug-1.8.2%2Bcpu.zip</a>", "lts,libtorch,windows,cuda10.2,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-win-shared-with-deps-1.8.2%2Bcu102.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-win-shared-with-deps-1.8.2%2Bcu102.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-win-shared-with-deps-debug-1.8.2%2Bcu102.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu102/libtorch-win-shared-with-deps-debug-1.8.2%2Bcu102.zip</a>", "lts,libtorch,windows,cuda11.x,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-win-shared-with-deps-1.8.2%2Bcu111.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-win-shared-with-deps-1.8.2%2Bcu111.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-win-shared-with-deps-debug-1.8.2%2Bcu111.zip'>https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-win-shared-with-deps-debug-1.8.2%2Bcu111.zip</a>", "lts,libtorch,windows,rocm4.2,cplusplus": "<b>NOTE:</b> ROCm is not supported in LTS", "stable,pip,linux,accnone,python": "pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html", "stable,pip,linux,cuda10.2,python": "pip3 install torch torchvision torchaudio", "stable,pip,linux,cuda11.x,python": "pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html", "stable,pip,linux,rocm4.2,python": "pip3 install torch torchvision==0.11.1 -f https://download.pytorch.org/whl/rocm4.2/torch_stable.html", "stable,conda,linux,cuda10.2,python": "conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch", "stable,conda,linux,cuda11.x,python": "conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch", "stable,conda,linux,rocm4.2,python": "<b>NOTE:</b> Conda packages are not currently available for ROCm, please use pip instead<br />", "stable,conda,linux,accnone,python": "conda install pytorch torchvision torchaudio cpuonly -c pytorch", "stable,libtorch,linux,accnone,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.10.0%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.10.0%2Bcpu.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcpu.zip</a>", "stable,libtorch,linux,cuda10.2,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.10.0%2Bcu102.zip'>https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.10.0%2Bcu102.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcu102.zip'>https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcu102.zip</a>", "stable,libtorch,linux,cuda11.x,cplusplus": "Download here (Pre-cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.10.0%2Bcu113.zip'>https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.10.0%2Bcu113.zip</a><br />Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcu113.zip'>https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcu113.zip</a>", "stable,libtorch,linux,rocm4.2,cplusplus": "LibTorch binaries are not available for ROCm, please build it from source", "stable,pip,macos,cuda10.2,python": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed<br />pip3 install torch torchvision torchaudio", "stable,pip,macos,cuda11.x,python": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed<br />pip3 install torch torchvision torchaudio", "stable,pip,macos,rocm4.2,python": "<b>NOTE:</b> ROCm is not available on MacOS", "stable,pip,macos,accnone,python": "pip3 install torch torchvision torchaudio", "stable,conda,macos,cuda10.2,python": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed<br />conda install pytorch torchvision torchaudio -c pytorch", "stable,conda,macos,cuda11.x,python": "# MacOS Binaries dont support CUDA, install from source if CUDA is needed<br />conda install pytorch torchvision torchaudio -c pytorch", "stable,conda,macos,rocm4.2,python": "<b>NOTE:</b> ROCm is not available on MacOS", "stable,conda,macos,accnone,python": "conda install pytorch torchvision torchaudio -c pytorch", "stable,libtorch,macos,accnone,cplusplus": "Download here:<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.10.0.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.10.0.zip</a>", "stable,libtorch,macos,cuda10.2,cplusplus": "MacOS binaries do not support CUDA. Download CPU libtorch here:<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.10.0.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.10.0.zip</a>", "stable,libtorch,macos,cuda11.x,cplusplus": "MacOS binaries do not support CUDA. Download CPU libtorch here:<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.10.0.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.10.0.zip</a>", "stable,libtorch,macos,rocm4.2,cplusplus": "<b>NOTE:</b> ROCm is not available on MacOS", "stable,pip,windows,accnone,python": "pip3 install torch torchvision torchaudio", "stable,pip,windows,cuda10.2,python": "pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html", "stable,pip,windows,cuda11.x,python": "pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html", "stable,pip,windows,rocm4.2,python": "<b>NOTE:</b> ROCm is not available on Windows", "stable,conda,windows,cuda10.2,python": "conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch", "stable,conda,windows,cuda11.x,python": "conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch", "stable,conda,windows,rocm4.2,python": "<b>NOTE:</b> ROCm is not available on Windows", "stable,conda,windows,accnone,python": "conda install pytorch torchvision torchaudio cpuonly -c pytorch", "stable,libtorch,windows,accnone,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.10.0%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.10.0%2Bcpu.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.10.0%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.10.0%2Bcpu.zip</a>", "stable,libtorch,windows,cuda10.2,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-1.10.0%2Bcu102.zip'>https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-1.10.0%2Bcu102.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-debug-1.10.0%2Bcu102.zip'>https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-debug-1.10.0%2Bcu102.zip</a>", "stable,libtorch,windows,cuda11.x,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-1.10.0%2Bcu113.zip'>https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-1.10.0%2Bcu113.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-debug-1.10.0%2Bcu113.zip'>https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-debug-1.10.0%2Bcu113.zip</a>", "stable,libtorch,windows,rocm4.2,cplusplus": "<b>NOTE:</b> ROCm is not available on Windows"};
  var lts_notice = "<div class='alert-secondary'><b>Note</b>: Additional support for these binaries may be provided by <a href='/enterprise-support-program' style='font-size:100%'>PyTorch Enterprise Support Program Participants</a>.</div>";

  if (!object.hasOwnProperty(key)) {
    $("#command").html(
      "<pre> # Follow instructions at this URL: https://github.com/pytorch/pytorch#from-source </pre>"
    );
  } else if (key.indexOf("lts") == 0  && key.indexOf('rocm') < 0) {
    $("#command").html("<pre>" + object[key] + "</pre>" + lts_notice);
  } else {
    $("#command").html("<pre>" + object[key] + "</pre>");
  }
}

