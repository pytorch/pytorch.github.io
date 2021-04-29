// Keys are Substrings as diplayed by navigator.platform
var supportedOperatingSystems = new Map([
  ['linux', 'linux'],
  ['mac', 'macos'],
  ['win', 'windows'],
]);

var opts = {
  cuda: 'cuda10.2',
  os: getAnchorSelectedOS() || getDefaultSelectedOS(),
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
  if (userOsOption) {
    $(userOsOption).trigger("click")
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
  }
  commandMessage(buildMatcher());
  if (category === "os") {
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
  var object = {
    "stable,conda,linux,cuda10.2,python":
      "conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch",

    "stable,conda,linux,cuda11.1,python":
      "<b>NOTE:</b> 'nvidia' channel is required for cudatoolkit 11.1<br />conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia",

    "stable,conda,linux,rocm4.0,python":
      "<b>NOTE:</b> Conda packages are not currently available for ROCm, please use pip instead<br />",

    "stable,conda,linux,accnone,python":
      "conda install pytorch torchvision torchaudio cpuonly -c pytorch",

    "stable,conda,macos,cuda10.2,python":
      "conda install pytorch torchvision torchaudio -c pytorch<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,conda,macos,cuda11.1,python":
      "conda install pytorch torchvision torchaudio -c pytorch<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,conda,macos,rocm4.0,python":
      "<b>NOTE:</b> ROCm is not available on MacOS",

    "stable,conda,macos,accnone,python":
      "conda install pytorch torchvision torchaudio -c pytorch",

    "stable,conda,windows,cuda10.2,python":
      "conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch",

    "stable,conda,windows,cuda11.1,python":
      "<b>NOTE:</b> 'nvidia' channel is required for cudatoolkit 11.1<br />conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia",

    "stable,conda,windows,rocm4.0,python":
      "<b>NOTE:</b> ROCm is not available on Windows",

    "stable,conda,windows,accnone,python":
      "conda install pytorch torchvision torchaudio cpuonly -c pytorch",

    "stable,pip,macos,cuda10.2,python":
      "pip3 install torch torchvision torchaudio<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,pip,macos,cuda11.1,python":
      "pip3 install torch torchvision torchaudio<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,pip,macos,rocm4.0,python":
      "<b>NOTE:</b> ROCm is not available on MacOS",

    "stable,pip,macos,accnone,python": "pip3 install torch torchvision torchaudio",

    "stable,pip,linux,accnone,python":
      "pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,linux,cuda10.2,python":
      "pip3 install torch torchvision torchaudio",

    "stable,pip,linux,cuda11.1,python":
      "pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,linux,rocm4.0,python":
      "pip3 install torch -f https://download.pytorch.org/whl/rocm4.0.1/torch_stable.html<br />pip3 install ninja && pip3 install 'git+https://github.com/pytorch/vision.git@v0.9.1'",

    "stable,pip,windows,accnone,python":
      "pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,cuda10.2,python":
      "pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,cuda11.1,python":
      "pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,rocm4.0,python":
      "<b>NOTE:</b> ROCm is not available on Windows",

    "stable,libtorch,linux,accnone,cplusplus":
      "Download here (Pre-cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.8.1%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.8.1%2Bcpu.zip</a><br/><br> Download here (cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip</a>",

    "stable,libtorch,linux,cuda10.2,cplusplus":
      "Download here (Pre-cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.8.1%2Bcu102.zip'>https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.8.1%2Bcu102.zip</a><br/><br> Download here (cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu102.zip'>https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu102.zip</a>",

    "stable,libtorch,linux,cuda11.1,cplusplus":
      "Download here (Pre-cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.8.1%2Bcu111.zip'>https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.8.1%2Bcu111.zip</a><br/><br> Download here (cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu111.zip'>https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu111.zip</a>",

    "stable,libtorch,linux,rocm4.0,cplusplus":
      "LibTorch binaries are not available for ROCm, please build it from source",

    "stable,libtorch,macos,accnone,cplusplus":
      "Download here: <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.1.zip'> https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.1.zip </a>",

    "stable,libtorch,macos,cuda10.2,cplusplus":
      "MacOS binaries do not support CUDA. Download CPU libtorch here: <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.1.zip'> https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.1.zip </a>",

    "stable,libtorch,macos,cuda11.1,cplusplus":
      "MacOS binaries do not support CUDA. Download CPU libtorch here: <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.1.zip'> https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.1.zip </a>",

    "stable,libtorch,macos,rocm4.0,cplusplus":
      "<b>NOTE:</b> ROCm is not available on MacOS",

    "stable,libtorch,windows,accnone,cplusplus":
      "Windows stable binaries do not support Java, but nighty binaries do. Support is only available for Linux and MacOS.  Download here for C++ (Release version): <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.8.1%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.8.1%2Bcpu.zip</a><br/><br>  Download here for C++ (Debug version): <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.8.1%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.8.1%2Bcpu.zip</a>",

    "stable,libtorch,windows,cuda10.2,cplusplus":
      "Windows stable binaries do not support Java, but nighty binaries do. Support is only available for Linux and MacOS.  Download here for C++ (Release version): <br/><a href='https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-1.8.1%2Bcu102.zip'>https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-1.8.1%2Bcu102.zip</a><br/><br>  Download here for C++ (Debug version): <br/><a href='https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-debug-1.8.1%2Bcu102.zip'>https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-debug-1.8.1%2Bcu102.zip</a>",

    "stable,libtorch,windows,cuda11.1,cplusplus":
      "Windows stable binaries do not support Java, but nighty binaries do. Support is only available for Linux and MacOS.  Download here for C++ (Release version): <br/><a href='https://download.pytorch.org/libtorch/cu111/libtorch-win-shared-with-deps-1.8.1%2Bcu111.zip'>https://download.pytorch.org/libtorch/cu111/libtorch-win-shared-with-deps-1.8.1%2Bcu111.zip</a><br/><br>  Download here for C++ (Debug version): <br/><a href='https://download.pytorch.org/libtorch/cu111/libtorch-win-shared-with-deps-debug-1.8.1%2Bcu111.zip'>https://download.pytorch.org/libtorch/cu111/libtorch-win-shared-with-deps-debug-1.8.1%2Bcu111.zip</a>",

    "stable,libtorch,windows,rocm4.0,cplusplus":
      "<b>NOTE:</b> ROCm is not available on Windows",

    "preview,conda,linux,cuda10.2,python":
      "conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly",

    "preview,conda,linux,cuda11.1,python":
      "<b>NOTE:</b> 'nvidia' channel is required for cudatoolkit 11.1<br />conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-nightly -c nvidia",

    "preview,conda,linux,rocm4.0,python":
      "<b>NOTE:</b> Conda packages are not currently available for ROCm, please use pip instead<br />",

    "preview,conda,linux,accnone,python":
      "conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly",

    "preview,conda,macos,cuda10.2,python":
      "conda install pytorch torchvision torchaudio -c pytorch-nightly",

    "preview,conda,macos,cuda11.1,python":
      "conda install pytorch torchvision torchaudio -c pytorch-nightly",

    "preview,conda,macos,rocm4.0,python":
      "<b>NOTE:</b> ROCm is not available on MacOS",

    "preview,conda,macos,accnone,python":
      "conda install pytorch torchvision torchaudio -c pytorch-nightly",

    "preview,conda,windows,cuda10.2,python":
      "conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly",

    "preview,conda,windows,cuda11.1,python":
      "<b>NOTE:</b> 'nvidia' channel is required for cudatoolkit 11.1<br />conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-nightly -c nvidia",

    "preview,conda,windows,rocm4.0,python":
      "<b>NOTE:</b> ROCm is not available on Windows",

    "preview,conda,windows,accnone,python":
      "conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly",

    "preview,pip,macos,cuda10.2,python":
      "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html<br /># On MacOS, we provide CPU-only packages, CUDA functionality is not provided",

    "preview,pip,macos,cuda11.1,python":
      "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html<br /># On MacOS, we provide CPU-only packages, CUDA functionality is not provided",

    "preview,pip,macos,rocm4.0,python":
      "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html<br /># ROCm is not available on MacOS",

    "preview,pip,macos,accnone,python":
      "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html",

    "preview,pip,linux,accnone,python":
      "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html",

    "preview,pip,linux,cuda10.2,python":
      "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html",

    "preview,pip,linux,cuda11.1,python":
      "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html",

    "preview,pip,linux,rocm4.0,python":
      "pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/rocm4.0.1/torch_nightly.html",

    "preview,pip,windows,accnone,python":
      "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html",

    "preview,pip,windows,cuda10.2,python":
      "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html",

    "preview,pip,windows,cuda11.1,python":
      "pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html",

    "preview,pip,windows,rocm4.0,python":
      "<b>NOTE:</b> ROCm is not available on Windows",

    "preview,libtorch,linux,accnone,cplusplus":
      "Download here (Pre-cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip</a><br/><br> Download here (cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>",

    "preview,libtorch,linux,cuda10.2,cplusplus":
      "Download here (Pre-cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu102/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu102/libtorch-shared-with-deps-latest.zip</a><br/><br> Download here (cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu102/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu102/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>",

    "preview,libtorch,linux,cuda11.1,cplusplus":
      "Download here (Pre-cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu111/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu111/libtorch-shared-with-deps-latest.zip</a><br/><br> Download here (cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu111/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu111/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>",

    "preview,libtorch,linux,rocm4.0,cplusplus":
      "LibTorch binaries are not available for ROCm, please build it from source",

    "preview,libtorch,macos,accnone,cplusplus":
      "Download here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip </a>",

    "preview,libtorch,macos,cuda10.2,cplusplus":
      "MacOS binaries do not support CUDA. Download CPU libtorch here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip </a>",

    "preview,libtorch,macos,cuda11.1,cplusplus":
      "MacOS binaries do not support CUDA. Download CPU libtorch here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip </a>",

    "preview,libtorch,macos,rocm4.0,cplusplus":
      "ROCm is not available on MacOS. Download CPU libtorch here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip </a>",

    "preview,libtorch,windows,accnone,cplusplus":
      "Download here for C++ (Release version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip</a><br/><br>  Download here for C++ (Debug version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-debug-latest.zip</a>",

    "preview,libtorch,windows,cuda10.2,cplusplus":
      "Download here for C++ (Release version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu102/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu102/libtorch-win-shared-with-deps-latest.zip</a><br/><br>  Download here for C++ (Debug version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu102/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu102/libtorch-win-shared-with-deps-debug-latest.zip</a>",

    "preview,libtorch,windows,cuda11.1,cplusplus":
      "Download here for C++ (Release version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu111/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu111/libtorch-win-shared-with-deps-latest.zip</a><br/><br>  Download here for C++ (Debug version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu111/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu111/libtorch-win-shared-with-deps-debug-latest.zip</a>",

    "preview,libtorch,windows,rocm4.0,cplusplus":
      "<b>NOTE:</b> ROCm is not available on Windows",
  };

  if (!object.hasOwnProperty(key)) {
    $("#command").html(
      "<pre> # Follow instructions at this URL: https://github.com/pytorch/pytorch#from-source </pre>"
    );
  } else {
    $("#command").html("<pre>" + object[key] + "</pre>");
  }
}
