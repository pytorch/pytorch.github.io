// Keys are Substrings as diplayed by navigator.platform
var supportedOperatingSystems = new Map([
  ['linux', 'linux'],
  ['mac', 'macos'],
  ['win', 'windows'],
]);

var opts = {
  cuda: 'cuda9.2',
  os: getAnchorSelectedOS() || getDefaultSelectedOS(),
  pm: 'conda',
  language: 'python3.6',
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
$(document).ready(function() {
  var userOsOption = document.getElementById(opts.os);

  if (userOsOption) {
    selectedOption(os, userOsOption, "os");
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
      $(elements["python3.7"]).addClass("selected");
      opts["language"] = "python3.7";
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
    "stable,conda,linux,cuda9.2,python2.7":
      "conda install pytorch torchvision cudatoolkit=9.2 -c pytorch",

    "stable,conda,linux,cuda10.0,python2.7":
      "conda install pytorch torchvision cudatoolkit=10.0 -c pytorch",

    "stable,conda,linux,cudanone,python2.7":
      "conda install pytorch torchvision cpuonly -c pytorch",

    "stable,conda,linux,cuda9.2,python3.5":
      "conda install pytorch torchvision cudatoolkit=9.2 -c pytorch",

    "stable,conda,linux,cuda10.0,python3.5":
      "conda install pytorch torchvision cudatoolkit=10.0 -c pytorch",

    "stable,conda,linux,cudanone,python3.5":
      "conda install pytorch torchvision cpuonly -c pytorch",

    "stable,conda,linux,cuda9.2,python3.6":
      "conda install pytorch torchvision cudatoolkit=9.2 -c pytorch",

    "stable,conda,linux,cuda10.0,python3.6":
      "conda install pytorch torchvision cudatoolkit=10.0 -c pytorch",

    "stable,conda,linux,cudanone,python3.6":
      "conda install pytorch torchvision cpuonly -c pytorch",

    "stable,conda,linux,cuda9.2,python3.7":
      "conda install pytorch torchvision cudatoolkit=9.2 -c pytorch",

    "stable,conda,linux,cuda10.0,python3.7":
      "conda install pytorch torchvision cudatoolkit=10.0 -c pytorch",

    "stable,conda,linux,cudanone,python3.7":
      "conda install pytorch torchvision cpuonly -c pytorch",

    "stable,conda,macos,cuda9.2,python2.7":
      "conda install pytorch torchvision -c pytorch<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,conda,macos,cuda10.0,python2.7":
      "conda install pytorch torchvision -c pytorch<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,conda,macos,cudanone,python2.7":
      "conda install pytorch torchvision -c pytorch",

    "stable,conda,macos,cuda9.2,python3.5":
      "conda install pytorch torchvision -c pytorch<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,conda,macos,cuda10.0,python3.5":
      "conda install pytorch torchvision -c pytorch<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,conda,macos,cudanone,python3.5":
      "conda install pytorch torchvision -c pytorch",

    "stable,conda,macos,cuda9.2,python3.6":
      "conda install pytorch torchvision -c pytorch<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,conda,macos,cuda10.0,python3.6":
      "conda install pytorch torchvision -c pytorch<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,conda,macos,cudanone,python3.6":
      "conda install pytorch torchvision -c pytorch",

    "stable,conda,macos,cuda9.2,python3.7":
      "conda install pytorch torchvision -c pytorch<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,conda,macos,cuda10.0,python3.7":
      "conda install pytorch torchvision -c pytorch<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,conda,macos,cudanone,python3.7":
      "conda install pytorch torchvision -c pytorch",

    "stable,conda,windows,cuda9.2,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "stable,conda,windows,cuda10.0,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "stable,conda,windows,cudanone,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "stable,conda,windows,cuda9.2,python3.5":
      "conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev ",

    "stable,conda,windows,cuda10.0,python3.5":
      "conda install pytorch torchvision cudatoolkit=10.0 -c pytorch",

    "stable,conda,windows,cudanone,python3.5":
      "conda install pytorch torchvision cpuonly -c pytorch",

    "stable,conda,windows,cuda9.2,python3.6":
      "conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev",

    "stable,conda,windows,cuda10.0,python3.6":
      "conda install pytorch torchvision cudatoolkit=10.0 -c pytorch",

    "stable,conda,windows,cudanone,python3.6":
      "conda install pytorch torchvision cpuonly -c pytorch",

    "stable,conda,windows,cuda9.2,python3.7":
      "conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev",

    "stable,conda,windows,cuda10.0,python3.7":
      "conda install pytorch torchvision cudatoolkit=10.0 -c pytorch",

    "stable,conda,windows,cudanone,python3.7":
      "conda install pytorch torchvision cpuonly -c pytorch",

    "stable,pip,macos,cuda9.2,python2.7":
      "pip install torch torchvision<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,pip,macos,cuda10.0,python2.7":
      "pip install torch torchvision<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,pip,macos,cudanone,python2.7": "pip install torch torchvision",

    "stable,pip,macos,cuda9.2,python3.5":
      "pip3 install torch torchvision<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,pip,macos,cuda10.0,python3.5":
      "pip3 install torch torchvision<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,pip,macos,cudanone,python3.5": "pip3 install torch torchvision",

    "stable,pip,macos,cuda9.2,python3.6":
      "pip3 install torch torchvision<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,pip,macos,cuda10.0,python3.6":
      "pip3 install torch torchvision<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,pip,macos,cudanone,python3.6": "pip3 install torch torchvision",

    "stable,pip,macos,cuda9.2,python3.7":
      "pip3 install torch torchvision<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,pip,macos,cuda10.0,python3.7":
      "pip3 install torch torchvision<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,pip,macos,cudanone,python3.7": "pip3 install torch torchvision",

    "stable,pip,linux,cudanone,python2.7":
      "pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,linux,cuda9.2,python2.7":
      "pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,linux,cuda10.0,python2.7":
       "pip install torch torchvision",

    "stable,pip,linux,cudanone,python3.5":
      "pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,linux,cuda9.2,python3.5": 
      "pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,linux,cuda10.0,python3.5":
      "pip3 install torch torchvision",

    "stable,pip,linux,cudanone,python3.6":
      "pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,linux,cuda9.2,python3.6": 
      "pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,linux,cuda10.0,python3.6":
      "pip3 install torch torchvision",

    "stable,pip,linux,cudanone,python3.7":
      "pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,linux,cuda9.2,python3.7":
      "pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,linux,cuda10.0,python3.7":
      "pip3 install torch torchvision",

    "stable,pip,windows,cudanone,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "stable,pip,windows,cuda9.2,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "stable,pip,windows,cuda10.0,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "stable,pip,windows,cudanone,python3.5":
      "pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,cuda9.2,python3.5":
      "pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,cuda10.0,python3.5":
      "pip3 install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,cudanone,python3.6":
      "pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,cuda9.2,python3.6":
      "pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,cuda10.0,python3.6":
      "pip3 install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,cudanone,python3.7":
      "pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,cuda9.2,python3.7":
      "pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,cuda10.0,python3.7":
      "pip3 install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,libtorch,linux,cudanone,cplusplus":
      "Download here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip</a>",

    "stable,libtorch,linux,cuda9.2,cplusplus":
      "Download here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cu92/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu92/libtorch-shared-with-deps-latest.zip</a>",

    "stable,libtorch,linux,cuda10.0,cplusplus":
      "Download here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cu100/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu100/libtorch-shared-with-deps-latest.zip</a>",

    "stable,libtorch,macos,cudanone,cplusplus":
      "Download here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip </a>",

    "stable,libtorch,macos,cuda9.2,cplusplus":
      "MacOS binaries do not support CUDA. Download CPU libtorch here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-1.2.0.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-1.2.0.zip </a>",

    "stable,libtorch,macos,cuda10.0,cplusplus":
      "MacOS binaries do not support CUDA. Download CPU libtorch here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-1.2.0.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-1.2.0.zip </a>",

    "stable,libtorch,windows,cudanone,cplusplus":
      "Download here (Release version): <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-latest.zip</a><br/><br> Download here (Debug version): <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-latest.zip</a>",

    "stable,libtorch,windows,cuda9.2,cplusplus":
      "Download here (Release version): <br/><a href='https://download.pytorch.org/libtorch/cu92/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/cu92/libtorch-win-shared-with-deps-latest.zip</a><br/><br> Download here (Debug version): <br/><a href='https://download.pytorch.org/libtorch/cu92/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/cu92/libtorch-win-shared-with-deps-debug-latest.zip</a>",

    "stable,libtorch,windows,cuda10.0,cplusplus":
      "Download here (Release version): <br/><a href='https://download.pytorch.org/libtorch/cu100/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/cu100/libtorch-win-shared-with-deps-latest.zip</a><br/><br> Download here (Debug version): <br/><a href='https://download.pytorch.org/libtorch/cu100/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/cu100/libtorch-win-shared-with-deps-debug-latest.zip</a>",

    "preview,conda,linux,cuda9.2,python2.7":
      "conda install pytorch cudatoolkit=9.2 -c pytorch-nightly",

    "preview,conda,linux,cuda10.0,python2.7":
      "conda install pytorch cudatoolkit=10.0 -c pytorch-nightly",

    "preview,conda,linux,cudanone,python2.7":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,linux,cuda9.2,python3.5":
      "conda install pytorch cudatoolkit=9.2 -c pytorch-nightly",

    "preview,conda,linux,cuda10.0,python3.5":
      "conda install pytorch cudatoolkit=10.0 -c pytorch-nightly",

    "preview,conda,linux,cudanone,python3.5":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,linux,cuda9.2,python3.6":
      "conda install pytorch cudatoolkit=9.2 -c pytorch-nightly",

    "preview,conda,linux,cuda10.0,python3.6":
      "conda install pytorch cudatoolkit=10.0 -c pytorch-nightly",

    "preview,conda,linux,cudanone,python3.6":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,linux,cuda9.2,python3.7":
      "conda install pytorch cudatoolkit=9.2 -c pytorch-nightly",

    "preview,conda,linux,cuda10.0,python3.7":
      "conda install pytorch cudatoolkit=10.0 -c pytorch-nightly",

    "preview,conda,linux,cudanone,python3.7":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,macos,cuda9.2,python2.7":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,macos,cuda10.0,python2.7":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,macos,cudanone,python2.7":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,macos,cuda9.2,python3.5":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,macos,cuda10.0,python3.5":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,macos,cudanone,python3.5":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,macos,cuda9.2,python3.6":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,macos,cuda10.0,python3.6":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,macos,cudanone,python3.6":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,macos,cuda9.2,python3.7":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,macos,cuda10.0,python3.7":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,macos,cudanone,python3.7":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,windows,cuda9.2,python2.7":
      "# Preview Build With Python 2.7 On Windows Not Supported.",

    "preview,conda,windows,cuda10.0,python2.7":
      "# Preview Build With Python 2.7 On Windows Not Supported.",

    "preview,conda,windows,cudanone,python2.7":
      "# Preview Build With Python 2.7 On Windows Not Supported.",

    "preview,conda,windows,cuda9.2,python3.5":
      "conda install pytorch cudatoolkit=9.2 -c pytorch-nightly -c defaults -c numba/label/dev",

    "preview,conda,windows,cuda10.0,python3.5":
      "conda install pytorch cudatoolkit=10.0 -c pytorch-nightly",

    "preview,conda,windows,cudanone,python3.5":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,windows,cuda9.2,python3.6":
      "conda install pytorch cudatoolkit=9.2 -c pytorch-nightly -c defaults -c numba/label/dev",

    "preview,conda,windows,cuda10.0,python3.6":
      "conda install pytorch cudatoolkit=10.0 -c pytorch-nightly",

    "preview,conda,windows,cudanone,python3.6":
      "conda install pytorch -c pytorch-nightly",

    "preview,conda,windows,cuda9.2,python3.7":
      "conda install pytorch cudatoolkit=9.2 -c pytorch-nightly -c defaults -c numba/label/dev",

    "preview,conda,windows,cuda10.0,python3.7":
      "conda install pytorch cudatoolkit=10.0 -c pytorch-nightly",

    "preview,conda,windows,cudanone,python3.7":
      "conda install pytorch -c pytorch-nightly",

    "preview,pip,macos,cuda9.2,python2.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html<br /># On MacOS, we provide CPU-only packages, CUDA functionality is not provided",

    "preview,pip,macos,cuda10.0,python2.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html<br /># On MacOS, we provide CPU-only packages, CUDA functionality is not provided",

    "preview,pip,macos,cudanone,python2.7":
      "pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html",

    "preview,pip,macos,cuda9.2,python3.5":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html<br /># On MacOS, we provide CPU-only packages, CUDA functionality is not provided",

    "preview,pip,macos,cuda10.0,python3.5":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html<br /># On MacOS, we provide CPU-only packages, CUDA functionality is not provided",

    "preview,pip,macos,cudanone,python3.5":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html",

    "preview,pip,macos,cuda9.2,python3.6":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html<br /># On MacOS, we provide CPU-only packages, CUDA functionality is not provided",

    "preview,pip,macos,cuda10.0,python3.6":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html<br /># On MacOS, we provide CPU-only packages, CUDA functionality is not provided",

    "preview,pip,macos,cudanone,python3.6":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html",

    "preview,pip,macos,cuda9.2,python3.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html <br /># On MacOS, we provide CPU-only packages, CUDA functionality is not provided",

    "preview,pip,macos,cuda10.0,python3.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html <br /># On MacOS, we provide CPU-only packages, CUDA functionality is not provided",

    "preview,pip,macos,cudanone,python3.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html",

    "preview,pip,linux,cudanone,python2.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html",

    "preview,pip,linux,cuda9.2,python2.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu92/torch.html",

    "preview,pip,linux,cuda10.0,python2.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu100/torch.html",

    "preview,pip,linux,cudanone,python3.5":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html",

    "preview,pip,linux,cuda9.2,python3.5":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu92/torch.html",

    "preview,pip,linux,cuda10.0,python3.5":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu100/torch.html",

    "preview,pip,linux,cudanone,python3.6":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html",

    "preview,pip,linux,cuda9.2,python3.6":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu92/torch.html",

    "preview,pip,linux,cuda10.0,python3.6":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu100/torch.html",

    "preview,pip,linux,cudanone,python3.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html",

    "preview,pip,linux,cuda9.2,python3.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu92/torch.html",

    "preview,pip,linux,cuda10.0,python3.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu100/torch.html",

    "preview,pip,windows,cudanone,python2.7":
      "# Preview Build With Python 2.7 On Windows Not Supported.",

    "preview,pip,windows,cuda9.2,python2.7":
      "# Preview Build With Python 2.7 On Windows Not Supported.",

    "preview,pip,windows,cuda10.0,python2.7":
      "# Preview Build With Python 2.7 On Windows Not Supported.",

    "preview,pip,windows,cudanone,python3.5":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html",

    "preview,pip,windows,cuda9.2,python3.5":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu92/torch.html",

    "preview,pip,windows,cuda10.0,python3.5":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu100/torch.html",

    "preview,pip,windows,cudanone,python3.6":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html",

    "preview,pip,windows,cuda9.2,python3.6":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu92/torch.html",

    "preview,pip,windows,cuda10.0,python3.6":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu100/torch.html",

    "preview,pip,windows,cudanone,python3.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html",

    "preview,pip,windows,cuda9.2,python3.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu92/torch.html",

    "preview,pip,windows,cuda10.0,python3.7":
      "pip install numpy<br />pip install torch -f https://download.pytorch.org/whl/nightly/cu100/torch.html",

    "preview,libtorch,linux,cudanone,cplusplus":
      "Download here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip</a>",

    "preview,libtorch,linux,cuda9.2,cplusplus":
      "Download here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cu92/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu92/libtorch-shared-with-deps-latest.zip</a>",

    "preview,libtorch,linux,cuda10.0,cplusplus":
      "Download here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cu100/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu100/libtorch-shared-with-deps-latest.zip</a>",

    "preview,libtorch,macos,cudanone,cplusplus":
      "Download here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip </a>",

    "preview,libtorch,macos,cuda9.2,cplusplus":
      "MacOS binaries do not support CUDA. Download CPU libtorch here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip </a>",

    "preview,libtorch,macos,cuda10.0,cplusplus":
      "MacOS binaries do not support CUDA. Download CPU libtorch here: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip </a>",

    "preview,libtorch,windows,cudanone,cplusplus":
      "Download here (Release version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip</a><br/><br> Download here (Debug version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-debug-latest.zip</a>",

    "preview,libtorch,windows,cuda9.2,cplusplus":
      "Download here (Release version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu92/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu92/libtorch-win-shared-with-deps-latest.zip</a><br/><br> Download here (Debug version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu92/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu92/libtorch-win-shared-with-deps-debug-latest.zip</a>",

    "preview,libtorch,windows,cuda10.0,cplusplus":
      "Download here (Release version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu100/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu100/libtorch-win-shared-with-deps-latest.zip</a><br/><br> Download here (Debug version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu100/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu100/libtorch-win-shared-with-deps-debug-latest.zip</a>",
  };

  if (!object.hasOwnProperty(key)) {
    $("#command").html(
      "<pre> # Follow instructions at this URL: https://github.com/pytorch/pytorch#from-source </pre>"
    );
  } else {
    $("#command").html("<pre>" + object[key] + "</pre>");
  }
}
