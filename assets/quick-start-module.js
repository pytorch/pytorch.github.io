// Keys are Substrings as diplayed by navigator.platform
var supportedOperatingSystems = new Map([
  ['linux', 'linux'],
  ['mac', 'macos'],
  ['win', 'windows'],
]);

var opts = {
  cuda: 'cuda10.1',
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
    "stable,conda,linux,cuda9.2,python":
      "conda install pytorch torchvision cudatoolkit=9.2 -c pytorch",

    "stable,conda,linux,cuda10.1,python":
      "conda install pytorch torchvision cudatoolkit=10.1 -c pytorch",

    "stable,conda,linux,cudanone,python":
      "conda install pytorch torchvision cpuonly -c pytorch",

    "stable,conda,macos,cuda9.2,python":
      "conda install pytorch torchvision -c pytorch<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,conda,macos,cuda10.1,python":
      "conda install pytorch torchvision -c pytorch<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,conda,macos,cudanone,python":
      "conda install pytorch torchvision -c pytorch",

    "stable,conda,windows,cuda9.2,python":
      "conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev ",

    "stable,conda,windows,cuda10.1,python":
      "conda install pytorch torchvision cudatoolkit=10.1 -c pytorch",

    "stable,conda,windows,cudanone,python":
      "conda install pytorch torchvision cpuonly -c pytorch",

    "stable,pip,macos,cuda9.2,python":
      "pip install torch torchvision<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,pip,macos,cuda10.1,python":
      "pip install torch torchvision<br /># MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "stable,pip,macos,cudanone,python": "pip install torch torchvision",

    "stable,pip,linux,cudanone,python":
      "pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,linux,cuda9.2,python":
      "pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,linux,cuda10.1,python":
      "pip install torch torchvision",

    "stable,pip,windows,cudanone,python":
      "pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,cuda9.2,python":
      "pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,pip,windows,cuda10.1,python":
      "pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html",

    "stable,libtorch,linux,cudanone,cplusplus":
      "Download here (Pre-cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.4.0%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.4.0%2Bcpu.zip</a><br/><br> Download here (cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip</a>",

    "stable,libtorch,linux,cuda9.2,cplusplus":
      "Download here (Pre-cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/cu92/libtorch-shared-with-deps-1.4.0%2Bcu92.zip'>https://download.pytorch.org/libtorch/cu92/libtorch-shared-with-deps-1.4.0%2Bcu92.zip</a><br/><br> Download here (cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/cu92/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcu92.zip'>https://download.pytorch.org/libtorch/cu92/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcu92.zip</a>",

    "stable,libtorch,linux,cuda10.1,cplusplus":
      "Download here (Pre-cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.4.0.zip'>https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.4.0.zip</a><br/><br> Download here (cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.4.0.zip'>https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.4.0.zip</a>",

    "stable,libtorch,macos,cudanone,cplusplus":
      "MacOS binaries do not support Java. Support is only available for Linux. Download here for C++: <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip'> https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip </a>",

    "stable,libtorch,macos,cuda9.2,cplusplus":
      "MacOS binaries do not support Java. Support is only available for Linux. MacOS binaries do not support CUDA. Download CPU libtorch here for C++: <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip'> https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip </a>",

    "stable,libtorch,macos,cuda10.1,cplusplus":
      "MacOS binaries do not support Java. Support is only available for Linux. MacOS binaries do not support CUDA. Download CPU libtorch here for C++: <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip'> https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip </a>",

    "stable,libtorch,windows,cudanone,cplusplus":
      "Windows binaries do not support Java. Support is only available for Linux.  Download here for C++ (Release version): <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.4.0.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.4.0.zip</a><br/><br>  Download here for C++ (Debug version): <br/><a href='https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.4.0.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.4.0.zip</a>",

    "stable,libtorch,windows,cuda9.2,cplusplus":
      "Windows binaries do not support Java. Support is only available for Linux.  Download here for C++ (Release version): <br/><a href='https://download.pytorch.org/libtorch/cu92/libtorch-win-shared-with-deps-1.4.0.zip'>https://download.pytorch.org/libtorch/cu92/libtorch-win-shared-with-deps-1.4.0.zip</a><br/><br>  Download here for C++ (Debug version): <br/><a href='https://download.pytorch.org/libtorch/cu92/libtorch-win-shared-with-deps-debug-1.4.0.zip'>https://download.pytorch.org/libtorch/cu92/libtorch-win-shared-with-deps-debug-1.4.0.zip</a>",

    "stable,libtorch,windows,cuda10.1,cplusplus":
      "Windows binaries do not support Java. Support is only available for Linux.  Download here for C++ (Release version): <br/><a href='https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-1.4.0.zip'>https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-1.4.0.zip</a><br/><br>  Download here for C++ (Debug version): <br/><a href='https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-debug-1.4.0.zip'>https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-debug-1.4.0.zip</a>",

    "preview,conda,linux,cuda9.2,python":
      "conda install pytorch torchvision cudatoolkit=9.2 -c pytorch-nightly",

    "preview,conda,linux,cuda10.1,python":
      "conda install pytorch torchvision cudatoolkit=10.1 -c pytorch-nightly",

    "preview,conda,linux,cudanone,python":
      "conda install pytorch torchvision cpuonly -c pytorch-nightly",

    "preview,conda,macos,cuda9.2,python":
      "conda install pytorch torchvision -c pytorch-nightly",

    "preview,conda,macos,cuda10.1,python":
      "conda install pytorch torchvision -c pytorch-nightly",

    "preview,conda,macos,cudanone,python":
      "conda install pytorch torchvision -c pytorch-nightly",

    "preview,conda,windows,cuda9.2,python":
      "conda install pytorch torchvision cudatoolkit=9.2 -c pytorch-nightly -c defaults -c conda-forge -c numba/label/dev",

    "preview,conda,windows,cuda10.1,python":
      "conda install pytorch torchvision cudatoolkit=10.1 -c pytorch-nightly -c defaults -c conda-forge",

    "preview,conda,windows,cudanone,python":
      "conda install pytorch torchvision cpuonly -c pytorch-nightly -c defaults -c conda-forge",

    "preview,pip,macos,cuda9.2,python":
      "pip install numpy<br />pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html<br /># On MacOS, we provide CPU-only packages, CUDA functionality is not provided",

    "preview,pip,macos,cuda10.1,python":
      "pip install numpy<br />pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html<br /># On MacOS, we provide CPU-only packages, CUDA functionality is not provided",

    "preview,pip,macos,cudanone,python":
      "pip install numpy<br />pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html",

    "preview,pip,linux,cudanone,python":
      "pip install numpy<br />pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html",

    "preview,pip,linux,cuda9.2,python":
      "pip install numpy<br />pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html",

    "preview,pip,linux,cuda10.1,python":
      "pip install numpy<br />pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html",

    "preview,pip,windows,cudanone,python":
      "pip install numpy<br />pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html",

    "preview,pip,windows,cuda9.2,python":
      "pip install numpy<br />pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html",

    "preview,pip,windows,cuda10.1,python":
      "pip install numpy<br />pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html",

    "preview,libtorch,linux,cudanone,cplusplus":
      "Download here (Pre-cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip</a><br/><br> Download here (cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>",

    "preview,libtorch,linux,cuda9.2,cplusplus":
      "Download here (Pre-cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu92/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu92/libtorch-shared-with-deps-latest.zip</a><br/><br> Download here (cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu92/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu92/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>",

    "preview,libtorch,linux,cuda10.1,cplusplus":
      "Download here (Pre-cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu101/libtorch-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu101/libtorch-shared-with-deps-latest.zip</a><br/><br> Download here (cxx11 ABI): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu101/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu101/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>",

    "preview,libtorch,macos,cudanone,cplusplus":
      "MacOS binaries do not support Java. Support is only available for Linux. Download here for C++: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip </a>",

    "preview,libtorch,macos,cuda9.2,cplusplus":
      "MacOS binaries do not support Java. Support is only available for Linux. MacOS binaries do not support CUDA. Download CPU libtorch here for C++: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip </a>",

    "preview,libtorch,macos,cuda10.1,cplusplus":
      "MacOS binaries do not support Java. Support is only available for Linux. MacOS binaries do not support CUDA. Download CPU libtorch here for C++: <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip'> https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip </a>",

    "preview,libtorch,windows,cudanone,cplusplus":
      "Windows binaries do not support Java. Support is only available for Linux.  Download here for C++ (Release version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip</a><br/><br>  Download here for C++ (Debug version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-debug-latest.zip</a>",

    "preview,libtorch,windows,cuda9.2,cplusplus":
      "Windows binaries do not support Java. Support is only available for Linux.  Download here for C++ (Release version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu92/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu92/libtorch-win-shared-with-deps-latest.zip</a><br/><br>  Download here for C++ (Debug version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu92/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu92/libtorch-win-shared-with-deps-debug-latest.zip</a>",

    "preview,libtorch,windows,cuda10.1,cplusplus":
      "Windows binaries do not support Java. Support is only available for Linux.  Download here for C++ (Release version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu101/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu101/libtorch-win-shared-with-deps-latest.zip</a><br/><br>  Download here for C++ (Debug version): <br/><a href='https://download.pytorch.org/libtorch/nightly/cu101/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu101/libtorch-win-shared-with-deps-debug-latest.zip</a>",
  };

  if (!object.hasOwnProperty(key)) {
    $("#command").html(
      "<pre> # Follow instructions at this URL: https://github.com/pytorch/pytorch#from-source </pre>"
    );
  } else {
    $("#command").html("<pre>" + object[key] + "</pre>");
  }
}
