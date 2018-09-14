// Keys are Substrings as diplayed by navigator.platform
var supportedOperatingSystems = new Map([
  ['linux', 'linux'],
  ['mac', 'macos'],
  ['win', 'windows'],
]);

var opts = {
  cuda: 'cuda9.0',
  os: getAnchorSelectedOS() || getDefaultSelectedOS(),
  pm: 'conda',
  python: 'python3.6',
};

var supportedCloudPlatforms = [
  'aws',
  'google-cloud',
  'microsoft-azure',
];

var os = $(".os > .option");
var package = $(".package > .option");
var python = $(".python > .option");
var cuda = $(".cuda > .option");

os.on("click", function() {
  selectedOption(os, this, "os");
});
package.on("click", function() {
  selectedOption(package, this, "pm");
});
python.on("click", function() {
  selectedOption(python, this, "python");
});
cuda.on("click", function() {
  selectedOption(cuda, this, "cuda");
});

// Force a selection onclick to get the right operating system selected from
// the start
$( document ).ready(function() {
    document.getElementById(opts.os).click();
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
  console.log("getAnchorSelectedOS - 1 "+anchor);
  var ANCHOR_REGEX = /^#[^ ]+$/;
  // Look for anchor in the href
  if (!ANCHOR_REGEX.test(anchor)) {
    return false;
  }
  // Look for anchor with OS in the first portion
  var testOS = anchor.slice(1).split("-")[0];
  console.log("getAnchorSelectedOS - 2 "+testOS);
  for (var [navPlatformSubstring, os] of supportedOperatingSystems.entries()) {
    if (testOS.indexOf(navPlatformSubstring) !== -1) {
      console.log("getAnchorSelectedOS - 3 "+os);
      return os;
    }
  }
  return false;
}

function selectedOption(option, selection, category) {
  $(option).removeClass("selected");
  $(selection).addClass("selected");
  opts[category] = selection.id;
  commandMessage(buildMatcher());
  if (category === "os") {
    display(opts.os, 'installation', 'os');
  }
}

function display(selection, id, category) {
  var container = document.getElementById(id);
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
    opts.pm.toLowerCase() +
    "," +
    opts.os.toLowerCase() +
    "," +
    opts.cuda.toLowerCase() +
    "," +
    opts.python.toLowerCase()
  );
}

function commandMessage(key) {
  var object = {
    "conda,linux,cuda8,python2.7":
      "conda install pytorch torchvision cuda80 -c pytorch",

    "conda,linux,cuda9.0,python2.7":
      "conda install pytorch torchvision -c pytorch",

    "conda,linux,cuda9.2,python2.7":
      "conda install pytorch torchvision cuda92 -c pytorch",

    "conda,linux,cudanone,python2.7":
      "conda install pytorch-cpu torchvision-cpu -c pytorch",

    "conda,linux,cuda8,python3.5":
      "conda install pytorch torchvision cuda80 -c pytorch",

    "conda,linux,cuda9.0,python3.5":
      "conda install pytorch torchvision -c pytorch",

    "conda,linux,cuda9.2,python3.5":
      "conda install pytorch torchvision cuda92 -c pytorch",

    "conda,linux,cudanone,python3.5":
      "conda install pytorch-cpu torchvision-cpu -c pytorch",

    "conda,linux,cuda8,python3.6":
      "conda install pytorch torchvision cuda80 -c pytorch",

    "conda,linux,cuda9.0,python3.6":
      "conda install pytorch torchvision -c pytorch",

    "conda,linux,cuda9.2,python3.6":
      "conda install pytorch torchvision cuda92 -c pytorch",

    "conda,linux,cudanone,python3.6":
      "conda install pytorch-cpu torchvision-cpu -c pytorch",

    "conda,linux,cuda8,python3.7":
      "conda install pytorch torchvision cuda80 -c pytorch",

    "conda,linux,cuda9.0,python3.7":
      "conda install pytorch torchvision -c pytorch",

    "conda,linux,cuda9.2,python3.7":
      "conda install pytorch torchvision cuda92 -c pytorch",

    "conda,linux,cudanone,python3.7":
      "conda install pytorch-cpu torchvision-cpu -c pytorch",

    "conda,macos,cuda8,python2.7":
      "conda install pytorch torchvision -c pytorch <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "conda,macos,cuda9.0,python2.7":
      "conda install pytorch torchvision -c pytorch <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "conda,macos,cuda9.2,python2.7":
      "conda install pytorch torchvision -c pytorch <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "conda,macos,cudanone,python2.7":
      "conda install pytorch torchvision -c pytorch",

    "conda,macos,cuda8,python3.5":
      "conda install pytorch torchvision -c pytorch <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "conda,macos,cuda9.0,python3.5":
      "conda install pytorch torchvision -c pytorch <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "conda,macos,cuda9.2,python3.5":
      "conda install pytorch torchvision -c pytorch <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "conda,macos,cudanone,python3.5":
      "conda install pytorch torchvision -c pytorch",

    "conda,macos,cuda8,python3.6":
      "conda install pytorch torchvision -c pytorch <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "conda,macos,cuda9.0,python3.6":
      "conda install pytorch torchvision -c pytorch <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "conda,macos,cuda9.2,python3.6":
      "conda install pytorch torchvision -c pytorch <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "conda,macos,cudanone,python3.6":
      "conda install pytorch torchvision -c pytorch",

    "conda,macos,cuda8,python3.7":
      "conda install pytorch torchvision -c pytorch <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "conda,macos,cuda9.0,python3.7":
      "conda install pytorch torchvision -c pytorch <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "conda,macos,cuda9.2,python3.7":
      "conda install pytorch torchvision -c pytorch <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "conda,macos,cudanone,python3.7":
      "conda install pytorch torchvision -c pytorch",

    "conda,windows,cuda8,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "conda,windows,cuda9.0,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "conda,windows,cuda9.2,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "conda,windows,cudanone,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "conda,windows,cuda8,python3.5":
      "conda install pytorch cuda80 -c pytorch <br /> pip3 install torchvision",

    "conda,windows,cuda9.0,python3.5":
      "conda install pytorch -c pytorch <br /> pip3 install torchvision",

    "conda,windows,cuda9.2,python3.5":
      "conda install pytorch cuda92 -c pytorch <br /> pip3 install torchvision",

    "conda,windows,cudanone,python3.5":
      "conda install pytorch-cpu -c pytorch <br /> pip3 install torchvision",

    "conda,windows,cuda8,python3.6":
      "conda install pytorch cuda80 -c pytorch <br /> pip3 install torchvision",

    "conda,windows,cuda9.0,python3.6":
      "conda install pytorch -c pytorch <br /> pip3 install torchvision",

    "conda,windows,cuda9.2,python3.6":
      "conda install pytorch cuda92 -c pytorch <br /> pip3 install torchvision",

    "conda,windows,cudanone,python3.6":
      "conda install pytorch-cpu -c pytorch <br /> pip3 install torchvision",

    "conda,windows,cuda8,python3.7":
      "conda install pytorch cuda80 -c pytorch <br /> pip3 install torchvision",

    "conda,windows,cuda9.0,python3.7":
      "conda install pytorch -c pytorch <br /> pip3 install torchvision",

    "conda,windows,cuda9.2,python3.7":
      "conda install pytorch cuda92 -c pytorch <br /> pip3 install torchvision",

    "conda,windows,cudanone,python3.7":
      "conda install pytorch-cpu -c pytorch <br /> pip3 install torchvision",

    "pip,macos,cuda8,python2.7":
      "pip install torch torchvision <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "pip,macos,cuda9.0,python2.7":
      "pip install torch torchvision <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "pip,macos,cuda9.2,python2.7":
      "pip install torch torchvision <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "pip,macos,cudanone,python2.7": "pip install torch torchvision",

    "pip,macos,cuda8,python3.5":
      "pip3 install torch torchvision <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "pip,macos,cuda9.0,python3.5":
      "pip3 install torch torchvision <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "pip,macos,cuda9.2,python3.5":
      "pip3 install torch torchvision <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "pip,macos,cudanone,python3.5": "pip3 install torch torchvision",

    "pip,macos,cuda8,python3.6":
      "pip3 install torch torchvision <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "pip,macos,cuda9.0,python3.6":
      "pip3 install torch torchvision <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "pip,macos,cuda9.2,python3.6":
      "pip3 install torch torchvision <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "pip,macos,cudanone,python3.6": "pip3 install torch torchvision",

    "pip,macos,cuda8,python3.7":
      "pip3 install torch torchvision <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "pip,macos,cuda9.0,python3.7":
      "pip3 install torch torchvision <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "pip,macos,cuda9.2,python3.7":
      "pip3 install torch torchvision <br /> # MacOS Binaries dont support CUDA, install from source if CUDA is needed",

    "pip,macos,cudanone,python3.7": "pip3 install torch torchvision",

    "pip,linux,cudanone,python2.7":
      "pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl <br/> pip install torchvision <br/><br/> # if the above command does not work, then you have python 2.7 UCS2, use this command <br/> pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp27-cp27m-linux_x86_64.whl",

    "pip,linux,cuda8,python2.7":
      "pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl <br/> pip install torchvision <br/><br/> # if the above command does not work, then you have python 2.7 UCS2, use this command <br/> pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp27-cp27m-linux_x86_64.whl",

    "pip,linux,cuda9.0,python2.7": "pip install torch torchvision",

    "pip,linux,cuda9.2,python2.7":
      "pip install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl <br/> pip install torchvision <br/><br/> # if the above command does not work, then you have python 2.7 UCS2, use this command <br/> pip install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp27-cp27m-linux_x86_64.whl",

    "pip,linux,cudanone,python3.5":
      "pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl <br/> pip3 install torchvision",

    "pip,linux,cuda8,python3.5":
      "pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl <br/> pip3 install torchvision",

    "pip,linux,cuda9.0,python3.5": "pip3 install torch torchvision",

    "pip,linux,cuda9.2,python3.5":
      "pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-linux_x86_64.whl <br/> pip3 install torchvision",

    "pip,linux,cudanone,python3.6":
      "pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl <br/> pip3 install torchvision",

    "pip,linux,cuda8,python3.6":
      "pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-linux_x86_64.whl <br/> pip3 install torchvision",

    "pip,linux,cuda9.0,python3.6": "pip3 install torch torchvision",

    "pip,linux,cuda9.2,python3.6":
      "pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl <br/> pip3 install torchvision",

    "pip,linux,cudanone,python3.7":
      "pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl <br/> pip3 install torchvision",

    "pip,linux,cuda8,python3.7":
      "pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl <br/> pip3 install torchvision",

    "pip,linux,cuda9.0,python3.7": "pip3 install torch torchvision",

    "pip,linux,cuda9.2,python3.7":
      "pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl <br/> pip3 install torchvision",

    "pip,windows,cudanone,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "pip,windows,cuda8,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "pip,windows,cuda9.0,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "pip,windows,cuda9.2,python2.7":
      "# PyTorch does not support Python 2.7 on Windows. Please install with Python 3.",

    "pip,windows,cudanone,python3.5":
      "pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-win_amd64.whl <br/> pip3 install torchvision",

    "pip,windows,cuda8,python3.5":
      "pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-win_amd64.whl <br/> pip3 install torchvision",

    "pip,windows,cuda9.0,python3.5":
      "pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp35-cp35m-win_amd64.whl <br/> pip3 install torchvision",

    "pip,windows,cuda9.2,python3.5":
      "pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-win_amd64.whl <br/> pip3 install torchvision",

    "pip,windows,cudanone,python3.6":
      "pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl <br/> pip3 install torchvision",

    "pip,windows,cuda8,python3.6":
      "pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-win_amd64.whl <br/> pip3 install torchvision",

    "pip,windows,cuda9.0,python3.6":
      "pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl <br/> pip3 install torchvision",

    "pip,windows,cuda9.2,python3.6":
      "pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl <br/> pip3 install torchvision",

    "pip,windows,cudanone,python3.7":
      "pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp37-cp37m-win_amd64.whl <br/> pip3 install torchvision",

    "pip,windows,cuda8,python3.7":
      "pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp37-cp37m-win_amd64.whl <br/> pip3 install torchvision",

    "pip,windows,cuda9.0,python3.7":
      "pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp37-cp37m-win_amd64.whl <br/> pip3 install torchvision",

    "pip,windows,cuda9.2,python3.7":
      "pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp37-cp37m-win_amd64.whl <br/> pip3 install torchvision"
  };

  if (!object.hasOwnProperty(key)) {
    $("#command").html(
      "# Follow instructions at this URL: https://github.com/pytorch/pytorch#from-source"
    );
  } else {
    $("#command").html(object[key]);
  }
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
      console.log(cls);
      if (cls.includes(supportedCloudPlatforms[i])) {
        display(supportedCloudPlatforms[i], 'cloud', 'platform');
      }
    }
  }
});

