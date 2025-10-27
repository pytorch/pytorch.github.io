// Keys are Substrings as diplayed by navigator.platform
var supportedOperatingSystems = new Map([
  ['linux', 'linux'],
  ['mac', 'macos'],
  ['win', 'windows'],
]);

var archInfoMap = new Map([
  ['cuda', {title: "CUDA", platforms: new Set(['linux', 'windows'])}],
  ['rocm', {title: "ROCm", platforms: new Set(['linux'])}],
  ['accnone', {title: "CPU", platforms: new Set(['linux', 'macos', 'windows'])}]
]);

let version_map={{ ACC ARCH MAP }}
let stable_version={{ VERSION }};

var default_selected_os = getAnchorSelectedOS() || getDefaultSelectedOS();
var opts = {
  cuda: getPreferredCuda(default_selected_os),
  os: default_selected_os,
  pm: 'pip',
  language: 'python',
  ptbuild: 'stable',
  'torch-compile': null
};

var supportedCloudPlatforms = [
  'aws',
  'google-cloud',
  'microsoft-azure',
  'lightning-studios',
];

var os = $(".os > .option");
var package = $(".package > .option");
var language = $(".language > .option");
var cuda = $(".cuda > .option");
var ptbuild = $(".ptbuild > .option");
var torchCompile = $(".torch-compile > .option")

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
torchCompile.on("click", function() {
    selectedOption(torchCompile, this, "torch-compile")
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
  return 'cuda.x';
}

// Disable compute platform not supported on OS
function disableUnsupportedPlatforms(os) {

  if(opts.ptbuild == "preview")
    archMap = version_map.nightly
  else
    archMap = version_map.release

  for (const [arch_key, info] of archInfoMap) {
    var elems = document.querySelectorAll('[id^="'+arch_key+'"]');
    if (elems == null) {
      console.log("Failed to find element for architecture " + arch_key);
      return;
    }
    for (var i=0; i < elems.length;i++) {
      var supported = info.platforms.has(os);
      elems[i].style.textDecoration = supported ? "" : "line-through";

      // Officially supported arch but not available
      if(!archMap[elems[i].id]) {
        elems[i].style.textDecoration =  "line-through";
      }
    }
  }
}

// Change compute versions depending on build type
function changeVersion(ptbuild) {

  if(ptbuild == "preview")
    archMap = version_map.nightly
  else
    archMap = version_map.release

  for (const [arch_key, info] of archInfoMap) {
    var elems = document.querySelectorAll('[id^="'+arch_key+'"]');
    for (var i=0; i < elems.length;i++) {
      if(archMap[elems[i].id]) {
        elems[i].style.textDecoration = "";
        elems[i].children[0].textContent = info.title + " " + archMap[elems[i].id][1]
      } else {
        elems[i].style.textDecoration = "line-through";
      }
    }
  }
  var stable_element = document.getElementById("stable");
  stable_element.children[0].textContent = stable_version;
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

function getIDFromBackend(backend) {
    const idTobackendMap = {
      inductor: 'inductor',
      cgraphs : 'cudagraphs',
      onnxrt: 'onnxrt',
      openvino: 'openvino',
      tensorrt: 'tensorrt',
      tvm: 'tvm',
    };
    return idTobackendMap[backend];
}

function getPmCmd(backend) {
    const pmCmd = {
        onnxrt: 'onnxruntime',
        tvm: 'apache-tvm',
        openvino: 'openvino',
        tensorrt: 'torch-tensorrt',
    };
    return pmCmd[backend];
}

function getImportCmd(backend) {
    const importCmd = {
        onnxrt: 'import onnxruntime',
        tvm: 'import tvm',
        openvino: 'import openvino.torch',
        tensorrt: 'import torch_tensorrt'
    }
    return importCmd[backend];
}

function getInstallCommand(optionID) {
    backend = getIDFromBackend(optionID);
    pmCmd = getPmCmd(optionID);
    finalCmd = "";
   if (opts.pm == "pip") {
        finalCmd = `pip3 install ${pmCmd}`;
   }
   else if (opts.pm == "conda") {
        finalCmd = `conda install ${pmCmd}`;
   }
    return finalCmd;
}

function getTorchCompileUsage(optionId) {
    backend = getIDFromBackend(optionId);
    importCmd = getImportCmd(optionId) + "<br>";
    finalCmd = "";
    tcUsage = "# Torch Compile usage: " + "<br>";
    backendCmd = `torch.compile(model, backend="${backend}")`;
    libtorchCmd = `# Torch compile ${backend} not supported with Libtorch`;

    if (opts.pm == "libtorch") {
        return libtorchCmd;
    }
    if (backend == "inductor" || backend == "cudagraphs") {
        return tcUsage + backendCmd;
    }
    if (backend == "openvino") {
        if (opts.pm == "source") {
            finalCmd += "# Follow instructions at this URL to build openvino from source: https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md" + "<br>" ;
            tcUsage += importCmd;
        }
        else if (opts.pm == "conda") {
            tcUsage += importCmd;
        }
        if (opts.os == "windows" && !tcUsage.includes(importCmd)) {
            tcUsage += importCmd;
        }
    }
    else{
        tcUsage += importCmd;
    }
    if (backend == "onnxrt") {
        if (opts.pm == "source") {
            finalCmd += "# Follow instructions at this URL to build onnxruntime from source: https://onnxruntime.ai/docs/build" + "<br>" ;
        }
    }
    if (backend == "tvm") {
        if (opts.pm == "source") {
            finalCmd += "# Follow instructions at this URL to build tvm from source: https://tvm.apache.org/docs/install/from_source.html" + "<br>" ;
        }
    }
    if (backend == "tensorrt") {
        if (opts.pm == "source") {
            finalCmd += "# Follow instructions at this URL to build tensorrt from source: https://pytorch.org/TensorRT/getting_started/installation.html#compiling-from-source" + "<br>" ;
        }
    }
    finalCmd += tcUsage + backendCmd;
    return finalCmd
}

function addTorchCompileCommandNote(selectedOptionId) {

    if (!selectedOptionId) {
        return;
    }
    if (selectedOptionId == "inductor" || selectedOptionId == "cgraphs") {
        $("#command").append(
            `<pre> ${getTorchCompileUsage(selectedOptionId)} </pre>`
        );
    }
    else {
        $("#command").append(
            `<pre> ${getInstallCommand(selectedOptionId)} </pre>`
        );
        $("#command").append(
            `<pre> ${getTorchCompileUsage(selectedOptionId)} </pre>`
        );
    }
}

function selectedOption(option, selection, category) {
  $(option).removeClass("selected");
  $(selection).addClass("selected");
  const previousSelection = opts[category];
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
    changeVersion(opts.ptbuild);
    //make sure unsupported platforms are disabled
    disableUnsupportedPlatforms(opts.os);
  } else if (category === "torch-compile") {
    if (selection.id === previousSelection) {
      $(selection).removeClass("selected");
      opts[category] = null;
    }
  }
  commandMessage(buildMatcher());
  if (category === "os") {
    disableUnsupportedPlatforms(opts.os);
    display(opts.os, 'installation', 'os');
  }
  changeAccNoneName(opts.os);
  addTorchCompileCommandNote(opts['torch-compile'])
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
  var object = {{ installMatrix }};

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

// Set cuda version right away
changeVersion("stable")
