// Keys are Substrings as diplayed by navigator.platform
var supportedOperatingSystems = new Map([
  ['linux', 'linux'],
  ['mac', 'macos'],
  ['win', 'windows'],
]);

var supportedComputePlatforms = new Map([
  ['cuda.x', new Set(['linux', 'windows'])],
  ['cuda.y', new Set(['linux', 'windows'])],
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
  return 'cuda11.6';
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
  var cuda_element_x = document.getElementById("cuda.x");
  var cuda_element_y = document.getElementById("cuda.y");
  var rocm_element = document.getElementById("rocm5.x");
  if (cuda_element_x == null || cuda_element_y == null) {
    console.log("Failed to find cuda11 elements");
    return;
  }
  if (cuda_element_x.childElementCount != 1 || cuda_element_y.childElementCount != 1) {
    console.log("Unexpected number of children for cuda11 element");
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
    rocm_element.children[0].textContent = "ROCm 5.2";
    cuda_element_x.children[0].textContent = "CUDA 11.6";
    cuda_element_y.children[0].textContent = "CUDA 11.7";
  } else {
    rocm_element.children[0].textContent = "ROCm 5.2";
    cuda_element_x.children[0].textContent = "CUDA 10.2";
    cuda_element_y.children[0].textContent = "CUDA 11.1";
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
