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

let version_map={"nightly": {"accnone": ["cpu", ""], "cuda.x": ["cuda", "12.6"], "cuda.y": ["cuda", "12.8"], "cuda.z": ["cuda", "12.9"], "rocm5.x": ["rocm", "6.4"]}, "release": {"accnone": ["cpu", ""], "cuda.x": ["cuda", "11.8"], "cuda.y": ["cuda", "12.6"], "cuda.z": ["cuda", "12.8"], "rocm5.x": ["rocm", "6.3"]}}
let stable_version="Stable (2.7.1)";

var default_selected_os = getAnchorSelectedOS() || getDefaultSelectedOS();
var opts = {
  cuda: getPreferredCuda(default_selected_os),
  os: default_selected_os,
  pm: 'pip',
  language: 'python',
  ptbuild: 'stable',
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
    changeVersion(opts.ptbuild);
    //make sure unsupported platforms are disabled
    disableUnsupportedPlatforms(opts.os);
  }
  commandMessage(buildMatcher());
  if (category === "os") {
    disableUnsupportedPlatforms(opts.os);
    display(opts.os, 'installation', 'os');
  }
  changeAccNoneName(opts.os);
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
  var object = {"preview,pip,linux,accnone,python": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu", "preview,pip,linux,cuda.x,python": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126", "preview,pip,linux,cuda.y,python": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128", "preview,pip,linux,cuda.z,python": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129", "preview,pip,linux,rocm5.x,python": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4", "preview,libtorch,linux,accnone,cplusplus": "Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,libtorch,linux,cuda.x,cplusplus": "Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu126/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu126/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,libtorch,linux,cuda.y,cplusplus": "Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu128/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu128/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,libtorch,linux,cuda.z,cplusplus": "Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu129/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu129/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,libtorch,linux,rocm5.x,cplusplus": "Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/nightly/rocm6.4/libtorch-cxx11-abi-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/rocm6.4/libtorch-cxx11-abi-shared-with-deps-latest.zip</a>", "preview,pip,macos,cuda.x,python": "# CUDA is not available on MacOS, please use default package<br />pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu", "preview,pip,macos,cuda.y,python": "# CUDA is not available on MacOS, please use default package<br />pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu", "preview,pip,macos,cuda.z,python": "# CUDA is not available on MacOS, please use default package<br />pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu", "preview,pip,macos,rocm5.x,python": "# ROCm is not available on MacOS, please use default package<br />pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu", "preview,pip,macos,accnone,python": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu", "preview,libtorch,macos,accnone,cplusplus": "Download arm64 libtorch here (ROCm and CUDA are not supported):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip</a>", "preview,libtorch,macos,cuda.x,cplusplus": "Download arm64 libtorch here (ROCm and CUDA are not supported):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip</a>", "preview,libtorch,macos,cuda.y,cplusplus": "Download arm64 libtorch here (ROCm and CUDA are not supported):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip</a>", "preview,libtorch,macos,cuda.z,cplusplus": "Download arm64 libtorch here (ROCm and CUDA are not supported):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip</a>", "preview,libtorch,macos,rocm5.x,cplusplus": "Download arm64 libtorch here (ROCm and CUDA are not supported):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip</a>", "preview,pip,windows,accnone,python": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu", "preview,pip,windows,cuda.x,python": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126", "preview,pip,windows,cuda.y,python": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128", "preview,pip,windows,cuda.z,python": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129", "preview,pip,windows,rocm5.x,python": "<b>NOTE:</b> ROCm is not available on Windows", "preview,libtorch,windows,accnone,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-debug-latest.zip</a>", "preview,libtorch,windows,cuda.x,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu126/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu126/libtorch-win-shared-with-deps-latest.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu126/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu126/libtorch-win-shared-with-deps-debug-latest.zip</a>", "preview,libtorch,windows,cuda.y,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu128/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu128/libtorch-win-shared-with-deps-latest.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu128/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu128/libtorch-win-shared-with-deps-debug-latest.zip</a>", "preview,libtorch,windows,cuda.z,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu129/libtorch-win-shared-with-deps-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu129/libtorch-win-shared-with-deps-latest.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/nightly/cu129/libtorch-win-shared-with-deps-debug-latest.zip'>https://download.pytorch.org/libtorch/nightly/cu129/libtorch-win-shared-with-deps-debug-latest.zip</a>", "preview,libtorch,windows,rocm5.x,cplusplus": "<b>NOTE:</b> ROCm is not available on Windows", "stable,pip,linux,accnone,python": "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", "stable,pip,linux,cuda.x,python": "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", "stable,pip,linux,cuda.y,python": "pip3 install torch torchvision torchaudio", "stable,pip,linux,cuda.z,python": "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128", "stable,pip,linux,rocm5.x,python": "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3", "stable,libtorch,linux,accnone,cplusplus": "Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip</a>", "stable,libtorch,linux,cuda.x,cplusplus": "Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu118.zip'>https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu118.zip</a>", "stable,libtorch,linux,cuda.y,cplusplus": "Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu126.zip'>https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu126.zip</a>", "stable,libtorch,linux,cuda.z,cplusplus": "Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip'>https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip</a>", "stable,libtorch,linux,rocm5.x,cplusplus": "Download here (cxx11 ABI):<br /><a href='https://download.pytorch.org/libtorch/rocm6.3/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Brocm6.3.zip'>https://download.pytorch.org/libtorch/rocm6.3/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Brocm6.3.zip</a>", "stable,pip,macos,cuda.x,python": "# CUDA is not available on MacOS, please use default package<br />pip3 install torch torchvision torchaudio", "stable,pip,macos,cuda.y,python": "# CUDA is not available on MacOS, please use default package<br />pip3 install torch torchvision torchaudio", "stable,pip,macos,cuda.z,python": "# CUDA is not available on MacOS, please use default package<br />pip3 install torch torchvision torchaudio", "stable,pip,macos,rocm5.x,python": "# ROCm is not available on MacOS, please use default package<br />pip3 install torch torchvision torchaudio", "stable,pip,macos,accnone,python": "pip3 install torch torchvision torchaudio", "stable,libtorch,macos,accnone,cplusplus": "Download arm64 libtorch here (ROCm and CUDA are not supported):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip</a>", "stable,libtorch,macos,cuda.x,cplusplus": "Download arm64 libtorch here (ROCm and CUDA are not supported):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip</a>", "stable,libtorch,macos,cuda.y,cplusplus": "Download arm64 libtorch here (ROCm and CUDA are not supported):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip</a>", "stable,libtorch,macos,cuda.z,cplusplus": "Download arm64 libtorch here (ROCm and CUDA are not supported):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip</a>", "stable,libtorch,macos,rocm5.x,cplusplus": "Download arm64 libtorch here (ROCm and CUDA are not supported):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip</a>", "stable,pip,windows,accnone,python": "pip3 install torch torchvision torchaudio", "stable,pip,windows,cuda.x,python": "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", "stable,pip,windows,cuda.y,python": "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126", "stable,pip,windows,cuda.z,python": "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128", "stable,pip,windows,rocm5.x,python": "<b>NOTE:</b> ROCm is not available on Windows", "stable,libtorch,windows,accnone,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.7.1%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.7.1%2Bcpu.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.7.1%2Bcpu.zip'>https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.7.1%2Bcpu.zip</a>", "stable,libtorch,windows,cuda.x,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.7.1%2Bcu118.zip'>https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.7.1%2Bcu118.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-debug-2.7.1%2Bcu118.zip'>https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-debug-2.7.1%2Bcu118.zip</a>", "stable,libtorch,windows,cuda.y,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-2.7.1%2Bcu126.zip'>https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-2.7.1%2Bcu126.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-debug-2.7.1%2Bcu126.zip'>https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-debug-2.7.1%2Bcu126.zip</a>", "stable,libtorch,windows,cuda.z,cplusplus": "Download here (Release version):<br /><a href='https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.7.1%2Bcu128.zip'>https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.7.1%2Bcu128.zip</a><br />Download here (Debug version):<br /><a href='https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.7.1%2Bcu128.zip'>https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.7.1%2Bcu128.zip</a>", "stable,libtorch,windows,rocm5.x,cplusplus": "<b>NOTE:</b> ROCm is not available on Windows"};

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

