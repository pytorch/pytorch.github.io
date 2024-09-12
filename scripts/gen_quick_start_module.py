#!/usr/bin/env python3
"""
Generates quick start module for https://pytorch.org/get-started/locally/ page
If called from update-quick-start-module.yml workflow (--autogenerate parameter set)
Will output new quick-start-module.js, and new published_version.json file
based on the current release matrix.
If called standalone will generate quick-start-module.js from existing
published_version.json file
"""

import json
import copy
import argparse
from pathlib import Path
from typing import Dict
from enum import Enum

BASE_DIR =  Path(__file__).parent.parent

class OperatingSystem(Enum):
    LINUX: str = "linux"
    WINDOWS: str = "windows"
    MACOS: str = "macos"

PRE_CXX11_ABI = "pre-cxx11"
CXX11_ABI = "cxx11-abi"
DEBUG = "debug"
RELEASE = "release"
DEFAULT = "default"
ENABLE = "enable"
DISABLE = "disable"
MACOS = "macos"

# Mapping json to release matrix default values
acc_arch_ver_default = {
    "nightly": {
        "accnone": ("cpu", ""),
        "cuda.x": ("cuda", "11.8"),
        "cuda.y": ("cuda", "12.1"),
        "cuda.z": ("cuda", "12.4"),
        "rocm5.x": ("rocm", "6.0")
        },
    "release": {
        "accnone": ("cpu", ""),
        "cuda.x": ("cuda", "11.8"),
        "cuda.y": ("cuda", "12.1"),
        "cuda.z": ("cuda", "12.4"),
        "rocm5.x": ("rocm", "6.0")
        }
    }

# Initialize arch version to default values
# these default values will be overwritten by
# extracted values from the release marix
acc_arch_ver_map = acc_arch_ver_default

LIBTORCH_DWNL_INSTR = {
        PRE_CXX11_ABI: "Download here (Pre-cxx11 ABI):",
        CXX11_ABI: "Download here (cxx11 ABI):",
        RELEASE: "Download here (Release version):",
        DEBUG: "Download here (Debug version):",
        MACOS: "Download arm64 libtorch here (ROCm and CUDA are not supported):",
    }

def load_json_from_basedir(filename: str):
    try:
        with open(BASE_DIR / filename) as fptr:
            return json.load(fptr)
    except FileNotFoundError as exc:
        raise ImportError(f"File {filename} not found error: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise ImportError(f"Invalid JSON {filename}") from exc

def read_published_versions():
    return load_json_from_basedir("published_versions.json")

def write_published_versions(versions):
    with open(BASE_DIR / "published_versions.json", "w") as outfile:
        json.dump(versions, outfile, indent=2)

def read_matrix_for_os(osys: OperatingSystem, channel: str):
    jsonfile = load_json_from_basedir(f"{osys.value}_{channel}_matrix.json")
    return jsonfile["include"]

def read_quick_start_module_template():
    with open(BASE_DIR / "_includes" / "quick-start-module.js") as fptr:
        return fptr.read()

def get_package_type(pkg_key: str, os_key: OperatingSystem) -> str:
    if pkg_key != "pip":
        return pkg_key
    return "manywheel" if os_key == OperatingSystem.LINUX.value else "wheel"

def get_gpu_info(acc_key, instr, acc_arch_map):
    gpu_arch_type, gpu_arch_version = acc_arch_map[acc_key]
    if DEFAULT in instr:
        gpu_arch_type, gpu_arch_version = acc_arch_map["accnone"]
    return (gpu_arch_type, gpu_arch_version)

# This method is used for generating new published_versions.json file
# It will modify versions json object with installation instructions
# Provided by generate install matrix Github Workflow, stored in release_matrix
# json object.
def update_versions(versions, release_matrix, release_version):
    version = "preview"
    template = "preview"
    acc_arch_map = acc_arch_ver_map[release_version]

    if release_version != "nightly":
        version = release_matrix[OperatingSystem.LINUX.value][0]["stable_version"]
        if version not in versions["versions"]:
            versions["versions"][version] = copy.deepcopy(versions["versions"][template])
            versions["latest_stable"] = version

    # Perform update of the json file from release matrix
    for os_key, os_vers in versions["versions"][version].items():
        for pkg_key, pkg_vers in os_vers.items():
            for acc_key, instr in pkg_vers.items():
                package_type = get_package_type(pkg_key, os_key)
                gpu_arch_type, gpu_arch_version = get_gpu_info(acc_key, instr, acc_arch_map)

                pkg_arch_matrix = [
                    x for x in release_matrix[os_key]
                    if (x["package_type"], x["gpu_arch_type"], x["gpu_arch_version"]) ==
                    (package_type, gpu_arch_type, gpu_arch_version)
                    ]

                if pkg_arch_matrix:
                    if package_type != "libtorch":
                        instr["command"] = pkg_arch_matrix[0]["installation"]
                    else:
                        if os_key == OperatingSystem.LINUX.value:
                            rel_entry_dict = {
                                x["devtoolset"]: x["installation"] for x in pkg_arch_matrix
                                if x["libtorch_variant"] == "shared-with-deps"
                                }
                            if instr["versions"] is not None:
                                for ver in [PRE_CXX11_ABI, CXX11_ABI]:
                                    instr["versions"][LIBTORCH_DWNL_INSTR[ver]] = rel_entry_dict[ver]
                        elif os_key == OperatingSystem.WINDOWS.value:
                            rel_entry_dict = {x["libtorch_config"]: x["installation"] for x in pkg_arch_matrix}
                            if instr["versions"] is not None:
                                for ver in [RELEASE, DEBUG]:
                                     instr["versions"][LIBTORCH_DWNL_INSTR[ver]] = rel_entry_dict[ver]
                        elif os_key == OperatingSystem.MACOS.value:
                            if instr["versions"] is not None:
                                instr["versions"][LIBTORCH_DWNL_INSTR[MACOS]] = pkg_arch_matrix[0]["installation"]

# This method is used for generating new quick-start-module.js
# from the versions json object
def gen_install_matrix(versions) -> Dict[str, str]:
    result = {}
    version_map = {
        "preview": "preview",
        "stable": versions["latest_stable"],
    }
    for ver, ver_key in version_map.items():
        for os_key, os_vers in versions["versions"][ver_key].items():
            for pkg_key, pkg_vers in os_vers.items():
                for acc_key, instr in pkg_vers.items():
                    extra_key = 'python' if pkg_key != 'libtorch' else 'cplusplus'
                    key = f"{ver},{pkg_key},{os_key},{acc_key},{extra_key}"
                    note = instr["note"]
                    lines = [note] if note is not None else []
                    if pkg_key == "libtorch":
                        ivers = instr["versions"]
                        if ivers is not None:
                            lines += [f"{lab}<br /><a href='{val}'>{val}</a>" for (lab, val) in ivers.items()]
                    else:
                        command = instr["command"]
                        if command is not None:
                            lines.append(command)
                    result[key] = "<br />".join(lines)
    return result

# This method is used for extracting two latest verisons of cuda and
# last verion of rocm. It will modify the acc_arch_ver_map object used
# to update getting started page.
def extract_arch_ver_map(release_matrix):
    def gen_ver_list(chan, gpu_arch_type):
        return {
            x["desired_cuda"]: x["gpu_arch_version"]
            for x in release_matrix[chan]["linux"]
            if x["gpu_arch_type"] == gpu_arch_type
        }

    for chan in ("nightly", "release"):
        cuda_ver_list = gen_ver_list(chan, "cuda")
        rocm_ver_list = gen_ver_list(chan, "rocm")
        cuda_list = sorted(cuda_ver_list.values())
        acc_arch_ver_map[chan]["rocm5.x"] = ("rocm", max(rocm_ver_list.values()))
        for cuda_ver, label in zip(cuda_list, ["cuda.x", "cuda.y", "cuda.z"]):
            acc_arch_ver_map[chan][label] = ("cuda", cuda_ver)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autogenerate', dest='autogenerate', action='store_true')
    parser.set_defaults(autogenerate=True)

    options = parser.parse_args()
    versions = read_published_versions()

    if options.autogenerate:
        release_matrix = {}
        for val in ("nightly", "release"):
            release_matrix[val] = {}
            for osys in OperatingSystem:
                release_matrix[val][osys.value] = read_matrix_for_os(osys, val)

        extract_arch_ver_map(release_matrix)
        for val in ("nightly", "release"):
            update_versions(versions, release_matrix[val], val)

        write_published_versions(versions)

    template = read_quick_start_module_template()
    versions_str = json.dumps(gen_install_matrix(versions))
    template = template.replace("{{ installMatrix }}", versions_str)
    template = template.replace("{{ VERSION }}", f"\"Stable ({versions['latest_stable']})\"")
    print(template.replace("{{ ACC ARCH MAP }}", json.dumps(acc_arch_ver_map)))

if __name__ == "__main__":
    main()
