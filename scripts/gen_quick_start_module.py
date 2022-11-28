#!/usr/bin/env python3
import json
import os
import argparse
import io
import sys
from pathlib import Path
from typing import Dict, Set, List, Iterable
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

# Mapping json to release matrix is here for now
# TBD drive the mapping via:
#  1. Scanning release matrix and picking 2 latest cuda versions and 1 latest rocm
#  2. Possibility to override the scanning algorithm with arguments passed from workflow

acc_arch_ver_map = {
    "nightly": {
        "accnone": ("cpu", ""),
        "cuda.x": ("cuda", "11.6"),
        "cuda.y": ("cuda", "11.7"),
        "rocm5.x": ("rocm", "5.2")
        },
    "release": {
        "accnone": ("cpu", ""),
        "cuda.x": ("cuda", "11.6"),
        "cuda.y": ("cuda", "11.7"),
        "rocm5.x": ("rocm", "5.2")
        }
    }

LIBTORCH_DWNL_INSTR = {
        PRE_CXX11_ABI: "Download here (Pre-cxx11 ABI):",
        CXX11_ABI: "Download here (cxx11 ABI):",
        RELEASE: "Download here (Release version):",
        DEBUG: "Download here (Debug version):",
    }

def load_json_from_basedir(filename: str):
    try:
        with open(BASE_DIR / filename) as fp:
            return json.load(fp)
    except FileNotFoundError as e:
        raise ImportError(f"File {filname} not found error: {e.strerror}") from e
    except json.JSONDecodeError as e:
        raise ImportError(f"Invalid JSON {filname} error: {e.strerror}") from e

def read_published_versions():
    return load_json_from_basedir("published_versions.json")

def write_published_versions(versions):
    with open(BASE_DIR / "published_versions.json", "w") as outfile:
            json.dump(versions, outfile, indent=2)

def read_matrix_for_os(osys: OperatingSystem, value: str):
    jsonfile = load_json_from_basedir(f"{osys.value}_{value}_matrix.json")
    return jsonfile["include"]

def read_quick_start_module_template():
    with open(BASE_DIR / "_includes" / "quick-start-module.js") as fp:
        return fp.read()

def get_package_type(pkg_key, os_key):
    package_type = pkg_key
    if pkg_key == "pip":
        package_type = "manywheel" if os_key == OperatingSystem.LINUX.value else "wheel"
    return package_type

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
    acc_arch_map = acc_arch_ver_map[release_version]

    if(release_version != "nightly"):
        version = release_matrix[OperatingSystem.LINUX.value][0]["stable_version"]
        if version not in versions["versions"]:
            import copy
            new_version = copy.deepcopy(versions["versions"]["preview"])
            versions["versions"][version] = new_version
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
                    ] or None

                if pkg_arch_matrix is not None:
                    if package_type != "libtorch":
                        instr["command"] = pkg_arch_matrix[0]["installation"]
                    else:
                        if os_key == OperatingSystem.LINUX.value:
                            rel_entry_pre_cxx1 = next(
                                x for x in pkg_arch_matrix
                                if x["devtoolset"] == PRE_CXX11_ABI
                             )
                            rel_entry_cxx1_abi = next(
                                x for x in pkg_arch_matrix
                                if x["devtoolset"] == CXX11_ABI
                                )
                            if(instr["versions"] is not None):
                                instr["versions"][LIBTORCH_DWNL_INSTR[PRE_CXX11_ABI]] = rel_entry_pre_cxx1["installation"]
                                instr["versions"][LIBTORCH_DWNL_INSTR[CXX11_ABI]] = rel_entry_cxx1_abi["installation"]
                        elif os_key == OperatingSystem.WINDOWS.value:
                            rel_entry_release = next(
                                x for x in pkg_arch_matrix
                                if x["libtorch_config"] == RELEASE
                                )
                            rel_entry_debug = next(
                                x for x in pkg_arch_matrix
                                if x["libtorch_config"] == DEBUG
                                )
                            if(instr["versions"] is not None):
                                instr["versions"][LIBTORCH_DWNL_INSTR[RELEASE]] = rel_entry_release["installation"]
                                instr["versions"][LIBTORCH_DWNL_INSTR[DEBUG]] = rel_entry_debug["installation"]


def gen_install_matrix(versions) -> Dict[str, str]:
    rc = {}
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
                   rc[key] = "<br />".join(lines)
    return rc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autogenerate', dest='autogenerate', action='store_true')
    parser.set_defaults(autogenerate=False)

    options = parser.parse_args()
    versions = read_published_versions()

    if options.autogenerate:
        release_matrix = {}
        for val in ("nightly", "release"):
            release_matrix[val] = {}
            for osys in OperatingSystem:
                release_matrix[val][osys.value] = read_matrix_for_os(osys, val)

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
