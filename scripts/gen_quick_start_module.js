#!/usr/bin/env python3
import json
import os
from typing import Dict
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def read_published_versions():
    with open(os.path.join(BASE_DIR, "published_versions.json")) as fp:
        return json.load(fp)


def read_quick_start_module_template():
    with open(os.path.join(BASE_DIR, "_includes", "quick-start-module.js")) as fp:
        return fp.read()


def gen_install_matrix(versions) -> Dict[str, str]:
    rc = {}
    version_map = {
        "preview": "preview",
        "lts": versions["latest_lts"],
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
    versions = read_published_versions()
    template = read_quick_start_module_template()
    versions_str = json.dumps(gen_install_matrix(versions))
    print(template.replace("{{ installMatrix }}", versions_str))


if __name__ == "__main__":
    main()
