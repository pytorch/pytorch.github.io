#!/usr/bin/env python3
# Generates published versions based on unstructured quick-start-module

import json
from urllib.request import urlopen
from typing import Any, Dict, Optional, Union


class ConfigStr:
    version: str
    conf_type: str
    os: str
    accel: str
    extra: str

    @staticmethod
    def parse(val: str) -> "ConfigStr":
        vals = val.split(",")
        assert len(vals) == 5
        rc = ConfigStr()
        for k, v in zip(["version", "conf_type", "os", "accel", "extra"], vals):
            rc.__setattr__(k, v)
        return rc

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class LibTorchInstruction:
    note: Optional[str]
    versions: Union[Dict[str, str], str, None]

    def __init__(self, note: Optional[str] = None, versions: Union[Dict[str, str], str, None] = None) -> None:
        self.note = note
        self.versions = versions

    @staticmethod
    def parse(val: str) -> "LibTorchInstruction":
        import re
        href_pattern = re.compile("<a href=\'([^']*)\'>([^<]*)</a>")
        line_separator = "<br />"
        lines = val.split(line_separator)
        versions = {}
        idx_to_delete = set()
        for idx, line in enumerate(lines):
            url = href_pattern.findall(line)
            if len(url) == 0:
                continue
            # There should be only one URL per line and value inside and outside of URL shoudl match
            assert len(url) == 1
            assert url[0][0] == url[0][1].rstrip(), url
            versions[lines[idx - 1].strip()] = url[0][0]
            idx_to_delete.add(idx - 1)
            idx_to_delete.add(idx)
        lines = [lines[idx] for idx in range(len(lines)) if idx not in idx_to_delete]
        if len(lines) == 1 and len(lines[0]) == 0:
            lines = []
        return LibTorchInstruction(note=line_separator.join(lines) if len(lines) > 0 else None,
                                   versions=versions if len(versions) > 0 else None)

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class PyTorchInstruction:
    note: Optional[str]
    command: Optional[str]

    def __init__(self, note: Optional[str] = None, command: Optional[str] = None) -> None:
        self.note = note
        self.command = command

    @staticmethod
    def parse(val: str) -> "PyTorchInstruction":
        def is_cmd(cmd: str) -> bool:
            return cmd.startswith("pip3 install") or cmd.startswith("conda install")
        line_separator = "<br />"
        lines = val.split(line_separator)
        if is_cmd(lines[-1]):
            note = line_separator.join(lines[:-1]) if len(lines) > 1 else None
            command = lines[-1]
        elif is_cmd(lines[0]):
            note = line_separator.join(lines[1:]) if len(lines) > 1 else None
            command = lines[0]
        else:
            note = val
            command = None
        return PyTorchInstruction(note=note, command=command)

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class PublishedAccVersion:
    libtorch: Dict[str, LibTorchInstruction]
    conda: Dict[str, PyTorchInstruction]
    pip: Dict[str, PyTorchInstruction]

    def __init__(self):
        self.pip = dict()
        self.conda = dict()
        self.libtorch = dict()

    def __repr__(self) -> str:
        return self.__dict__.__repr__()

    def add_instruction(self, conf: ConfigStr, val: str) -> None:
        if conf.conf_type == "libtorch":
            self.libtorch[conf.accel] = LibTorchInstruction.parse(val)
        elif conf.conf_type == "conda":
            self.conda[conf.accel] = PyTorchInstruction.parse(val)
        elif conf.conf_type == "pip":
            self.pip[conf.accel] = PyTorchInstruction.parse(val)
        else:
            raise RuntimeError(f"Unknown config type {conf.conf_type}")


class PublishedOSVersion:
    linux: PublishedAccVersion
    macos: PublishedAccVersion
    windows: PublishedAccVersion

    def __init__(self):
        self.linux = PublishedAccVersion()
        self.macos = PublishedAccVersion()
        self.windows = PublishedAccVersion()

    def add_instruction(self, conf: ConfigStr, val: str) -> None:
        if conf.os == "linux":
            self.linux.add_instruction(conf, val)
        elif conf.os == "macos":
            self.macos.add_instruction(conf, val)
        elif conf.os == "windows":
            self.windows.add_instruction(conf, val)
        else:
            raise RuntimeError(f"Unknown os type {conf.os}")

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class PublishedVersions:
    latest_stable: str
    latest_lts: str
    versions: Dict[str, PublishedOSVersion] = dict()

    def __init__(self, latest_stable: str, latest_lts: str) -> None:
        self.latest_stable = latest_stable
        self.latest_lts = latest_lts
        self.versions = dict()

    def parse_objects(self, objects: Dict[str, str]) -> None:
        for key, val in objects.items():
            conf = ConfigStr.parse(key)
            if conf.version not in self.versions:
                self.versions[conf.version] = PublishedOSVersion()
            self.versions[conf.version].add_instruction(conf, val)
        if 'stable' in self.versions:
            self.versions[self.latest_stable] = self.versions.pop('stable')
        if 'lts' in self.versions:
            self.versions[self.latest_lts] = self.versions.pop('lts')


def get_objects(commit_hash: str = "0ba2a203045bc94d165d52e56c87ceaa463f4284") -> Dict[str, str]:
    """
    Extract install commands as they are currently hardcoded
    in pytorch.github.io/assets/quick-start-module.js
    """
    raw_base = "raw.githubusercontent.com"
    obj_start = "var object = {"
    obj_end = "};"
    with urlopen(f"https://{raw_base}/pytorch/pytorch.github.io/{commit_hash}/assets/quick-start-module.js") as url:
        raw_data = url.read().decode("latin1")
    start_idx = raw_data.find(obj_start)
    end_idx = raw_data.find(obj_end, start_idx)
    # Adjust start end end indexes
    start_idx = raw_data.find("{", start_idx, end_idx)
    end_idx = raw_data.rfind('"', start_idx, end_idx)
    if any(x < 0 for x in [start_idx, end_idx]):
        raise RuntimeError("Unexpected raw_data")
    return json.loads(raw_data[start_idx:end_idx] + '"}')


def dump_to_file(fname: str, o: Any) -> None:
    class DictEncoder(json.JSONEncoder):
        def default(self, o):
            return o.__dict__

    with open(fname, "w") as fp:
        json.dump(o, fp, indent=2, cls=DictEncoder)


def main() -> None:
    install_objects = get_objects()
    rc = PublishedVersions(latest_stable="1.9.0", latest_lts="lts-1.8.2")
    rc.parse_objects(install_objects)
    dump_to_file("published_versions.json", rc)


if __name__ == "__main__":
    main()
