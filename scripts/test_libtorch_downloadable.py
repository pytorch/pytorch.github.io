#!/usr/bin/env python3
# Tests libtorch is downloadable

import sys


def check_url_downloadable(url: str) -> bool:
    from urllib.request import Request, urlopen
    req = Request(url, method="HEAD")
    try:
        with urlopen(req):
            return True
    except:
        pass
    return False


def read_published_versions():
    import json
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base_dir, "published_versions.json")) as fp:
        return json.load(fp)


def main() -> None:
    versions = read_published_versions()
    for ver in [versions["latest_stable"], versions["latest_lts"]]:
        for os, os_vers in versions["versions"][ver].items():
            for acc, acc_vers in os_vers["libtorch"].items():
                vers = acc_vers["versions"]
                if vers is None:
                    continue
                if isinstance(vers, str):
                    if not check_url_downloadable(vers):
                        print(f"Can not download libtorch at url {vers}")
                        sys.exit(-1)
                    print(f"{vers} can be downloaded")
                assert isinstance(vers, dict)
                for name, url in vers.items():
                    if not check_url_downloadable(url):
                        print(f"Can not download libtorch at url {url}")
                        sys.exit(-1)
                    print(f"{url} can be downloaded")


if __name__ == "__main__":
    main()
