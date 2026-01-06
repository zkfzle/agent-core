# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import glob
import os
import sys

HEADER = "# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.\n"


def needs_header(path: str):
    """Checks if header is needed"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                continue
            if line.strip() in ["#!/usr/bin/env python", "# coding: utf-8"]:
                return True
            return not line.strip().startswith("#")
    return False


def add_header(path):
    """Add header if header is needed"""
    with open(path, "r+", encoding="utf-8") as f:
        content = f.read().replace("#!/usr/bin/env python\n", "").replace("# coding: utf-8\n", "")
        f.seek(0)
        f.write(HEADER + content)


def main():
    """Main entrypoint"""
    if len(sys.argv) != 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} '<glob>'")
        sys.exit(1)

    pattern = sys.argv[1]
    add_counter = 0
    for file in glob.glob(pattern, recursive=True):
        if os.path.isfile(file) and needs_header(file):
            print(f"Adding header to {file}")
            add_counter += 1
            add_header(file)
    print(f"Added header to {add_counter} files")


if __name__ == "__main__":
    main()
