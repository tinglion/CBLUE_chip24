import json
import logging
import os
import platform
import re
import sys
import traceback

import jieba
from rouge import Rouge

sys.path.append(".")
from conf import data_path
from chip import data_loader

logger = logging.getLogger(__name__)


def load_lines(fn):
    with open(fn, "r", encoding="utf8") as fp:
        lines = fp.readlines()
        fp.close()
        return lines
    return []


def doit(result1, result2, dst, col=[1]):
    dst_lines = []

    lines1 = load_lines(result1)
    lines2 = load_lines(result2)
    for i, line in enumerate(lines1):
        segs = line.split("@")
        segs2 = lines2[i].split("@")
        # print(segs2)
        dst_lines.append(f"{segs[0]}@{segs2[1]}@{segs[2]}@{segs[3]}@{segs2[4]}")
    with open(dst, "w", encoding="utf8") as fp:
        fp.writelines(dst_lines)
        fp.close()


if __name__ == "__main__":
    doit(
        result1=f"./data/NE_B_41.txt",
        result2=f"./data/NE_B_43.txt",
        dst=f"./temp/B_fuse_41_43_14.txt",
    )
