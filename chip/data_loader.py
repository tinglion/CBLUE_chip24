import json
import logging
import os
import platform
import sys

sys.path.append(".")

logger = logging.getLogger(__name__)

data_path = (
    "D:/fdata/ai/med/CHIP2004"
    if platform.system() == "Windows"
    else "/mnt/windows/sting/data/CHIP2004"
)


def load(fn):
    data = []
    with open(fn, "r", encoding="utf8") as fp:
        data = json.load(fp)
        fp.close()
    return data

    # {
    #     "案例编号": "",
    #     "临床资料": "",
    #     "信息抽取能力-核心临床表现信息": "",
    #     "推理能力-病机推断": "",
    #     "信息抽取能力-核心病机": "",
    #     "病机答案": "",
    #     "病机选项": "",
    #     "推理能力-证候推断": "",
    #     "信息抽取能力-核心证候": "",
    #     "证候答案": "",
    #     "证候选项": "",
    #     "辨证": ""
    # }


def parse_candidate(candidate: str):
    l = {}
    l2 = {}
    segs = candidate.split(";")
    for seg in segs:
        words = seg.split(":")
        l[words[0]] = words[1]
        l2[words[1]] = words[0]
    return l, l2


# [
#     {"A": 10},
#     {"B": 9},
#     {"F": 8},
#     {"J": 7},
#     {"D": 6},
#     {"I": 5},
# ]
# 1、>0,
# 2、保留最高得分
def get_top(response_obj, top=3):
    sorted_data = sorted(response_obj, key=lambda x: -next(iter(x.values())))

    result2 = []
    max_v = 0
    for r2 in sorted_data:
        for k in r2:
            if r2[k] > max_v:
                max_v = r2[k]

    for r2 in sorted_data:
        for k in r2:
            if r2[k] > 0 and r2[k] >= max_v:
                result2.append(k)
    return result2


# [{'name':['tag',score]}]
def rank(data, min_number=3):
    if len(data) < 1:
        return []
    rlist = {}
    # merge
    for i, k in enumerate(data):
        # print(f"{i}={k}")
        tag = data[k][0]
        if tag not in rlist:
            rlist[tag] = 0
        rlist[tag] += data[k][1]
    # sort
    total_sum = sum(rlist.values())
    sorted_items = sorted(rlist.items(), key=lambda item: -item[1])
    # print(sorted_items)
    return sorted_items


# {'A':{'a':4}, 'B':{'b': 5, 'c':2}, 'C':{'c': 3}}
# => [('b',5), ('c', 5), ('a', 4)]
def rank3(data, min_number=3):
    if len(data) < 1:
        return []
    rlist = {}
    # merge
    for k1 in data:
        for k2 in data[k1]:
            if k2 not in rlist:
                rlist[k2] = 0
            rlist[k2] += data[k1][k2]
    # sort
    sorted_items = sorted(rlist.items(), key=lambda item: -item[1])
    # print(sorted_items)
    return sorted_items


def filter(sorted_items, max_number=4, min_number=1, filter_ratio=0.3):
    # if len(sorted_items) <= min_number:
    #     return sorted_items
    # filter
    th = sorted_items[0][1] * filter_ratio
    print(f"threshold={th}")
    filtered_items = [item for item in sorted_items if item[1] > th]
    # print(filtered_items)
    if len(filtered_items) > max_number:
        filtered_items = filtered_items[0:max_number]
    return filtered_items


def load_map23(fn="data/map23.json", src=f"{data_path}/round1_traning_data/train.json"):
    map23 = {}
    if os.path.exists(fn):
        with open(fn, "r", encoding="utf8") as fp:
            map23 = json.load(fp)
            fp.close()
    else:
        data_raw = data_loader.load(src)
        for i, r in enumerate(data_raw):
            text = r.get("推理能力-证候推断", None)
            segs = text.split(";")
            for seg in segs:
                pair = seg.split(":")
                k = pair[0]
                v = pair[1]
                if k not in map23:
                    map23[k] = []
                map23[k].append(v)
        with open(fn, "w", encoding="utf8") as fp:
            json.dump(map23, fp, ensure_ascii=False)
            fp.close()

    return map23


def load_map12(fn="data/map12.json", src=f"{data_path}/round1_traning_data/train.json"):
    newmap = {}
    if os.path.exists(fn):
        with open(fn, "r", encoding="utf8") as fp:
            newmap = json.load(fp)
            fp.close()
    else:
        data_raw = load(src)
        for i, r in enumerate(data_raw):
            text = r.get("推理能力-病机推断", None)
            segs = text.split(";")
            for seg in segs:
                pair = seg.split(":")
                k = pair[0]
                v = pair[1]
                if k not in newmap:
                    newmap[k] = []
                newmap[k].append(v)
        with open(fn, "w", encoding="utf8") as fp:
            json.dump(newmap, fp, ensure_ascii=False)
            fp.close()

    return newmap

if __name__ == "__main__":
    # data_list = [
    #     {
    #         "发热": ["C", 2],
    #         "浮肿": ["G", 2],
    #         "嗜睡": ["E", 3],
    #         "鼻衄": ["E", 3],
    #         "恶心": ["H", 2],
    #         "呕吐": ["H", 2],
    #         "尿少": ["G", 2],
    #         "明显消瘦": ["G", 2],
    #         "皮肤干燥": ["E", 2],
    #         "鼻翼煽动": ["G", 2],
    #         "呼吸困难": ["E", 3],
    #         "心律不齐": ["E", 3],
    #         "呕吐咖啡样物": ["H", 3],
    #         "大便1日数次，呈柏油样便": ["H", 3],
    #         "呕血": ["H", 3],
    #         "呼吸慢而不整": ["E", 3],
    #         "面色晦暗": ["E", 3],
    #         "嗜睡衰竭状态": ["E", 3],
    #         "时有恶 心呕吐": ["H", 2],
    #         "呼吸深长而慢": ["E", 3],
    #         "脉沉细微弱无力而迟": ["E", 3],
    #         "舌嫩润齿痕尖微赤": ["G", 2],
    #         "苔薄白干中心微黄": ["H", 2],
    #     },
    #     {
    #         "干咳": ["I", 2],
    #         "入夜尤甚": ["I", 2],
    #         "咳时无痰": ["I", 2],
    #         "胸中闷胀": ["I", 1],
    #         "唇舌及咽喉灼干": ["D", 2],
    #         "声音略带嘶哑": ["D", 1],
    #         "心烦": ["D", 1],
    #         "食欲减退": ["J", 1],
    #         "无苔": ["I", 2],
    #         "脉数无力": ["D", 1],
    #     },
    # ]
    # for data in data_list:
    #     print(rank(data))

    load_map12()
    # load_map23()
