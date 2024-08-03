import json
import logging
import os
import platform
import sys
import traceback

sys.path.append(".")
from chip import data_loader
from utils import ee_wrapper, file_utils, openai_wrapper

logger = logging.getLogger(__name__)

data_path = (
    "D:/fdata/ai/med/CHIP2004"
    if platform.system() == "Windows"
    else "/mnt/windows/sting/data/CHIP2004"
)

# 1 临床表现信息（实体）
template_prompt1 = """您是一位中医学专家，请根据下面这段文字提取主要的临床表现信息（包括症状描述），用json格式返回。
注意严格保留原文。
病情描述：{raw_text}
严格按照输出json格式：{"临床表现信息": {sugguest1}}
"""

# 2 病机
template_prompt2 = """您是一位中医学专家，请针对每条临床表现信息，结合病情描述，从病机候选项中选择最相关的一条记录标签，严格按照json格式输出。
临床表现信息：{result1}
病情描述：{raw_text}
病机候选项：{candidate2}
严格按照输出json格式：["逸多劳少", "体弱而肥"]
"""

# 3 证候
template_prompt3 = """您是一位中医学专家，请针对每条病机信息，结合病情描述，从证候候选项中选择最相关的一条记录标签，严格按照json格式输出。
病机信息：{result2}
病情描述：{raw_text}
证候候选项：{candidate3}
输出json格式：["肺气虚", "阴虚素质"]
"""

# 4 临证体会
template_prompt4 = """您是一位中医学专家，请根据提供的病情描述、临床表现信息、病机信息、证候信息，总结临证体会和辨证，用json格式输出。
病情描述：{raw_text}
临床表现信息：{result1}
病机信息：{result2}
证候信息：{result3}
输出json格式：["临证体会":"临证体会：中医认为是外感疫毒入里，耗损心气和心阴，故出现神委气短、胸闷心悸", "辨证":"外感时疫，热毒内侵"]
"""

map23 = data_loader.load_map23()


def gen_train_set(fn_dst, i_from=0, i_to=sys.maxsize):
    raw_train = data_loader.load(f"{data_path}/round1_traning_data/train.json")

    text_list = [r.get("临床资料") for r in raw_train]
    entities_list = ee_wrapper.do_predict(text_list)

    qalist = []
    for i, r in enumerate(raw_train):
        name = r.get("案例编号", None)
        print(f"{i}={name}")
        if i < i_from:
            continue
        if i >= i_to:
            break

        raw_text = r.get("临床资料", None)

        # 一个字的不要
        entity_list = []
        for en in entities_list[i]:
            if len(en["entity"]) > 1:
                entity_list.append(en["entity"])

        # 1
        prompt1 = template_prompt1.replace("{raw_text}", raw_text).replace(
            "{sugguest1}", '["大便干"]'
        )
        result1 = r.get("信息抽取能力-核心临床信息", None).split(";")

        # 2 病机
        candidate2 = r.get("病机选项")
        prompt2 = (
            template_prompt2.replace("{raw_text}", raw_text)
            .replace("{result1}", json.dumps(result1, ensure_ascii=False))
            .replace("{candidate2}", candidate2)
        )
        result2 = r.get("信息抽取能力-核心病机", None).split(";")

        # 3 证候
        candidate3 = r.get("证候选项")
        prompt3 = (
            template_prompt3.replace("{raw_text}", raw_text)
            .replace("{result2}", json.dumps(result2, ensure_ascii=False))
            .replace("{candidate3}", candidate3)
        )
        result3 = r.get("信息抽取能力-核心证候", None).split(";")

        # 临证体会
        prompt4 = (
            template_prompt4.replace("{raw_text}", raw_text)
            .replace("{result1}", json.dumps(result1, ensure_ascii=False))
            .replace("{result2}", json.dumps(result2, ensure_ascii=False))
            .replace("{result3}", json.dumps(result3, ensure_ascii=False))
        )
        result4 = {"临证体会": r.get("临证体会", None), "辨证": r.get("辨证", None)}
        qalist.append(
            [
                (prompt1, result1),
                (prompt2, result2),
                (prompt3, result3),
                (prompt4, result4),
            ]
        )

        # debug
        # break
    with open(fn_dst, "w", encoding="utf8") as fpw:
        json.dump(qalist, fpw, ensure_ascii=False)


if __name__ == "__main__":
    gen_train_set(fn_dst="data/test_gt.json", i_from=180)
