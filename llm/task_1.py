import json
import os
import re
import sys

sys.path.append(".")
from llm import llm_wrapper
from llm import prompt_template_44 as prompt_template
from utils import dict_utils, file_utils, json_utils

separators = "[，‌。‌：‌；、‌:;]"


def seg_sentence(raw_text):
    result = set()

    s = raw_text
    s = (
        s.replace("刻诊", "")
        .replace("自诉", "")
        .replace("诊其", "")
        .replace("诊时", "")
        .replace("症见", "")
        .replace("为甚", "")
        .replace("经常", "")
        .replace("时常", "")
        .replace("常常", "")
        .replace("平常", "")
        .replace("平时", "")
        .replace("平素", "")
        .replace("素有", "")
        .replace("向有", "")
        .replace("每次", "")
        .replace("患者", "")
        .replace("骤然", "")
        .replace("突然", "")
        .replace("则", "，")
        .replace("而", "，")
    )
    raw_split = re.split(separators, raw_text) + re.split(separators, s)
    for seg in raw_split:
        if not seg or len(seg) <= 1:
            continue
        result.add(seg)

        if seg.find("伴") == 0:
            result.add(seg[1:])
        if seg.find("(") >= 0:
            result.add(re.sub(r"\(.*?\)", "", seg))
        if seg.find("（") >= 0:
            result.add(re.sub(r"\（.*?\）", "", seg))
    return result


def predict_1(raw_text, model_list=["openai"], entity_list=[]):
    result1_full = seg_sentence(raw_text)
    result1_weighted = dict()

    prompt1 = prompt_template.template_prompt1.replace("{raw_text}", raw_text)
    prompt1 = prompt1.replace(
        "{sugguest1}",
        json.dumps(entity_list, ensure_ascii=False),
    )
    print(f"prompt1={prompt1}")

    for llm_model in model_list:
        response1 = llm_wrapper.chat_complete(prompt1, llm_name=llm_model)
        response1_obj = json_utils.cvt_str_to_obj(response1)
        print(f"{llm_model}={response1}")

        response1_map = response1_obj.get("临床表现信息", {})
        for k in response1_map:
            result1_full.add(k)

        if llm_model == "openai":
            result1_weighted = response1_map
    print(f"result1_full={result1_full}")
    print(f"result1_weighted={result1_weighted}")
    return result1_full, result1_weighted


if __name__ == "__main__":
    predict_1(
        "李某，男，43岁。初诊：1972年2月9日。主诉及病史：右眼眉无故脱落，已2年余。先是逐渐脱落，约1年多时间掉去3/4，仅余内端少许。近数月来右眉脱落殆尽。左眉依旧。经中西药物治疗均未见效。诊查：舌质微淡，脉小濡，两尺略差。头顶光秃，右眉全无。余无明显异状可见。"
    )
