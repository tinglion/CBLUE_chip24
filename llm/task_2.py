import json
import os
import re
import sys

sys.path.append(".")
from chip import data_loader
from conf import test_only
from llm import llm_wrapper
from llm import prompt_template_44 as prompt_template
from llm import task_2_reverse
from utils import dict_utils, file_utils, json_utils

map12 = data_loader.load_map12()


# 从临床信息推病机
def predict_2(raw_text, result1_weighted, candidate2, model_list=["openai"]):
    candidate2_map, candidate2_map_r = data_loader.parse_candidate(candidate2)
    result1 = [k for k in result1_weighted]

    sugguest2 = set()
    for name1 in result1:
        if name1 in map12:
            for t1 in map12[name1]:
                if t1 in candidate2_map_r:
                    sugguest2.add(t1)

    example2 = {v1: {"A": 1} for v1 in result1}  # , "B": 2
    prompt2 = (
        prompt_template.template_prompt2.replace("{raw_text}", raw_text)
        .replace("{result1}", ";".join(result1))
        .replace("{candidate2}", candidate2)
        .replace("{example2}", json.dumps(example2, ensure_ascii=False))
    )
    if not test_only:
        prompt2 = prompt2.replace(
            "{sugguest2}", json.dumps(list(sugguest2), ensure_ascii=False)
        )
    print(f"promp2={prompt2}")

    response2_obj = dict()
    for llm_model in model_list:
        response2_tmp = llm_wrapper.chat_complete(prompt2, llm_name=llm_model)
        response2_tmp_obj = json_utils.cvt_str_to_obj(response2_tmp)
        print(f"{llm_model}={response2_tmp_obj}")

        # weighted
        for k1_name in response2_tmp_obj:
            factor = result1_weighted[k1_name] if k1_name in result1_weighted else 0
            for k2_tag in response2_tmp_obj[k1_name]:
                response2_tmp_obj[k1_name][k2_tag] *= factor

        response2_tmp_obj2 = dict_utils.revise_dict3(
            response2_tmp_obj, candidate2_map, candidate2_map_r
        )
        response2_obj = dict_utils.merge_dict_3(
            response2_obj, response2_tmp_obj2, min_value=2
        )
    print(f"response2_obj={response2_obj}")

    result2_scored = data_loader.rank3(response2_obj)
    print(f"result2_scored={result2_scored}")

    # tag, score, name
    # score 归一化
    result2_full = [
        (r2[0], r2[1] / result2_scored[0][1], candidate2_map[r2[0]])
        for r2 in result2_scored
    ]
    print(f"result2_full={result2_full}")
    return result2_full


# 从病机角度算分
def predict_2_v2(raw_text, candidate2, model_list=["openai"]):
    candidate2_map, candidate2_map_r = data_loader.parse_candidate(candidate2)

    prompt2 = (
        prompt_template.template_prompt2_v2.replace("{raw_text}", raw_text)
        .replace("{candidate2}", candidate2)
        .replace("{example2}", json.dumps(prompt_template.example2, ensure_ascii=False))
    )
    print(f"promp2_v2={prompt2}")

    # 多次结果，对抗增强
    response2_obj = dict()
    for llm_model in model_list:
        response2_tmp = llm_wrapper.chat_complete(prompt2, llm_name=llm_model)
        response2_tmp_obj = json_utils.cvt_str_to_obj(response2_tmp)
        print(f"{llm_model}={response2_tmp_obj}")
        for k in response2_tmp_obj:
            response2_tmp_obj[k]["打分"] = (
                response2_tmp_obj[k]["显著性"]
                + response2_tmp_obj[k]["相关性"]
                + response2_tmp_obj[k]["倾向性"]
                # + response2_tmp_obj[k]["重要性"]
            ) / 3
            if k in response2_obj:
                if response2_tmp_obj[k]["打分"] > response2_obj[k]["打分"]:
                    response2_obj[k] = response2_tmp_obj[k]
            else:
                response2_obj[k] = response2_tmp_obj[k]

    #
    max_score = 0
    for k in response2_obj:
        if response2_obj[k]["打分"] > max_score:
            max_score = response2_obj[k]["打分"]

    # tag, score, name
    # score 归一化
    result2_full = [
        (k, response2_obj[k]["打分"] / max_score, candidate2_map[k])
        for k in response2_obj
    ]

    result2_full.sort(key=lambda r: -r[1])
    print(f"result2_full_v2={result2_full}")
    return result2_full


# 从提取信息推病机(有的分值特别大，0.1)，从整体得到病机（分值拉不开，0.8）
# 两者得分分别归一化后，取和/最大？
def predict_2_a(raw_text, result1_weighted, candidate2, model_list=["openai"]):
    result2_full_v1 = predict_2(
        raw_text,
        result1_weighted=result1_weighted,
        candidate2=candidate2,
        model_list=model_list,
    )
    result2_full_v2 = predict_2_v2(
        raw_text,
        candidate2=candidate2,
        model_list=model_list,
    )
    result2_reverse = task_2_reverse.predict_2_reverse(
        raw_text,
        candidate2=candidate2,
    )

    result2_map = {}
    # 1推1，更符合题目逻辑
    for r in result2_full_v1:
        result2_map[r[0]] = list(r)
    # 全推1，更全面
    for r in result2_full_v2:
        if r[0] in result2_map:
            result2_map[r[0]][1] += r[1] * 0.8
            # result2_map[r[0]][1] = max(result2_map[r[0]][1], r[1])
        else:
            result2_map[r[0]] = list(r)
    # reverse，对病史诊断病机降权
    # 这个比较难判断，汉语不明说之前诊断对错，但是模型推理会引用
    for r in result2_reverse:
        if r[0] in result2_map and result2_map[r[0]][1] > 0.81:
            result2_map[r[0]][1] -= 0.1
            
    result2_full = [result2_map[k] for k in result2_map]

    result2_full.sort(key=lambda r: -r[1])
    print(f"result2_full_a={result2_full}")
    return result2_full


if __name__ == "__main__":
    predict_2_a(
        raw_text="袁某，男，12岁。初诊:1975年11月6日。主诉及病史:患者感染白喉，住市传染病医院治疗，喉头白膜已脱落，但精神委靡，心跳缓慢，心律不齐。诊断为白喉并发中毒性心肌炎，建议中医药治疗，故邀请陆老会诊。诊查:诊见患儿面色灰白，呼吸气短，心悸胸闷，神疲乏力，自汗不止，睡眠不安，口咽干燥，胃纳呆滞。舌质红少苔，脉细弱并见结代。",
        result1_weighted={
            "感染白喉": 0.15,
            "精神委靡": 0.1,
            "心跳缓慢": 0.1,
            "心律不齐": 0.1,
            "面色灰白": 0.1,
            "呼吸气短": 0.1,
            "心悸胸闷": 0.1,
            "神疲乏力": 0.05,
            "自汗不止": 0.05,
            "睡眠不安": 0.05,
            "口咽干燥": 0.05,
            "胃纳呆滞": 0.05,
        },
        candidate2="A:外感疫毒入里;B:脾胃亏损;C:痰湿阻滞;D:肺阴亦不足;E:肝阴耗损;F:出入升降之机停废;G:胃失和降;H:腐秽留滞;I:内陷心包;J:耗损心气和心阴",
    )
