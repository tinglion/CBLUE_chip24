import json
import logging
import os
import platform
import sys
import traceback

sys.path.append(".")
from chip import data_loader
from utils import (
    dict_utils,
    ee_wrapper,
    file_utils,
    json_utils,
    local_wrapper,
    openai_wrapper,
    qwen_wrapper,
)

logger = logging.getLogger(__name__)

data_path = (
    "D:/fdata/ai/med/CHIP2004"
    if platform.system() == "Windows"
    else "/mnt/windows/sting/data/CHIP2004"
)
map23 = data_loader.load_map23()

# 1 临床表现信息（实体）：，不包括体温等指标型数据 已知部分信息：{sugguest1}
template_prompt1 = """您是一位中医学专家，请根据下面这段文字提取全面的临床表现信息，严格按照json格式返回。
注意严格保留原文。
病情描述：{raw_text}
严格按照输出json格式：{"临床表现信息": ["大便干", "咳嗽", "脉滑", "舌苔微黄"]}
"""

# 2 病机 {"唾痰": ["A", 0]}
template_prompt2 = """您是一位中医学专家，请针对每条临床表现信息，结合病情描述，从病机候选项中选择最相关的一条记录标签（A~Z），并根据相关性和重要性打分，分值范围0、1、2或者3，分值越高重要性越高，严格按照json格式输出。
临床表现信息：{result1}

病情描述：{raw_text}

病机候选项：{candidate2}

严格按照输出json格式：{example2}
"""

# 3 证候 {"肾水不足": ["B", 0]}
template_prompt3 = """您是一位中医学专家，请针对每条病机信息，结合病情描述，从证候候选项中，选择最相关的一条记录标签（A~Z），并根据相关性和重要性打分，分值范围0、1、2或者3，分值越高重要性越高，严格按照json格式输出。
病机信息：{result2}

病情描述：{raw_text}

证候候选项：{candidate3}

参考证候推理结果：{sugguest3}

严格按照输出json格式：{example3}
"""

# 4 临证体会
template_prompt4 = """您是一位中医学专家，请根据提供的病情描述、临床表现信息、病机信息、证候信息，总结临证体会和辨证，不要复述症状和病机，不要给出治疗方案，严格按照json格式输出。
病情描述：{raw_text}
病机信息：{result2}
证候信息：{result3}
严格按照输出json格式：{example4}
"""
template_example4 = {
    "临证体会": "string",
    "辨证": "string",
}


def predict_1(raw_text, entity_list):
    prompt1 = template_prompt1.replace("{raw_text}", raw_text).replace(
        "{sugguest1}",
        json.dumps(entity_list, ensure_ascii=False),
    )
    print(f"prompt1={prompt1}")
    response1 = local_wrapper.chat_complete(prompt1)
    # print(f"response={response1}")
    response1_obj = json_utils.cvt_str_to_obj(response1)
    result1 = response1_obj.get("临床表现信息", [])
    print(f"result1={result1}")
    return result1


def revise_dict(response_obj, cmap, cmap_r):
    response_obj2 = {}
    for k1 in response_obj:
        k2 = response_obj[k1][0]
        if k2 not in cmap and k2 in cmap_r:
            k2 = cmap_r[k2]
        response_obj2[k1] = [k2, response_obj[k1][1]]
    print(response_obj2)
    return response_obj2


def predict_2(raw_text, result1, candidate2):
    candidate2_map, candidate2_map_r = data_loader.parse_candidate(candidate2)

    example2 = {v1: ["A", 0] for v1 in result1}
    prompt2 = (
        template_prompt2.replace("{raw_text}", raw_text)
        .replace("{result1}", ";".join(result1))
        .replace("{candidate2}", candidate2)
        .replace("{example2}", json.dumps(example2, ensure_ascii=False))
    )
    print(f"promp2={prompt2}")
    example2s = {result1[0]: ["A", 0]}
    prompt2s = (
        template_prompt2.replace("{raw_text}", raw_text)
        .replace("{result1}", ";".join(result1))
        .replace("{candidate2}", candidate2)
        .replace("{example2}", json.dumps(example2s, ensure_ascii=False))
    )
    print(f"promp2s={prompt2s}")

    response2_local = local_wrapper.chat_complete(prompt2s)
    response2_local_obj = json_utils.cvt_str_to_obj(response2_local)
    response2_local_obj2 = revise_dict(
        response2_local_obj, candidate2_map, candidate2_map_r
    )

    response2_gpt = openai_wrapper.chat_complete(prompt2)
    response2_gpt_obj = json_utils.cvt_str_to_obj(response2_gpt)
    response2_gpt_obj2 = revise_dict(
        response2_gpt_obj, candidate2_map, candidate2_map_r
    )

    response2_qwen = qwen_wrapper.chat_complete(prompt2)
    response2_qwen_obj = json_utils.cvt_str_to_obj(response2_qwen)
    response2_qwen_obj2 = revise_dict(
        response2_qwen_obj, candidate2_map, candidate2_map_r
    )

    # merge
    response2_obj = dict_utils.merge_dict_2(
        dict_utils.merge_dict_2(response2_gpt_obj2, response2_qwen_obj2),
        response2_local_obj2,
    )
    print(f"merged={response2_obj}")
    result2_scored = data_loader.rank(response2_obj)

    result2_full = [(r2[0], r2[1], candidate2_map[r2[0]]) for r2 in result2_scored]
    print(result2_full)
    return result2_full


def predict_3(raw_text, result2_full, candidate3):
    result2_name = [r2[2] for r2 in result2_full]
    candidate3_map, candidate3_map_r = data_loader.parse_candidate(candidate3)

    # 根据训练集合得到的结果，但是未必在候选项里面
    sugguest3 = []
    for name2 in result2_name:
        if name2 in map23 and name2 in candidate3_map_r:
            sugguest3 += map23[name2]

    example3 = {v2: ["A", 0] for v2 in result2_name}
    prompt3 = (
        template_prompt3.replace("{raw_text}", raw_text)
        .replace("{example3}", json.dumps(example3, ensure_ascii=False))
        .replace("{result2}", ";".join(result2_name))
        .replace("{candidate3}", candidate3)
        .replace("{sugguest3}", json.dumps(sugguest3, ensure_ascii=False))
    )
    print(f"promp3={prompt3}")
    example3s = {result2_name[0]: ["A", 0]}
    prompt3s = (
        template_prompt3.replace("{raw_text}", raw_text)
        .replace("{example3}", json.dumps(example3s, ensure_ascii=False))
        .replace("{result2}", ";".join(result2_name))
        .replace("{candidate3}", candidate3)
        .replace("{sugguest3}", json.dumps(sugguest3, ensure_ascii=False))
    )
    print(f"promp3s={prompt3s}")

    response3_local = local_wrapper.chat_complete(prompt3s)
    response3_local_obj = json_utils.cvt_str_to_obj(response3_local)
    response3_local_obj2 = revise_dict(
        response3_local_obj, candidate3_map, candidate3_map_r
    )

    response3_gpt = openai_wrapper.chat_complete(prompt3)
    response3_gpt_obj = json_utils.cvt_str_to_obj(response3_gpt)
    response3_gpt_obj2 = revise_dict(
        response3_gpt_obj, candidate3_map, candidate3_map_r
    )

    response3_qwen = qwen_wrapper.chat_complete(prompt3)
    response3_qwen_obj = json_utils.cvt_str_to_obj(response3_qwen)
    response3_qwen_obj2 = revise_dict(
        response3_qwen_obj, candidate3_map, candidate3_map_r
    )

    response3_obj = dict_utils.merge_dict_2(
        dict_utils.merge_dict_2(response3_gpt_obj2, response3_qwen_obj2),
        response3_local_obj2,
    )
    print(response3_obj)

    result3_scored = data_loader.rank(response3_obj)
    result3_full = [
        (r3[0], r3[1], candidate3_map[r3[0]])
        for r3 in result3_scored
        if r3[0] in candidate3_map
    ]
    print(result3_full)
    return result3_full


def predict_4(raw_text, result2_full, result3_full):
    prompt4 = (
        template_prompt4.replace("{raw_text}", raw_text)
        .replace("{example4}", json.dumps(template_example4, ensure_ascii=False))
        .replace("{result2}", ";".join([r2[2] for r2 in result2_full]))
        .replace("{result3}", ";".join([r3[2] for r3 in result3_full]))
    )
    print(f"prompt4={prompt4}")
    response4 = local_wrapper.chat_complete(prompt4)
    print(f"response4={response4}")
    response4_obj = json_utils.cvt_str_to_obj(response4)
    result4 = "临证体会：%s。辨证：%s" % (
        response4_obj.get("临证体会", ""),
        response4_obj.get("辨证", ""),
    )
    return result4


def predict(fn, fn_dst=None, i_from=0, i_to=sys.maxsize):
    result_lines = []

    data_test = data_loader.load(fn)
    text_list = [r.get("临床资料") for r in data_test]
    entities_list = ee_wrapper.do_predict(text_list)

    #  一个字的不要，仅作参考（包括但不限于）
    for i, r in enumerate(data_test):
        result_final = ""
        try:
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
            result1 = predict_1(
                raw_text,
                entity_list=entity_list if entity_list else ["大便干"],
            )

            # 2 病机 [(A,2,肾阴虚)]
            result2_full = predict_2(
                raw_text=raw_text,
                result1=result1,
                candidate2=r.get("病机选项"),
            )

            # 3 证候 [(A,2,肾阴虚)]
            result3_full = predict_3(
                raw_text,
                result2_full=result2_full,
                candidate3=r.get("证候选项"),
            )

            # 临证体会
            result4 = predict_4(
                raw_text,
                result2_full=result2_full,
                result3_full=result3_full,
            )

            #
            result_final = "%s@%s@%s@%s@%s" % (
                name,
                ";".join(result1),
                ";".join([r2[0] for r2 in result2_full]),
                ";".join([r3[0] for r3 in result3_full]),
                result4,
            )
            print(result_final)
            result_lines.append(result_final)
        except Exception as e:
            print(f"ERROR name={name} i={i}")
            logger.error(e)
            traceback.print_exc()

        if fn_dst:
            with open(fn_dst, "a", encoding="utf8") as fpr:
                fpr.write(f"{result_final}\n")
                fpr.close()
        # debug
        # break


if __name__ == "__main__":
    dst_folder = f"{data_path}/round1_traning_data_ee/CMeEE"
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    predict(
        f"{data_path}/round2_A榜_data/A榜.json",
        fn_dst="./temp/NE_A_6.txt",
        # i_from=1,
        # i_to=12,
    )
    # predict(
    #     f"{data_path}/round1_traning_data/train.json",
    #     fn_dst="./temp/NE_A_2.txt",
    #     i_to=3,
    # )