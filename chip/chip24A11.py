import json
import logging
import os
import platform
import sys
import traceback

sys.path.append(".")
from chip import data_loader
from conf import LLM_CONF, data_path
from llm import llm_wrapper, prompt_template
from utils import dict_utils, ee_wrapper, file_utils, json_utils

logger = logging.getLogger(__name__)

exp_settings = {
    "1": "openai",
    "2": [
        "openai",
        "qwen",
        "doubao",
        "qianfan",
    ],
    "3": [
        "openai",
        "qwen",
        "doubao",
        "qianfan",
    ],
    "4": "openai",
}

map23 = data_loader.load_map23()


def predict_1(raw_text, entity_list):
    prompt1 = prompt_template.template_prompt1.replace("{raw_text}", raw_text).replace(
        "{sugguest1}",
        json.dumps(entity_list, ensure_ascii=False),
    )
    print(f"prompt1={prompt1}")
    response1 = llm_wrapper.chat_complete(prompt1, exp_settings["1"])
    # print(f"response={response1}")
    response1_obj = json_utils.cvt_str_to_obj(response1)
    result1 = response1_obj.get("临床表现信息", [])
    print(f"result1={result1}")
    return result1


def revise_dict(response_obj, cmap, cmap_r):
    response_obj2 = {}
    for k1 in response_obj:
        if len(response_obj[k1]) < 2:
            continue

        k2 = response_obj[k1][0]
        # qianfan format bug
        if isinstance(k2, list):
            k2 = k2[0]

        if k2 not in cmap and k2 in cmap_r:
            k2 = cmap_r[k2]

        response_obj2[k1] = [k2, response_obj[k1][1]]
    return response_obj2


def revise_dict3(response_obj, cmap, cmap_r):
    response_obj2 = {}
    for k1 in response_obj:
        for k2 in response_obj[k1]:
            if response_obj[k1][k2] > 0:
                if k1 not in response_obj2:
                    response_obj2[k1] = {}

                real_k2 = k2
                # qianfan format bug
                if isinstance(k2, list):
                    real_k2 = k2[0]
                if real_k2 not in cmap and real_k2 in cmap_r:
                    real_k2 = cmap_r[real_k2]

                response_obj2[k1][real_k2] = response_obj[k1][k2]
    return response_obj2


def predict_2(raw_text, result1, candidate2):
    candidate2_map, candidate2_map_r = data_loader.parse_candidate(candidate2)

    example2 = {v1: ["A", 0] for v1 in result1}
    prompt2 = (
        prompt_template.template_prompt2.replace("{raw_text}", raw_text)
        .replace("{result1}", ";".join(result1))
        .replace("{candidate2}", candidate2)
        .replace("{example2}", json.dumps(example2, ensure_ascii=False))
    )
    print(f"promp2={prompt2}")

    response2_obj = dict()
    for llm_model in exp_settings["2"]:
        response2_tmp = llm_wrapper.chat_complete(prompt2, llm_name=llm_model)
        response2_tmp_obj = json_utils.cvt_str_to_obj(response2_tmp)
        response2_tmp_obj2 = revise_dict(
            response2_tmp_obj, candidate2_map, candidate2_map_r
        )
        print(f"{llm_model}={response2_tmp_obj2}")
        response2_obj = dict_utils.merge_dict_2(response2_obj, response2_tmp_obj2)
    print(f"merged={response2_obj}")

    result2_scored = data_loader.rank(response2_obj)

    result2_full = [(r2[0], r2[1], candidate2_map[r2[0]]) for r2 in result2_scored]
    return result2_full


def predict_3(raw_text, result2_full, candidate3):
    result2_name = [r2[2] for r2 in result2_full]
    candidate3_map, candidate3_map_r = data_loader.parse_candidate(candidate3)

    # 根据训练集合得到的结果，但是未必在候选项里面
    sugguest3 = []
    for name2 in result2_name:
        if name2 in map23 and name2 in candidate3_map_r:
            sugguest3 += map23[name2]

    example3 = {v2: {"A": 0, "B": 1} for v2 in result2_name}
    prompt3 = (
        prompt_template.template_prompt3.replace("{raw_text}", raw_text)
        .replace("{example3}", json.dumps(example3, ensure_ascii=False))
        .replace("{result2}", ";".join(result2_name))
        .replace("{candidate3}", candidate3)
        .replace("{sugguest3}", json.dumps(sugguest3, ensure_ascii=False))
    )
    print(f"promp3={prompt3}")

    response3_obj = {}
    for llm_model in exp_settings["3"]:
        response3_gpt = llm_wrapper.chat_complete(prompt3, llm_name=llm_model)
        response3_gpt_obj = json_utils.cvt_str_to_obj(response3_gpt)
        response3_gpt_obj2 = revise_dict3(
            response3_gpt_obj, candidate3_map, candidate3_map_r
        )
        print(f"{llm_model}={response3_gpt_obj2}")
        response3_obj = dict_utils.merge_dict_3(response3_obj, response3_gpt_obj2)

    result3_scored = data_loader.rank3(response3_obj)
    result3_full = [
        (r3[0], r3[1], candidate3_map[r3[0]])
        for r3 in result3_scored
        if r3[0] in candidate3_map
    ]
    return result3_full


def predict_4(raw_text, result2_full, result3_full):
    prompt4 = (
        prompt_template.template_prompt4.replace("{raw_text}", raw_text)
        .replace(
            "{example4}",
            json.dumps(prompt_template.template_example4, ensure_ascii=False),
        )
        .replace("{result2}", ";".join([r2[2] for r2 in result2_full]))
        .replace("{result3}", ";".join([r3[2] for r3 in result3_full]))
    )
    print(f"prompt4={prompt4}")
    response4 = llm_wrapper.chat_complete(prompt4, exp_settings["4"])
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
            print(f"result2_full={result2_full}")
            result2_filtered = data_loader.filter(result2_full)
            print(f"result2_filtered={result2_filtered}")

            # 3 证候 [(A,2,肾阴虚)]
            result3_full = predict_3(
                raw_text,
                result2_full=result2_full,
                candidate3=r.get("证候选项"),
            )
            print(f"result3_full={result3_full}")
            result3_filtered = data_loader.filter(result3_full)
            print(f"result3_filtered={result3_filtered}")

            # 临证体会
            result4 = predict_4(
                raw_text,
                result2_full=result2_full,
                result3_full=result3_full,
            )

            # 23用精简后的结果
            result_final = "%s@%s@%s@%s@%s" % (
                name,
                ";".join(result1),
                ";".join([r2[0] for r2 in result2_filtered]),
                ";".join([r3[0] for r3 in result3_filtered]),
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
        fn_dst="./temp/NE_A_11.txt",
        # i_from=2,
        # i_to=3,
    )
    # predict(
    #     f"{data_path}/round1_traning_data/train.json",
    #     fn_dst="./temp/NE_A_2.txt",
    #     i_from=199,
    #     i_to=181,
    # )
