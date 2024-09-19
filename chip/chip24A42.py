import json
import logging
import os
import platform
import re
import sys
import traceback

sys.path.append(".")
from chip import data_loader
from conf import LLM_CONF, data_path
from llm import llm_wrapper
from llm import prompt_template_42 as prompt_template
from utils import dict_utils, ee_wrapper, file_utils, json_utils

logger = logging.getLogger(__name__)

test_only = False

exp_settings = {
    "1": [
        "openai",
        "qwen",
        # "doubao",
        # "qianfan",
    ],
    "2": [
        "openai",
        "qwen",
        # "doubao",
        # "qianfan",
    ],
    "3": [
        "openai",
        "qwen",
        # "doubao",
        # "qianfan",
    ],
    "4": "openai",
}

map23 = data_loader.load_map23()
map12 = data_loader.load_map12()


separators = "[，‌。‌：‌；、‌:;]"


def seg_sentence(raw_text):
    s = (
        raw_text.replace("自诉", "")
        .replace("诊其", "")
        .replace("诊时", "")
        .replace("症见", "")
    )
    raw_split = re.split(separators, s)
    result = set()
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


def predict_1(raw_text, entity_list):
    result1 = seg_sentence(raw_text)

    prompt1 = prompt_template.template_prompt1.replace("{raw_text}", raw_text).replace(
        "{sugguest1}",
        json.dumps(list(result1), ensure_ascii=False),
    )
    print(f"prompt1={prompt1}")

    for llm_model in exp_settings["1"]:
        response1 = llm_wrapper.chat_complete(prompt1, llm_name=llm_model)
        response1_obj = json_utils.cvt_str_to_obj(response1)
        print(f"{llm_model}={response1}")

        for item in response1_obj.get("临床表现信息", []):
            # print(item)
            if item:
                result1 = result1.union(set(re.split(separators, item)))
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

    sugguest2 = set()
    for name1 in result1:
        if name1 in map12:
            for t1 in map12[name1]:
                if t1 in candidate2_map_r:
                    sugguest2.add(t1)

    example2 = {"B": 1}  # , "B": 2
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
    for llm_model in exp_settings["2"]:
        response2_tmp = llm_wrapper.chat_complete(prompt2, llm_name=llm_model)
        response2_tmp_obj = json_utils.cvt_str_to_obj(response2_tmp)
        print(f"{llm_model}={response2_tmp_obj}")

        for k in response2_tmp_obj:
            if response2_tmp_obj[k] < 1:
                continue
            if response2_tmp_obj[k] > 3:
                response2_tmp_obj[k] = 3

            if k not in response2_obj:
                response2_obj[k] = 0
            response2_obj[k] += response2_tmp_obj[k]
    print(f"merged={response2_obj}")

    result2_scored = sorted(response2_obj.items(), key=lambda item: -item[1])

    result2_full = [
        (r2[0], r2[1], candidate2_map[r2[0]]) for r2 in result2_scored if r2[1] >= 3
    ]
    return result2_full


def predict_3(raw_text, result2_full, candidate3):
    result2_name = [r2[2] for r2 in result2_full]
    candidate3_map, candidate3_map_r = data_loader.parse_candidate(candidate3)

    # 根据训练集合得到的结果，但是未必在候选项里面
    sugguest3 = set()
    for name2 in result2_name:
        if name2 in map23:
            for t3 in map23[name2]:
                if t3 in candidate3_map_r:
                    sugguest3.add(t3)

    example3 = {"B": 1}
    prompt3 = (
        prompt_template.template_prompt3.replace("{raw_text}", raw_text)
        .replace("{example3}", json.dumps(example3, ensure_ascii=False))
        .replace("{result2}", ";".join(result2_name))
        .replace("{candidate3}", candidate3)
    )
    if not test_only:
        prompt3 = prompt3.replace(
            "{sugguest3}", json.dumps(list(sugguest3), ensure_ascii=False)
        )
    print(f"promp3={prompt3}")

    response3_obj = {}
    for llm_model in exp_settings["3"]:
        response3_gpt = llm_wrapper.chat_complete(prompt3, llm_name=llm_model)
        response3_gpt_obj = json_utils.cvt_str_to_obj(response3_gpt)
        print(f"{llm_model}={response3_gpt_obj}")

        for k in response3_gpt_obj:
            if response3_gpt_obj[k] < 1:
                continue
            if response3_gpt_obj[k] > 3:
                response3_gpt_obj[k] = 3
            if k not in response3_obj:
                response3_obj[k] = 0
            response3_obj[k] += response3_gpt_obj[k]

    result3_scored = sorted(response3_obj.items(), key=lambda item: -item[1])

    result3_full = [
        (r3[0], r3[1], candidate3_map[r3[0]])
        for r3 in result3_scored
        if r3[0] in candidate3_map and r3[1] >= 3
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
    result4 = "临证体会：%s辨证：%s" % (
        response4_obj.get("临证体会", ""),
        response4_obj.get("辨证", ""),
    )
    return result4


# TODO 换模型，逗号切分，保证只推一个，临证体会更多提示
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
            # return

            # 2 病机 [(A,2,肾阴虚)]
            result2_full = predict_2(
                raw_text=raw_text,
                result1=result1,
                candidate2=r.get("病机选项"),
            )
            print(f"result2_full={result2_full}")
            result2_filtered = data_loader.filter(
                result2_full, max_number=5, filter_ratio=0.3
            )
            print(f"result2_filtered={result2_filtered}")

            # 3 证候 [(A,2,肾阴虚)]
            result3_full = predict_3(
                raw_text,
                result2_full=result2_filtered,  # result2_full result2_filtered
                candidate3=r.get("证候选项"),
            )
            print(f"result3_full={result3_full}")
            result3_filtered = data_loader.filter(
                result3_full, max_number=5, filter_ratio=0.3
            )
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

    # test_only = True
    # predict(
    #     f"{data_path}/round2_A榜_data/A榜.json",
    #     # fn_dst="./temp/NE_A_42s.txt",
    #     i_from=7,
    #     i_to=8,
    # )

    test_only = True
    predict(
        f"{data_path}/round1_traning_data/train.json",
        fn_dst="./temp/NE_train_42s.txt",
        i_from=195,
        i_to=196,
    )
