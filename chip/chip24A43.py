import json
import logging
import os
import platform
import sys
import traceback

sys.path.append(".")
from chip import data_loader
from conf import LLM_CONF, data_path
from llm import llm_wrapper
from llm import prompt_template_43 as prompt_template
from llm.task_1 import predict_1
from llm.task_2 import predict_2
from llm.task_4 import predict_4
from utils import dict_utils, ee_wrapper, file_utils, json_utils

logger = logging.getLogger(__name__)

test_only = False

exp_settings = {
    "1": [
        "openai",
        # "qwen",
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

    example3 = {v2: {"B": 3} for v2 in result2_name}
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

        response3_gpt_obj2 = dict_utils.revise_dict3(
            response3_gpt_obj, candidate3_map, candidate3_map_r
        )
        response3_obj = dict_utils.merge_dict_3(
            response3_obj, response3_gpt_obj2, min_value=2
        )

    result3_scored = data_loader.rank3(response3_obj)
    result3_full = [
        (r3[0], r3[1], candidate3_map[r3[0]])
        for r3 in result3_scored
        if r3[0] in candidate3_map
    ]
    return result3_full


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
            result1_full, result1_weighted = predict_1(
                raw_text,
                model_list=exp_settings["1"],
                entity_list=entity_list if entity_list else ["大便干"],
            )

            # 2 病机 [(A,2,肾阴虚)]
            # 上一步数据太多，很多重要性不高，需要weight
            result2_full = predict_2(
                raw_text=raw_text,
                result1_weighted=result1_weighted,
                candidate2=r.get("病机选项"),
                model_list=exp_settings["2"],
            )
            # 某一个值太高，导致后面大部分被过滤
            result2_filtered = data_loader.filter(
                result2_full, max_number=5, filter_ratio=0.2
            )
            if len(result2_filtered) >= 5:
                result2_filtered = data_loader.filter(
                    result2_full, max_number=5, filter_ratio=0.25
                )
            print(f"result2_filtered={result2_filtered}")

            # 3 证候 [(A,2,肾阴虚)]
            result3_full = predict_3(
                raw_text,
                result2_full=result2_full,  # result2_full result2_filtered
                candidate3=r.get("证候选项"),
            )
            print(f"result3_full={result3_full}")

            result3_filtered = data_loader.filter(
                result3_full, max_number=5, filter_ratio=0.3
            )
            # 控制数量
            if len(result3_filtered) >= 4:
                result3_filtered = data_loader.filter(
                    result3_full, max_number=5, filter_ratio=0.4
                )
            print(f"result3_filtered={result3_filtered}")

            # 临证体会
            result4 = predict_4(
                raw_text,
                result1=[k for k in result1_weighted],
                result2_full=result2_full,
                result3_full=result3_full,
                model_name=exp_settings["4"],
            )

            # 23用精简后的结果
            result_final = "%s@%s@%s@%s@%s" % (
                name,
                ";".join(result1_full),
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

    # test_only = False
    # predict(
    #     f"{data_path}/round2_A榜_data/A榜.json",
    #     fn_dst="./temp/NE_A_43.txt",
    #     i_from=43,
    #     i_to=44,
    # )

    test_only = False
    predict(
        f"{data_path}/round3_B榜_data.zip/B榜.json",
        fn_dst="./temp/NE_B_43.txt",
        # i_from=0,
        # i_to=1,
    )

    # predict(
    #     f"{data_path}/round1_traning_data/train.json",
    #     fn_dst="./temp/NE_train_43.txt",
    #     i_from=180,
    #     i_to=190,
    # )
