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
from llm import prompt_template_44 as prompt_template
from llm.task_1 import predict_1
from llm.task_2 import predict_2_a
from llm.task_3 import predict_3
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
        # "openai",
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

map12 = data_loader.load_map12()
map23 = data_loader.load_map23()


# TODO 换模型，逗号切分，保证只推一个，临证体会更多提示
def predict(fn, baseline="", fn_dst_prefix=None, i_from=0, i_to=sys.maxsize):
    fn_dst = f"{fn_dst_prefix}_{i_from}_{i_to}.json"

    # 加载原始数据
    dataset_test = []
    data_test = data_loader.load(fn)
    for i, r in enumerate(data_test):
        if i < i_from:
            continue
        if i >= i_to:
            break
        dataset_test.append(r)
    print(f"dataset={len(data_test)}")

    # 将baseline更新到dataset_test，只有[i_from, i_to)

    if baseline:
        if baseline.find("json") >= 0:
            data_baseline = data_loader.load(baseline)
            for i, r in enumerate(dataset_test):
                r.update(data_baseline[i])
        else:
            with open(baseline, "r", encoding="utf8") as fp_baseline:
                data_baseline = fp_baseline.readlines()
                for i, r in enumerate(dataset_test):
                    segs = data_baseline[i].split("@")
                    print(f'append gt {r.get("案例编号")} {segs[0]}')
                    for j in range(1, 5):
                        r[f"predict_{j}"] = segs[j].strip()
                fp_baseline.close()

    # 实体模型预测
    text_list = [r.get("临床资料") for r in data_test]
    entities_list = ee_wrapper.do_predict(text_list)

    result_lines = []
    for i, r in enumerate(dataset_test):
        #  debug
        # if i < 9 or i > 9:
        #     continue

        try:
            name = r.get("案例编号", None)
            raw_text = r.get("临床资料", None)
            print(f"{i}={name}")

            # 一个字的不要，仅作参考（包括但不限于）
            entity_list = [
                en["entity"] for en in entities_list[i] if len(en["entity"]) > 1
            ]

            # 1
            if "1" in exp_settings and exp_settings["1"]:
                result1_full, result1_weighted = predict_1(
                    raw_text,
                    model_list=exp_settings["1"],
                    entity_list=entity_list if entity_list else ["大便干"],
                )
                r["predict_1_full"] = list(result1_full)
                r["preidct_1_wighted"] = result1_weighted
                r["predict_1"] = ";".join([r for r in result1_full])

            # 2 病机 [(A,2,肾阴虚)]
            # 上一步数据太多，需要weight
            # TODO 反面权重：用历史诊断，降低相关病机权重
            if "2" in exp_settings and exp_settings["2"]:
                result2_full = predict_2_a(
                    raw_text=raw_text,
                    candidate2=r.get("病机选项"),
                    result1_weighted=r["preidct_1_wighted"],
                    model_list=exp_settings["2"],
                )
                # 某一个值太高，导致后面大部分被过滤
                result2_filtered = data_loader.filter(
                    result2_full, max_number=5, filter_ratio=0.5
                )
                print(f"result2_filtered={result2_filtered}")
                r["predict_2"] = ";".join(sorted([r2[0] for r2 in result2_filtered]))
                r["predict_2_full"] = result2_full
                r["predict_2_filtered"] = result2_filtered

            # # 3 证候 [(A,2,肾阴虚)]
            if "3" in exp_settings and exp_settings["3"]:
                result3_full = predict_3(
                    raw_text,
                    result2_full=r["predict_2_full"],  # result2_full result2_filtered
                    candidate3=r.get("证候选项"),
                    model_list=exp_settings["3"],
                )
                print(f"result3_full={result3_full}")
                # 控制数量
                # 低限定条件，为了召回
                result3_filtered = data_loader.filter(
                    result3_full, max_number=5, filter_ratio=0.3
                )
                # 如果4or5个，提高限定条件
                if len(result3_filtered) >= 4:
                    result3_filtered = data_loader.filter(
                        result3_full, max_number=5, filter_ratio=0.4
                    )
                print(f"result3_filtered={result3_filtered}")
                r["predict_3"] = ";".join(sorted([r2[0] for r2 in result3_filtered]))
                r["predict_3_full"] = result3_full
                r["predict_3_filtered"] = result3_filtered

            # 临证体会
            if "4" in exp_settings and exp_settings["4"]:
                result4 = predict_4(
                    raw_text,
                    result1=[k for k in result1_weighted],
                    result2_full=r["predict_2_full"],
                    result3_full=r["predict_3_full"],
                    model_name=exp_settings["4"],
                )
                r["predict_4"] = result4

        except Exception as e:
            print(f"ERROR name={name} i={i}")
            logger.error(e)
            traceback.print_exc()

        # 每完成一个任务，都保存完整结果数据
        with open(fn_dst, "w", encoding="utf8") as fp_dst:
            json.dump(dataset_test, fp=fp_dst, ensure_ascii=False, indent=4)
            fp_dst.close()


if __name__ == "__main__":
    test_only = False
    predict(
        f"{data_path}/round3_B榜_data.zip/B榜.json",
        # baseline=f"./temp/NE_B_44.json",
        fn_dst_prefix="./temp/NE_B_44",
        i_from=27,
        i_to=28,
    )

    # predict(
    #     f"{data_path}/round2_A榜_data/A榜.json",
    #     baseline=f"./temp/NE_fuse_41_43_14.txt",
    #     fn_dst_prefix="./temp/NE_A_44",
    #     # i_from=44,
    #     # i_to=44,
    # )

    # predict(
    #     # src
    #     f"{data_path}/round1_traning_data/train.json",
    #     i_from=180,
    #     i_to=190,
    #     # baseline
    #     baseline=f"./temp/NE_train_43.txt",
    #     # baseline=f"./temp/NE_train_44s1_180_190.json",
    #     # dst
    #     fn_dst_prefix="./temp/NE_train_44s2",
    # )
