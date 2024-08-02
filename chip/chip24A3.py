import json
import logging
import os
import platform
import sys
import traceback

sys.path.append(".")
from chip import data_loader
from utils import ee_wrapper, file_utils, json_utils, qwen_wrapper

logger = logging.getLogger(__name__)

data_path = (
    "D:/fdata/ai/med/CHIP2004"
    if platform.system() == "Windows"
    else "/mnt/windows/sting/data/CHIP2004"
)

# 1 临床表现信息（实体）
template_prompt1 = """您是一位中医学专家，请根据下面这段文字提取主要的临床表现信息（包括症状描述，不包括体温等指标型数据），用json格式返回。
注意严格保留原文。
病情描述：{raw_text}
严格按照输出json格式：{"临床表现信息": {sugguest1}}
"""

# 2 病机
template_prompt2 = """您是一位中医学专家，请针对每条临床表现信息，结合病情描述，从病机候选项中选择最相关的一条记录标签，并根据相关性和重要性打分，分值范围0、1、2或者3，分值越高重要性越高，严格按照json格式输出。
临床表现信息：{result1}
病情描述：{raw_text}
病机候选项：{candidate2}
严格按照输出json格式：{example2}
"""

# 3 证候
template_prompt3 = """您是一位中医学专家，请针对每条病机信息，结合病情描述，从证候候选项中选择最相关的一条记录标签，并根据相关性和重要性打分，分值范围0、1、2或者3，分值越高重要性越高，严格按照json格式输出(```json)。
病机信息：{result2}
病情描述：{raw_text}
证候候选项：{candidate3}
已知证候推理结果：{sugguest3}
输出json格式：{example3}
"""

# 4 临证体会
template_prompt4 = """您是一位中医学专家，请根据提供的病情描述、临床表现信息、病机信息、证候信息，总结临证体会和辨证，不要复述症状和病机，不要给出治疗方案，用json格式输出。
病情描述：{raw_text}
病机信息：{result2}
证候信息：{result3}
输出json格式：{example4}
"""
template_example4 = {
    "临证体会": "本例因情志不舒,思虑过度,劳伤心脾,导致心阴亏损、气血亏耗,以致神不守舍,胆虚不眠。",
    "辨证": "心胆亏虚",
}


# 找到实体出现位置，sym类型
def cvt(fn, fn_dst, i_type="train", i_from=0, i_to=sys.maxsize):
    result = []

    data_train = data_loader.load(fn)
    for i, r in enumerate(data_train):
        name = r.get("案例编号", None)
        print(f"{i}={name}")
        if i < i_from:
            continue
        if i >= i_to:
            break

        try:
            raw_text = r.get("临床资料", None)
            if not raw_text:
                logger.error(f"no 临床资料 {i}")
                return
            # print(raw_text)

            s = r.get("信息抽取能力-核心临床信息", "")
            # print(s)

            entities = []
            if i_type == "train":
                for seg in s.split(";"):
                    posi = raw_text.find(seg)
                    entities.append(
                        {
                            "start_idx": posi,
                            "end_idx": posi + len(seg),
                            "type": "sym",
                            "entity": seg,
                        }
                    )

            result.append({"text": raw_text, "entities": entities})
        except Exception as e:
            print(f"ERROR name={name} i={i}")
            logger.error(e)
    print(f"len={len(result)}")
    with open(fn_dst, "w", encoding="utf8") as fp:
        json.dump(result, fp=fp, indent=2, ensure_ascii=False)
        fp.close()


def predict(fn, fn_dst=None, i_from=0, i_to=sys.maxsize):
    result_lines = []
    map23 = data_loader.load_map23()

    data_test = data_loader.load(fn)
    text_list = [r.get("临床资料") for r in data_test]
    entities_list = ee_wrapper.do_predict(text_list)

    #  一个字的不要，仅作参考（包括但不限于）
    for i, r in enumerate(data_test):
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
            prompt1 = template_prompt1.replace("{raw_text}", raw_text).replace(
                "{sugguest1}",
                json.dumps(
                    entity_list if entities_list else ["大便干"], ensure_ascii=False
                ),
            )
            print(f"prompt1={prompt1}")
            response1 = qwen_wrapper.chat_complete(prompt1)
            # print(f"response={response1}")
            response1_obj = json_utils.cvt_str_to_obj(response1)
            result1 = response1_obj.get("临床表现信息", [])
            print(f"result1={result1}")

            # 病机
            candidate2 = r.get("病机选项")
            candidate2_map = data_loader.parse_candidate(candidate2)
            example2 = {v1: ["A", 0] for v1 in result1}
            prompt2 = (
                template_prompt2.replace("{raw_text}", raw_text)
                .replace("{result1}", ";".join(result1))
                .replace("{candidate2}", candidate2)
                .replace("{example2}", json.dumps(example2, ensure_ascii=False))
            )
            print(f"promp2={prompt2}")
            response2 = qwen_wrapper.chat_complete(prompt2)
            response2_obj = json_utils.cvt_str_to_obj(response2)
            result2_scored = data_loader.rank(response2_obj)
            result2_full = [
                (r2[0], r2[1], candidate2_map[r2[0]]) for r2 in result2_scored
            ]
            print(result2_full)
            result2 = [r2[0] for r2 in result2_full]
            result2_name = [r2[2] for r2 in result2_full]

            # 证候
            candidate3 = r.get("证候选项")
            candidate3_map = data_loader.parse_candidate(candidate3)
            sugguest3 = [map23[name2] for name2 in result2_name if name2 in map23]
            example3 = {v2: ["B", 0] for v2 in result2_name}
            prompt3 = (
                template_prompt3.replace("{raw_text}", raw_text)
                .replace("{example3}", json.dumps(example3, ensure_ascii=False))
                .replace("{result2}", ";".join(result2_name))
                .replace("{candidate3}", candidate3)
                .replace("{sugguest3}", json.dumps(sugguest3, ensure_ascii=False))
            )
            print(f"promp3={prompt3}")
            response3 = qwen_wrapper.chat_complete(prompt3)
            response3_obj = json_utils.cvt_str_to_obj(response3)
            result3_scored = data_loader.rank(response3_obj)
            result3_full = [
                (r3[0], r3[1], candidate3_map[r3[0]])
                for r3 in result3_scored
                if r3[0] in candidate3_map
            ]
            print(result3_full)
            result3 = [r3[0] for r3 in result3_full]
            result3_name = [r3[2] for r3 in result3_full]

            # 临证体会
            prompt4 = (
                template_prompt4.replace("{raw_text}", raw_text)
                .replace(
                    "{example4}", json.dumps(template_example4, ensure_ascii=False)
                )
                .replace("{result2}", ";".join(result2_name))
                .replace("{result3}", ";".join(result3_name))
            )
            print(f"prompt4={prompt4}")
            response4 = qwen_wrapper.chat_complete(prompt4)
            print(f"response4={response4}")
            response4_obj = json_utils.cvt_str_to_obj(response4)
            result4 = "临证体会：%s。辨证：%s" % (
                response4_obj.get("临证体会", ""),
                response4_obj.get("辨证", ""),
            )

            #
            result_final = "%s@%s@%s@%s@%s" % (
                name,
                ";".join(result1),
                ";".join(result2),
                ";".join(result3),
                result4,
            )
            print(result_final)
            result_lines.append(result_final)

            # debug
            # break
        except Exception as e:
            print(f"ERROR name={name} i={i}")
            logger.error(e)
            traceback.print_exc()
            result_lines.append("")
    if fn_dst:
        with open(fn_dst, "w", encoding="utf8") as fpr:
            for line in result_lines:
                fpr.write(line)
            fpr.close()


if __name__ == "__main__":
    dst_folder = f"{data_path}/round1_traning_data_ee/CMeEE"
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    # cvt(
    #     fn=f"{data_path}/round1_traning_data/train.json",
    #     fn_dst=f"{dst_folder}/CMeEE-V2_train.json",
    #     i_from=0,
    #     i_to=180,
    # )
    # cvt(
    #     fn=f"{data_path}/round1_traning_data/train.json",
    #     fn_dst=f"{dst_folder}/CMeEE-V2_eval.json",
    #     i_from=180,
    # )
    # cvt(
    #     fn=f"{data_path}/round2_A榜_data/A榜.json",
    #     fn_dst=f"{dst_folder}/CMeEE-V2_test.json",
    #     i_type="test",
    # )

    predict(
        f"{data_path}/round2_A榜_data/A榜.json",
        fn_dst="./temp/NE_A_3.txt",
        i_from=17,
        i_to=18,
    )
    # predict(
    #     f"{data_path}/round1_traning_data/train.json",
    #     fn_dst="./temp/NE_A_2.txt",
    #     i_to=3,
    # )
