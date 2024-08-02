import json
import logging
import os
import platform
import sys

sys.path.append(".")
from chip.data_loader import get_top, load, parse_candidate
from utils import file_utils, openai_wrapper

logger = logging.getLogger(__name__)

data_path = (
    "D:/fdata/ai/med/CHIP2004"
    if platform.system() == "Windows"
    else "/mnt/windows/sting/data/"
)

# 临床表现信息（实体）
template_prompt1 = """您是一位中医学专家，请根据下面这段文字提取主要的临床表现信息（实体），用json格式返回。
注意严格保留原文。
病情描述：{raw_text}
输出json格式：{example1}
"""
template_example1 = {"临床表现信息": ["鼻流血", "胸闷气逆"]}

# 病机
template_prompt2 = """您是一位中医学专家，请根据提供的临床表现信息，参考病情描述，从病机候选项中选择重要选项并根据相关性给出0到3的打分，按照得分从高到低排序，输出选项编号（类似ABC）和对应得分，严格按照json格式输出。
临床表现信息：{result1}
病情描述：{raw_text}
病机候选项：{candidate2}
输出json格式：{example2}
"""
template_example2 = [{"C": 2}, {"B": 1}]

# 证候
template_prompt3 = """您是一位中医学专家，请根据提供的病机信息，参考病情描述，从证候候选项中选择重要选项并根据相关性给出0到3的打分，按照得分从高到低排序，输出选项编号（类似ABC）和对应得分，严格按照json格式输出。
病机信息：{result2}
病情描述：{raw_text}
证候候选项：{candidate3}
输出json格式：{example3}
"""
template_example3 = [{"E": 3}, {"B": 1}]

# 临证体会
template_prompt4 = """您是一位中医学专家，请根据提供的病情描述、临床表现信息、病机信息、证候信息，总结临证体会，不要复述症状和病机，不要给出治疗方案，用json格式输出。
病情描述：{raw_text}
病机信息：{result2}
证候信息：{result3}
输出json格式：{example4}
"""
template_example4 = {
    "临证体会": "本例因情志不舒,思虑过度,劳伤心脾,导致心阴亏损、气血亏耗,以致神不守舍,胆虚不眠。辨证：心胆亏虚"
}


def run(fn_dst, name_list=None):
    result_lines = []

    data_train = load(f"{data_path}/round1_traning_data/train.json")
    data_A = load(f"{data_path}/round2_A榜_data/A榜.json")
    for i, r in enumerate(data_A):
        name = r.get("案例编号", None)
        print(f"{i}={name}")
        if name_list and name not in name_list:
            continue
        
        try:
            raw_text = r.get("临床资料", None)
            if not raw_text:
                logger.error(f"no 临床资料 {i}")
                return

            # 临床表现信息/实体
            prompt1 = template_prompt1.replace("{raw_text}", raw_text).replace(
                "{example1}", json.dumps(template_example1, ensure_ascii=False)
            )
            print(f"prompt={prompt1}")
            response1 = openai_wrapper.chat_complete(prompt1)
            print(f"response={response1}")
            response1_obj = openai_wrapper.cvt_str_to_obj(response1)
            # 临床表现信息，临床表现，临床表现信息
            result1 = response1_obj.get("临床表现信息", [])
            print(result1)

            # 病机
            candidate2 = r.get("病机选项")
            candidate2_map = parse_candidate(candidate2)
            prompt2 = (
                template_prompt2.replace("{raw_text}", raw_text)
                .replace(
                    "{example2}", json.dumps(template_example2, ensure_ascii=False)
                )
                .replace("{result1}", ";".join(result1))
                .replace("{candidate2}", candidate2)
            )
            print(f"prompt={prompt2}")
            response2 = openai_wrapper.chat_complete(prompt2)
            print(f"response={response2}")
            response2_obj = openai_wrapper.cvt_str_to_obj(response2)
            result2 = get_top(response2_obj)
            result2_name = [candidate2_map[r2] for r2 in result2]
            print(result2)

            # 证候
            candidate3 = r.get("证候选项")
            candidate3_map = parse_candidate(candidate3)
            prompt3 = (
                template_prompt3.replace("{raw_text}", raw_text)
                .replace(
                    "{example3}", json.dumps(template_example3, ensure_ascii=False)
                )
                .replace("{result2}", ";".join(result2_name))
                .replace("{candidate3}", candidate3)
            )
            print(f"prompt={prompt3}")
            response3 = openai_wrapper.chat_complete(prompt3)
            print(f"response={response3}")
            response3_obj = openai_wrapper.cvt_str_to_obj(response3)
            result3 = get_top(response3_obj)
            result3_name = [candidate3_map[r3] for r3 in result3]
            print(result3)

            # 临证体会
            prompt4 = (
                template_prompt4.replace("{raw_text}", raw_text)
                .replace(
                    "{example4}", json.dumps(template_example4, ensure_ascii=False)
                )
                .replace("{result2}", ";".join(result2_name))
                .replace("{result3}", ";".join(result3_name))
            )
            print(f"prompt={prompt4}")
            response4 = openai_wrapper.chat_complete(prompt4)
            print(f"response={response4}")
            response4_obj = openai_wrapper.cvt_str_to_obj(response4)
            result4 = response4_obj.get("临证体会", "")
            print(result4)

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
    with open(fn_dst, "w", encoding="utf8") as fpr:
        for line in result_lines:
            fpr.write(line)
        fpr.close()


if __name__ == "__main__":
    run(
        fn_dst="./temp/NE_A_1s.txt",
        name_list=[
            # "病例92", "病例261", "病例246", "病例38", "病例147"
        ],
    )
