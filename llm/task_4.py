import json
import os
import re
import sys

sys.path.append(".")
from llm import llm_wrapper
from llm import prompt_template_44 as prompt_template
from utils import dict_utils, file_utils, json_utils


def predict_4(raw_text, result1, result2_full, result3_full, model_name="openai"):
    prompt4 = (
        prompt_template.template_prompt4.replace("{raw_text}", raw_text)
        .replace(
            "{example4}",
            json.dumps(prompt_template.template_example4, ensure_ascii=False),
        )
        .replace("{result1}", ";".join(result1))
        .replace("{result2}", ";".join([r2[2] for r2 in result2_full]))
        .replace("{result3}", ";".join([r3[2] for r3 in result3_full]))
    )
    # 23字母序

    print(f"prompt4={prompt4}")
    response4 = llm_wrapper.chat_complete(prompt4, model_name)
    print(f"response4={response4}")

    response4_obj = json_utils.cvt_str_to_obj(response4)
    result4 = "临证体会：%s辨证：%s" % (
        response4_obj.get("临证体会", ""),
        response4_obj.get("辨证", ""),
    )
    return result4


if __name__ == "__main__":
    predict_4(
        raw_text="黄某，女，32岁。主诉及病史：患过敏性鼻炎已数年，累治未愈。此次发作特剧，头闷疼，鼻塞而痒，喷嚏连连，清涕常流，殊觉不适。诊查：苔 薄白，舌质略紫暗，脉弦，便干尿黄。",
        result1=[
            "鼻塞而痒",
            "清涕常流",
            "此次发作特剧",
            "喷嚏连连",
            "累治未愈",
            "便干尿黄",
            "患过敏性鼻炎已数年",
            "苔薄白",
            "舌质略紫暗",
            "脉弦",
            "殊觉不适",
            "头闷疼",
        ],
        result2_full=[("A", 27, "风邪伏肺"), ("F", 3, "内热"), ("G", 3, "久郁致瘀")],
        result3_full=[
            ("D", 6, "伏风夹瘀"),
            ("F", 4, "瘀热互结"),
            ("A", 2, "心肺脾(胃)积热"),
        ],
    )
