import json
import logging
import os
import re
import sys
import traceback

from openai import OpenAI

from conf import LLM_CONF

logger = logging.getLogger(__name__)


def chat_complete(q: str, openai_conf=LLM_CONF["openai"]):
    client = OpenAI(
        base_url=openai_conf["API_BASE"],
        api_key=openai_conf["API_KEY"],
    )
    completion = client.chat.completions.create(
        model=openai_conf["MODEL_NAME"],
        messages=[
            {
                "role": "system",
                "content": "你是一个中医医学专家，根据提供的内容回答问题",
            },
            {"role": "user", "content": q},
        ],
    )
    logger.info(f"模型返回结果:^{completion.choices[0].message}$")
    segs = completion.choices[0].message.content.split("output:")
    return segs[-1].strip()


#
# ```json
# {
#   "送检时间（日期)": "2022-10-11",
#   "原始血细胞": "N/A",
#   "血片:原始粒细胞": "N/A",
#   "血片:嗜碱性粒细胞(嗜碱性中幼+嗜碱性晚幼+嗜碱性杆状核+嗜碱性分叶核四值总和)": "N/A"
# }
# ```
def cvt_str_to_obj(s):
    try:
        json_str = s

        posi_start = s.find("```json")
        if posi_start >= 0:
            posi_end = s.find("```", posi_start + 7)
            if posi_end >= 0:
                json_str = s[posi_start + 7 : posi_end]
        obj = json.loads(json_str)
        return obj
    except Exception as e:
        logger.error(f"sth wrong")
        traceback.print_exc()
    return None
