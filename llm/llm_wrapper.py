import json
import logging
import os
import re
import sys
import traceback
from http import HTTPStatus

import dashscope
import qianfan
from openai import OpenAI

sys.path.append(".")
from conf import LLM_CONF
from llm import prompt_template

logger = logging.getLogger(__name__)
dashscope.api_key = LLM_CONF["qwen"]["API_KEY"]

os.environ["QIANFAN_ACCESS_KEY"] = LLM_CONF["qianfan"]["ACCESS_KEY"]
os.environ["QIANFAN_SECRET_KEY"] = LLM_CONF["qianfan"]["SECRET_KEY"]


class Message:
    def __init__(self, content):
        self.content = content


def chat_complete(q: str, llm_name="openai"):
    output_msg = ""
    input_msgs = [
        {
            "role": "system",
            "content": "你是一个中医医学专家，根据提供的内容回答问题",
        },
        {"role": "user", "content": q},
    ]

    if llm_name == "qwen":
        response = dashscope.Generation.call(
            LLM_CONF[llm_name]["MODEL_NAME"],
            messages=input_msgs,
            result_format="message",  # set the result to be "message"  format.
            stream=False,  # set streaming output
            incremental_output=False,  # get streaming output incrementally
        )
        if response.status_code == HTTPStatus.OK:
            output_msg = response.output.choices[0]["message"]
    elif llm_name == "qianfan":
        chat_comp = qianfan.ChatCompletion()
        resp = chat_comp.do(
            model=LLM_CONF[llm_name]["MODEL_NAME"],
            messages=[input_msgs[1]],
        )
        print(resp["body"])
        output_msg = Message(resp["body"]["result"])
    else:  # doubao openai
        client = OpenAI(
            base_url=LLM_CONF[llm_name]["API_BASE"],
            api_key=LLM_CONF[llm_name]["API_KEY"],
        )
        completion = client.chat.completions.create(
            model=LLM_CONF[llm_name]["MODEL_NAME"],
            messages=input_msgs,
        )
        output_msg = completion.choices[0].message

    logger.info(f"{llm_name}模型返回结果:^{output_msg}$")
    segs = output_msg.content.split("output:")
    return segs[-1].strip()


if __name__ == "__main__":
    msg = chat_complete(
        llm_name="qianfan",
        q=prompt_template.template_prompt4.replace(
            "{raw_text}",
            "病情描述：某女，62岁。初诊：1957年1月。主诉及病史：发病十数天，咳逆不能平卧，唾白色泡沫痰。诊查：短气，语音低微，神识昏愦不清，时妄 言语，终又复言，身有微热，手足厥冷，偶饮热一二口。脉浮细数而无力。\n",
        )
        .replace(
            "{example4}",
            json.dumps(prompt_template.template_example4, ensure_ascii=False),
        )
        .replace("{result2}", ";".join(["心气虚", "痰浊", "气阴两亏"]))
        .replace("{result3}", ";".join(["气虚血瘀", "阴亏之体"])),
    )
    print(msg)
