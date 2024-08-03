import os
import sys
from http import HTTPStatus

import dashscope

sys.path.append(".")
from conf import QWEN_CONF

os.environ["DASHSCOPE_API_KEY"] = QWEN_CONF["API_KEY"]
dashscope.api_key = QWEN_CONF["API_KEY"]


def chat_complete(q="中国的首都是哪里？"):
    messages = [{"role": "user", "content": q}]
    response = dashscope.Generation.call(
        "qwen-max-0428",
        messages=messages,
        result_format="message",  # set the result to be "message"  format.
        stream=False,  # set streaming output
        incremental_output=False,  # get streaming output incrementally
    )
    if response.status_code == HTTPStatus.OK:
        print(response.output.choices[0]["message"]["content"])
        return response.output.choices[0]["message"]["content"]
    else:
        print(
            "Request id: %s, Status code: %s, error code: %s, error message: %s"
            % (
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )
        )


if __name__ == "__main__":
    chat_complete(
        '您是一位中医学专家，请根据提供的病情描述、临床表现信息、病机信息、证候信息，总结临证体会和辨证，用json格式输出。\n病情描述：黄某，女，32岁。主诉及病史：患过敏性鼻炎已数年，累治未愈。此次发作特剧，头闷疼，鼻塞而痒，喷嚏连连，清涕常流，殊觉不适。诊查：苔薄白，舌质略紫暗，脉弦，便干尿黄。\n临床表现信息：["鼻塞而痒", "喷嚏连连", "清涕常流", "苔薄白", "舌质略紫暗", "脉弦", "便干尿黄"]\n病机信息：["风邪伏肺", "久郁致瘀"]\n证候信息：["伏风夹瘀"]\n输出json格式：["临证体会":"临证体会：中医认为是外感疫毒入里，耗损心气和心阴，故出现神委气短、胸闷心悸", "辨证":"外感时疫，热毒内侵"]'
    )
