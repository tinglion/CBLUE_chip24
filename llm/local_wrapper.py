import sys

import requests

URL_LOCAL_API = "http://192.168.19.211:8000/v1/model/chat"
request_template = {
    "model": "string",
    "messages": [
        {
            "role": "user",
            "content": '您是一位中医学专家，请根据提供的病情描述、临床表现信息、病机信息、证候信息，总结临证体会和辨证，用json格式输出。\n病情描述：初诊：1985年7月2日。主诉及病史：18岁月经初潮，每次经期小腹疼痛，恶冷，喜热敷，痛甚则呕，经潮2天后则痛渐缓，量少色暗红；每当经潮疼痛不能上班。平素腰酸耳鸣，倦怠无力，纳少便秘。月经周期为28天至40天，经期3天，末次月经6月26日。脉沉弦软(80次/分)，舌红苔灰。\n病机信息：["寒凝血瘀", "肾阳不足"]\n证候信息：["阳虚胞寒", "血虚血瘀"]\n输出json格式：["临证体会":"临证体会：中医认为是外感疫毒入里，耗损心气和心阴，故出现神委气短、胸闷心悸", "辨证":"外感时疫，热毒内侵"]',
        }
    ],
    "temperature": 0.3,
    "top_p": 0.7,
    "max_new_tokens": 1024,
    "repetition_penalty": 1,
    "length_penalty": 1.1,
    "top_k": 80,
    "n": 1,
    "stop": "string",
    "stream": False,
}


def chat_complete(q):
    data = request_template
    data["messages"][0]["content"] = q
    response = requests.post(URL_LOCAL_API, json=data)
    if response.status_code == 200:
        jdata = response.json()
        return jdata["choices"][0]["message"]["content"]


if __name__ == "__main__":
    chat_complete(
        '您是一位中医学专家，请根据提供的病情描述、临床表现信息、病机信息、证候信息，总结临证体会和辨证，用json格式输出。\n病情描述：黄某，女，32岁。主诉及病史：患过敏性鼻炎已数年，累治未愈。此次发作特剧，头闷疼，鼻塞而痒，喷嚏连连，清涕常流，殊觉不适。诊查：苔薄白，舌质略紫暗，脉弦，便干尿黄。\n临床表现信息：["鼻塞而痒", "喷嚏连连", "清涕常流", "苔薄白", "舌质略紫暗", "脉弦", "便干尿黄"]\n病机信息：["风邪伏肺", "久郁致瘀"]\n证候信息：["伏风夹瘀"]\n输出json格式：["临证体会":"临证体会：中医认为是外感疫毒入里，耗损心气和心阴，故出现神委气短、胸闷心悸", "辨证":"外感时疫，热毒内侵"]'
    )
