import json
import os
import re
import sys

sys.path.append(".")
from chip import data_loader
from conf import test_only
from llm import llm_wrapper
from llm import prompt_template_44 as prompt_template
from utils import dict_utils, file_utils, json_utils

map23 = data_loader.load_map23()


def predict_3(raw_text, result2_full, candidate3, model_list=["openai"]):
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
    for llm_model in model_list:
        response3_gpt = llm_wrapper.chat_complete(prompt3, llm_name=llm_model)
        response3_gpt_obj = json_utils.cvt_str_to_obj(response3_gpt)
        print(f"{llm_model}={response3_gpt_obj}")

        response3_gpt_obj2 = dict_utils.revise_dict3(
            response3_gpt_obj, candidate3_map, candidate3_map_r
        )
        response3_obj = dict_utils.merge_dict_3(response3_obj, response3_gpt_obj2)

    result3_scored = data_loader.rank3(response3_obj)
    result3_full = [
        (r3[0], r3[1], candidate3_map[r3[0]])
        for r3 in result3_scored
        if r3[0] in candidate3_map
    ]
    return result3_full


if __name__ == "__main__":
    predict_3(
        raw_text="陈某，男，61岁。初诊：1977年6月2日。主诉及病史：右侧腰部疼痛已1周，5月28日患者因持续性右下腹痛、阵发性加剧，历时4小时而至某医院就诊，查血白细胞9600/mm3，中性79%，拟诊为阑尾炎，医用青霉素、颠茄合剂治之，痛未止。至5月29日因痛势加剧，又至某医院急诊，诊时右腰部绞痛，放射 至右腹部，摄腹部X线平片未见阳性结石影。尿常规检查：蛋白微量，上皮细胞极少，脓细胞(0～2)，红细胞少许。医者根据临床表现，拟诊为泌尿系结石，用阿托品肌注，呋喃咀啶口服，但疼痛仍不止，阵发性绞痛日夜皆作。1977年6月2日转来门诊。诊查：诊时诉右侧腰痛，阵发性绞痛日发3～4次，面色黧黑呈痛 苦貌，昨大便泄泻8次，质如稀水，今解3次，微咳有痰(有慢性气管炎史)，平素嗜酒，向有胃痛，脉滑，苔黄厚上罩灰黑。",
        result2_full=[
            ("G", 1.0, "流注膀胱"),
            ("G", 1.0, "流注膀胱"),
            ("B", 0.9575971731448764, "湿与热结"),
            ("J", 0.8904593639575972, "酒家湿蕴"),
            ("D", 0.7773851590106007, "热入血络"),
            ("E", 0.7173144876325089, "伏湿内蕴"),
            ("B", 0.4782608695652172, "湿与热结"),
            ("C", 0.08695652173913042, "脾虚中寒"),
            ("J", 0.04347826086956521, "酒家湿蕴"),
            ("A", 0.04347826086956521, "痰湿未化"),
            ("H", 0.04347826086956521, "化热生痰"),
        ],
        candidate3="A:肝火犯肺;B:血虚生风;C:脾肾两亏;D:痰火内蕴;E:气滞血瘀;F:肝脾不调;G:肝肾阴虚;H:热痰上蒙心窍;I:酒湿阻络;J:湿热下注",
    )
