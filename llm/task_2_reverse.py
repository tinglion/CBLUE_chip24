import json
import os
import re
import sys

sys.path.append(".")
from chip import data_loader
from conf import test_only
from llm import llm_wrapper
from utils import dict_utils, file_utils, json_utils

template_prompt2_reverse = """您是一位中医学专家，结合中医相关专业知识，请从病情描述提取其他医疗结构的历史误诊诊断，并从病机候选中找到推理出来的病机，用json格式返回。
注意不要提取当前诊查信息。
注意诊断名称不是症状。

病情描述：{raw_text}

病机候选项：{candidate2}

严格按照输出json格式：{
  "病史诊断":[
{
"诊断疾病名称":"#原文#",
"推理病机":["#病机标签#"]
},{
"诊断疾病名称":"#原文#",
"推理病机":["#病机标签#"]
}],
"当前诊查"："#原文#"
}
"""


def predict_2_reverse(raw_text, candidate2, model_list=["openai"]):
    candidate2_map, candidate2_map_r = data_loader.parse_candidate(candidate2)

    prompt2 = template_prompt2_reverse.replace("{raw_text}", raw_text)
    prompt2 = prompt2.replace("{candidate2}", candidate2)
    print(f"promp2_reverse={prompt2}")

    response2_obj = []
    for llm_model in model_list:
        response2_tmp = llm_wrapper.chat_complete(prompt2, llm_name=llm_model)
        response2_tmp_obj = json_utils.cvt_str_to_obj(response2_tmp)
        print(f"{llm_model}={response2_tmp_obj}")
        response2_obj += response2_tmp_obj["病史诊断"]

    result2_reverse = []
    for r in response2_obj:
        if re.compile(r"(病|炎|症)").search(r["诊断疾病名称"]):
            for tag in r["推理病机"]:
                if tag not in candidate2_map and tag in candidate2_map_r:
                    tag = candidate2_map_r[tag]
                if tag in candidate2_map:
                    result2_reverse.append([tag, 1, candidate2_map[tag]])
    print(f"predict2_reverse={result2_reverse}")
    return result2_reverse


if __name__ == "__main__":
    # predict_2_reverse(
    #     raw_text="李某，男，50岁。初诊：1982年7月12日。主诉及病史：1981年3月发现舌面靠近根部有一约长3cm、宽0.8cm的椭圆形结节，高出舌面约0.2cm，状如荔壳，触之质地同舌无异。经吉林某医院切片化验，诊为“慢性炎症，典型增生”。经多方医治无效。现值返闽探亲之便，来我院门诊。诊查： 症见口干、舌麻微痛，常失眠、头晕心悸，形神倦怠，纳食尚可，时有便秘，既往曾患“梅核气”。有嗜烟酒史。舌淡红、苔浅剥，舌根结节同上述，脉细近弦。",
    #     candidate2="A:肾精亏虚;B:虚热上蒸;C:阳气虚;D:冲气上逆;E:肝气郁结;F:湿热内郁;G:疫毒乘虚内侵中焦;H:耗伤肾水;I:下元亏损;J:相火离位",
    # )
    # predict_2_reverse(
    #     raw_text="吴某，女，21岁。初诊：1975年3月16日。主诉及病史：患者因情绪刺激，患精神分裂症后住院1年半，出院后病症未见显著改善。诊查：刻诊面色眺白而浮肿，语言能够对答而无伦次，自诉心慌、胆怯，耳边听到有人讲话，大便干结。家属诉病人有时翻眼睛，有时发抖，有时胡思乱想；出言不伦，有时大声吵闹。行为幼稚，贪吃懒做。诊其脉促，舌淡边有齿印。",
    #     candidate2="A:气虚不能固卫腠理;B:气血瘀滞;C:心阴;D:肝木上亢;E:气乱于上之候;F:肾精不足;G:中气渐虚;H:痰涎壅积;I:内陷血室;J:心脾之损",
    # )
    # predict_2_reverse(
    #     raw_text="王某，女，38岁。初诊：1962年7月10日。主诉及病史：营养不良。引起肝大面肿；经事到期未至，已历5个月。诊查：面色咣白，神疲嗜睡，脉虚弱无力，舌淡白。",
    #     candidate2="A:胁肋内伤;B:气化失调;C:痰浊阻膈;D:寒邪;E:气血亏损;F:虚风易于上扰;G:肾阴不足;H:大量出血;I:饮邪化热;J:化燥伤阴",
    # )
    # predict_2_reverse(
    #     raw_text="陈某，男，61岁。初诊：1977年6月2日。主诉及病史：右侧腰部疼痛已1周，5月28日患者因持续性右下腹痛、阵发性加剧，历>时4小时而至某医院就诊，查血白细胞9600/mm3，中性79%，拟诊为阑尾炎，医用青霉素、颠茄合剂治之，痛未止。至5月29日因痛势加剧，又至某医院>急诊，诊时右腰部绞痛，放射至右腹部，摄腹部X线平片未见阳性结石影。尿常规检查：蛋白微量，上皮细胞极少，脓细胞(0～2)，红细胞少许。医者>根据临床表现，拟诊为泌尿系结石，用阿托品肌注，呋喃咀啶口服，但疼痛仍不止，阵发性绞痛日夜皆作。1977年6月2日转来门诊。诊查：诊时诉右侧腰痛，阵发性绞痛日发3～4次，面色黧黑呈痛苦貌，昨大便泄泻8次，质如稀水，今解3次，微咳有痰(有慢性气管炎史)，平素嗜酒，向有胃痛，脉滑，苔黄厚上罩灰黑。",
    #     candidate2="A:痰湿未化;B:湿与热结;C:脾虚中寒;D:热入血络;E:伏湿内蕴;F:慢惊风证;G:流注膀胱;H:化热生痰;I:血虚;J:酒家湿蕴",
    # )
    predict_2_reverse(
        raw_text="吴某，女，21岁。初诊：1975年3月16日。主诉及病史：患者因情绪刺激，患精神分裂症后住院1年半，出院后病症未见显著改善。诊查：刻诊面色 眺白而浮肿，语言能够对答而无伦次，自诉心慌、胆怯，耳边听到有人讲话，大便干结。家属诉病人有时翻眼睛，有时发抖，有时胡思乱想；出言不伦，有时 大声吵闹。行为幼稚，贪吃懒做。诊其脉促，舌淡边有齿印。",
        candidate2="A:气虚不能固卫腠理;B:气血瘀滞;C:心阴;D:肝木上亢;E:气乱于上之候;F:肾精不足;G:中气渐虚;H:痰涎壅积;I:内陷血室;J:心脾之损",
    )
