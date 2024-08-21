# 1 临床表现信息（实体）：
# template_prompt1 = """您是一位中医学专家，请根据下面这段文字提取全面的临床表现信息(完整全面提取，包括主述和病史，但不包括体温等指标型数据)，严格按照json格式返回。
# 已知部分信息：{sugguest1}
# 严格按照输出json格式：{"临床表现信息": ["感染新冠", "大便干", "咳嗽", "脉滑", "舌苔微黄"]}
template_prompt1 = """您是一位中医学专家，请根据下面这段文字提取主要的临床表现信息（包括症状描述，包括主述和病史，不包括体温等指标型数据，不要标点符号），用json格式返回。
注意严格保留原文。
病情描述：{raw_text}
严格按照输出json格式：{"临床表现信息": {sugguest1}}
"""

# 2 病机 {"唾痰": ["A", 0]}
# template_prompt2 = """您是一位中医学专家，请针对每条临床表现信息，结合病情描述，从病机候选项中，选择最相关的一条或多条记录标签（A~Z），并根据相关性和重要性打分，分值范围1、2或者3，分值越高重要性越高，不相关的不要显示，严格按照json格式输出。
template_prompt2 = """您是一位中医学专家，结合中医相关专业知识，请针对每条临床表现信息，结合病情描述，从病机候选项中选择一条最相关的记录标签（A~Z），并根据相关性和重要性打分，分值范围1、2或者3，最高3分，分值越高重要性越高，不相关的不要显示，严格按照json格式输出。
注意不要Explanation。

临床表现信息：{result1}

病情描述：{raw_text}

病机候选项：{candidate2}

参考病机推理结果：{sugguest2}

严格按照输出json格式：{example2}
"""

# 3 证候 {"肾水不足": ["B", 0]}
# template_prompt3 = """您是一位中医学专家，请针对每条病机信息，结合病情描述，从证候候选项中，选择相关的一条或多条记录标签（A~Z），并根据相关性和重要性打分，分值范围1、2或者3，分值越高重要性越高，不相关的不要显示，严格按照json格式输出。
template_prompt3 = """您是一位中医学专家，结合中医相关专业知识，请针对每条病机信息，结合病情描述，从证候候选项中选择一条最相关的记录标签（A~Z），并根据相关性和重要性打分，分值范围1、2或者3，最高3分，分值越高重要性越高，不相关的不要显示，严格按照json格式输出。
注意不要Explanation。

病机信息：{result2}

病情描述：{raw_text}

证候候选项：{candidate3}

参考证候推理结果：{sugguest3}

严格按照输出json格式：{example3}
"""

# 4 临证体会
template_prompt4 = """您是一位中医学专家，请根据提供的病情描述、病机信息、证候信息，总结临证体会和辨证，不要复述症状和病机，不要给出治疗方案，严格按照json格式输出。
病情描述：{raw_text}

病机信息：{result2}

证候信息：{result3}

严格按照输出json格式：{example4}
"""
template_example4 = {
    "临证体会": "本例因情志不舒,思虑过度,劳伤心脾,导致心阴亏损、气血亏耗,以致神不守舍,胆虚不眠",
    "辨证": "心胆亏虚",
}

# template_example4 = {
#     "临证体会": "string",
#     "辨证": "sting",
# }
# template_example4 = {
#     "临证体会": "此病例极为罕见，但治愈之理犹未尽解。中医学认为，肾气虚弱，或因生疮日久，或失治、复遭寒邪侵入，与脓毒 凝结，借人之气血化成多骨。多见于腮、腭、牙床、眼胞、颏下等部位。患者溺时晕厥，其脉虚大，重按无力，两尺尤甚，且年逾花甲，说明肾气已虚；虽证见患处肿硬疼痛，张口困难，亦非实热，实乃肾元不固、虚火上炎之故也。",
#     "辨证": "辨证：断为伤寒夹食之证“伤寒脉弦细，头痛发热者属少阳，少阳不可发汗，发汗则谵语，此属胃，胃和则愈”因食郁于胃，寒伤于表，表里拒遏，枢机不利，上脘不通，津液不行",
# }
