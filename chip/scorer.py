import json
import logging
import os
import platform
import re
import sys
import traceback

import jieba
from rouge import Rouge

sys.path.append(".")
from conf import data_path

logger = logging.getLogger(__name__)


# 将句子分词并进行处理
def jieba_text(text):
    words = jieba.lcut(text)
    processed_text = " ".join(words)
    return processed_text


def score_rouge(s_candidate, s_reference):
    rouge = Rouge()
    score = rouge.get_scores(s_candidate, s_reference)
    return score[0]["rouge-1"]["f"]


# 输入预测txt，ground truth .json，计算详细得分
def doit(ground_truth, fn_predict, i_from=0, i_to=sys.maxsize):
    with open(ground_truth, "r", encoding="utf8") as fp_gt:
        gt = json.load(fp_gt)
        fp_gt.close()
    with open(fn_predict, "r", encoding="utf8") as fp_p:
        presult = fp_p.readlines()
        fp_p.close()

    s_list = []
    s_1_list = []
    s_2_list = []
    s_3_list = []
    s_4_list = []
    for i, r in enumerate(presult):
        parts = r.split("@")
        r_gt = gt[i + i_from]
        print(i, parts[0], r_gt["案例编号"])

        # if i < i_from:
        #     continue
        # if i >= i_to:
        #     break

        # 1
        p_1 = parts[1].split(";")
        gt_1 = r_gt["信息抽取能力-核心临床信息"].split(";")
        n1 = 0
        for item in p_1:
            if item in gt_1:
                n1 += 1
        s_1 = n1 / len(gt_1)
        print(f"\ts1={s_1} \t{n1}/{len(gt_1)} \n\t\t{p_1} \n\t\t{gt_1}")
        s_1_list.append(s_1)

        # 2
        p_2 = parts[2].split(";")
        gt_2 = r_gt["病机答案"].split(";")
        n2 = 0
        for item in p_2:
            if item in gt_2:
                n2 += 1
        s_2 = n2 / (len(gt_2) + len(p_2) - n2)
        print(
            f"\ts2={s_2} \t{n2}/ ({len(gt_2)} + {len(p_2)} - {n2}) \n\t\t{p_2} \n\t\t{gt_2}"
        )
        s_2_list.append(s_2)

        # 3
        p_3 = parts[3].split(";")
        gt_3 = r_gt["病机答案"].split(";")
        n3 = 0
        for item in p_3:
            if item in gt_3:
                n3 += 1
        s_3 = n3 / (len(gt_3) + len(p_3) - n3)
        print(
            f"\ts3={s_3} \t{n3}/ ({len(gt_3)} + {len(p_3)} - {n3}) \n\t\t{p_3} \n\t\t{gt_3}"
        )
        s_3_list.append(s_3)

        # 4
        p_4 = parts[4]
        gt_4 = r_gt["临证体会"] + r_gt["辨证"]
        s_4 = score_rouge(jieba_text(p_4), jieba_text(gt_4))
        print(f"\ts4={s_4} \n\t\t{p_4} \n\t\t{gt_4}")
        s_4_list.append(s_4)

        s = s_1 * 0.2 + s_2 * 0.3 + s_3 * 0.4 + s_4 * 0.1
        s_list.append(s)
        print(f"{parts[0]}={s}")
        print("---------------------------------------")
    print(f"{sum(s_list)/len(presult)}")
    print(
        f"{sum(s_1_list)/len(presult)} {sum(s_2_list)/len(presult)} {sum(s_3_list)/len(presult)} {sum(s_4_list)/len(presult)} "
    )


if __name__ == "__main__":
    doit(
        ground_truth=f"{data_path}/round1_traning_data/train.json",
        fn_predict=f"./temp/NE_train_14.txt",
        i_from=180,
        # i_to=200,
    )
