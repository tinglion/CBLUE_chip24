import argparse
import os
import platform
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertForTokenClassification,
    BertTokenizer,
)

sys.path.append(".")
from cblue.data import EEDataProcessor, EEDataset
from cblue.metrics import ee_commit_prediction, ee_metric
from cblue.trainer import EETrainer

device = "cpu" if platform.system() == "Windows" else "cuda"

# "./data/output/ee/chinese-bert-wwm-ext"
# data_path = (
#     "D:/fdata/ai/med/CBLUE"
#     if platform.system() == "Windows"
#     else "/mnt/windows/sting/data/"
# )

model_name = "data/chip2024/output/ee/chinese-bert-wwm-ext"
data_path = (
    "D:/fdata/ai/med/CHIP2004"
    if platform.system() == "Windows"
    else "/mnt/windows/sting/data/CHIP2004"
)

data_processor = EEDataProcessor(root=f"{data_path}/round1_traning_data_ee")
print(f"@sting num_labels={data_processor.num_labels}")

model = BertForTokenClassification.from_pretrained(
    model_name, num_labels=data_processor.num_labels
)
model.to(device)
tokenizer = BertTokenizer.from_pretrained(model_name)
# 确保模型在评估模式下
model.eval()

# inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
# input_ids = inputs["input_ids"].to(device)
# token_type_ids = inputs["token_type_ids"].to(device)
# attention_mask = inputs["attention_mask"].to(device)


def preprocess(data):
    outputs = {"text": [], "label": [], "orig_text": []}
    for text in data:
        text_a = [
            "，" if t == " " or t == "\n" or t == "\t" else t
            for t in list(text.lower())
        ]
        outputs["text"].append(text_a)
        outputs["orig_text"].append(text)
    return outputs


def predict(test_dataloader, test_dataset):
    predictions = []
    for step, item in enumerate(test_dataloader):
        model.eval()

        input_ids = item[0].to(device)
        token_type_ids = item[1].to(device)
        attention_mask = item[2].to(device)

        with torch.no_grad():
            # print(f"@sting {input_ids}")

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            # logits = outputs.detach()
            logits = outputs[0].detach()
            # print(f"@sting {logits}")

            active_index = attention_mask == 1
            preds = logits.argmax(dim=-1)  # .cpu()
            # print(f"@sting {len(preds)} {len(active_index)}")
            # print(f"@sting {preds[0]} {active_index[0]}")

            for i in range(len(active_index)):
                predictions.append(preds[i][active_index[i]].tolist())
    test_inputs = test_dataset.texts
    predictions = [pred[1:-1] for pred in predictions]
    predicts = data_processor.extract_result(predictions, test_inputs)
    return predicts


def do_predict(text_list):

    test_dataset = EEDataset(
        preprocess(text_list),
        data_processor,
        tokenizer,
        mode="test",
        ngram_dict=None,
        max_length=128,
        model_type="bert",
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1)
    results = predict(test_dataloader, test_dataset)
    return results


if __name__ == "__main__":
    result = do_predict(
        [
            # "六、新生儿疾病筛查的发展趋势自1961年开展苯丙酮尿症筛查以来，随着医学技术的发展，符合进行新生儿疾病筛查标准的疾病也在不断增加，无论在新生儿疾病筛查的病种，还是在新生儿疾病筛查的技术方法上，都有了非常显著的进步。",
            # "如有哮喘家族史、反复发作、无前驱感染而突然发作、嗜酸性粒细胞增多、对单剂β<sub>2</sub>激动剂吸入反应良好而且迅速提示哮喘可能。",
            "王某，女，38岁。初诊：1962年7月10日。主诉及病史：营养不良。引起肝大面肿；经事到期未至，已历5个月。诊查：面色咣白，神疲嗜睡，脉虚弱无力，舌淡白。",
            "孙某，男，64岁。初诊:1972年9月13日。主诉及病史:8月21日晚，突感舌根发硬，说话吐字不清。次日，自觉步态不稳，左手持物不紧，有时自行落地，口中流涎，舌稍向左偏。在某医院检查，诊为闭塞性脑血管病。用血管舒缓素、烟酸、针灸等治疗，病情略有好转。诊查:体形稍胖，右侧眼裂较小，鼻唇沟变浅，口角略向左侧歪斜，左手握力较差。伸舌向左偏斜，舌苔根部稍厚，脉弦紧。",
        ]
    )
    print(result)
