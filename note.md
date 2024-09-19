# note

watch -n 1 nvidia-smi

## TODO

* finetune2
* finetune3
* 对抗

## chip

### 1 无监督，优化prompt

### 2 +训练bert模型：实体、关系

```bash
python ./baselines/run_classifier.py \
 --data_dir=/mnt/windows/sting/data/CHIP2004/round1_traning_data_ee \
 --model_type=bert --model_dir=/mnt/windows/sting/models --model_name=chinese-bert-wwm-ext --task_name=ee \
 --output_dir=data/chip2024/output --result_output_dir=data/chip2024/result_output \
 --do_train --max_length=128 --train_batch_size=16 --eval_batch_size=16 --learning_rate=3e-5 --epochs=5 --warmup_proportion=0.1 --earlystop_patience=100 --max_grad_norm=0.0 --logging_steps=10 --save_steps=10 --seed=2021
```

### 3 qwen only

## track

* CMeEE: 实体提取
* CMeIE：casrel 实体、关系提取
* CHIP-CDEE：该数据集由医渡云（北京）技术有限公司创建，任务为从中文电子病历中挖掘出临床事件
* CHIP-CDN：实体进行语义标准化，医渡云
  * 召回候选，相似度计算
* CHIP-CTC：句子分类，同济大学生命科学与技术学院
  * 定义了44种筛选标准语义类别：Bedtime、Life Expectancy、Addictive Behavior、Nursing、Device、Symptom、Pregnancy-related Activity、Ethnicity、Therapy or Surgery、Age、Encounter、Special Patient Characteristic、Sexual related、Multiple、Diet、Capacity、Organ or Tissue Status、Pharmaceutical Substance or Drug、Healthy、Oral related、Blood Donation、Risk Assessment、Alcohol Consumer、Sign、Non-Neoplasm Disease Stage、Enrollment in other studies、Receptor Status、Consent、Researcher Decision、Gender、Address、Data Accessible、Education、Smoking Status、Compliance with Protocol、Literacy、Ethical Audit、Allergy Intolerance、Disabilities、Disease、Laboratory Examinations、Diagnostic、Neoplasm Status、Exercise

https://mp.weixin.qq.com/s/O4r-gSr3EWdoCG5nHZIghQ?spm=a2c22.12282016.0.0.7a5d2195C3F5Td

## bash

```bash
conda create -n cblue python=3.12 pytorch torchvision torchaudio cpuonly -c pytorch -y

conda create -n cblue python=3.12
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### CMeEE

```powershell
# 10.5 hours cpu only
python .\baselines\run_classifier.py --data_dir=D:/fdata/ai/med/CBLUE/CMeEE-V2 --model_type=bert --model_dir=D:/fdata/ai/l --model_name=chinese-bert-wwm-ext --task_name=ee --output_dir=data/output --result_output_dir=data/result_output --do_train --max_length=128 --train_batch_size=16 --eval_batch_size=16 --learning_rate=3e-5 --epochs=5 --warmup_proportion=0.1 --earlystop_patience=100 --max_grad_norm=0.0 --logging_steps=200 --save_steps=200 --seed=2021

python baselines/run_classifier.py --data_dir=D:/fdata/ai/med/CBLUE/CMeEE-V2 --model_type=bert --model_dir=./data/output/ee --model_name=chinese-bert-wwm-ext --task_name=ee --output_dir=data/output --result_output_dir=data/result_output --do_predict    --max_length=128 --eval_batch_size=1   --seed=2021
```

#### linux

```bash
python ./baselines/run_classifier.py --data_dir=/mnt/windows/sting/data/CMeEE-V2 --model_type=bert --model_dir=/mnt/windows/sting/models --model_name=chinese-bert-wwm-ext --task_name=ee --output_dir=data/output --result_output_dir=data/result_output --do_train --max_length=128 --train_batch_size=16 --eval_batch_size=16 --learning_rate=3e-5 --epochs=5 --warmup_proportion=0.1 --earlystop_patience=100 --max_grad_norm=0.0 --logging_steps=200 --save_steps=800 --seed=2021
```

### CMeIE

```bash
python baselines/run_ie.py         --data_dir=D:/fdata/ai/med/CBLUE/CMeIE-V2        --model_type=bert        --model_dir=D:/fdata/ai/l          --model_name=chinese-bert-wwm-ext         --task_name=ie        --output_dir=data/output        --result_output_dir=data/result_output         --do_train         --max_length=128         --train_batch_size=32         --eval_batch_size=64         --learning_rate=3e-5         --epochs=7         --warmup_proportion=0.1         --earlystop_patience=100         --max_grad_norm=0.0         --logging_steps=200         --save_steps=200         --seed=2021
        
python baselines/run_ie.py --data_dir=D:/fdata/ai/med/CBLUE/CMeIE-V2        --model_type=bert        --model_dir=./data/output/ie       --model_name=chinese-bert-wwm-ext         --task_name=ie        --output_dir=data/output        --result_output_dir=data/result_output --do_predict  --max_length=128  --eval_batch_size=32
```

#### linux

```bash
# /mnt/windows/sting/models   data/output/ee  
nohup python baselines/run_ie.py    --data_dir=/mnt/windows/sting/data/CMeIE-V2    --model_type=bert     --model_dir=/mnt/windows/sting/models         --model_name=chinese-bert-wwm-ext    --task_name=ie      --output_dir=data/output     --result_output_dir=data/result_output         --do_train         --max_length=128         --train_batch_size=32         --eval_batch_size=64         --learning_rate=3e-5         --epochs=7     --warmup_proportion=0.1         --earlystop_patience=100         --max_grad_norm=0.0         --logging_steps=200         --save_steps=1000         --seed=2021 >logs/ie_train.log 2>&1 &

python baselines/run_ie.py --data_dir=/mnt/windows/sting/data/CMeIE-V2        --model_type=bert        --model_dir=./data/output/ie       --model_name=chinese-bert-wwm-ext         --task_name=ie        --output_dir=data/output        --result_output_dir=data/result_output --do_predict  --max_length=128  --eval_batch_size=32
```


您是一位中医学专家，结合中医相关专业知识，请从病情描述提取其他医疗结构的历史诊断，并从病机候选中找到推理出来的病机，用json格式返回。
注意不要提取当前诊查信息。
注意诊断名称不是症状。

病情描述：吴某，女，21岁。初诊：1975年3月16日。主诉及病史：患者因情绪刺激，患精神分裂症后住院1年半，出院后病症未见显著改善。诊查：刻诊面色眺白而浮肿，语言能够对答而无伦次，自诉心慌、胆怯，耳边听到有人讲话，大便干结。家属诉病人有时翻眼睛，有时发抖，有时胡思乱想；出言不伦，有时大声吵闹。行为幼稚，贪吃懒做。诊其脉促，舌淡边有齿印。

病机候选项：A:气虚不能固卫腠理;B:气血瘀滞;C:心阴;D:肝木上亢;E:气乱于上之候;F:肾精不足;G:中气渐虚;H:痰涎壅积;I:内陷血室;J:心脾之损

严格按照输出json格式：{
  "病史诊断":[
{
"诊断名称":"#原文#",
"推理病机":["#病机标签#"]
},{
"诊断名称":"#原文#",
"推理病机":["#病机标签#"]
}],
"当前诊查"："#原文#"
}