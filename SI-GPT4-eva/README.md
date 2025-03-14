# 红楼梦风格强度评估

使用两种方法来评估文本是否符合《红楼梦》的写作风格：
1. 基于的 Chinese-BERT-wwm-ext 模型
2. 基于 GPT-4 的评分系统

## 项目结构

```
.
├── train_style_classifier.py  # BERT模型训练脚本
├── style_intensity_evaluator.py  # BERT模型评估脚本
├── gpt4_style_evaluator.py  # GPT-4评分脚本
├── requirements.txt  # 项目依赖
├── style_classifier/  # 训练好的模型文件
│   ├── config.json
│   └── model.safetensors
├── result.json  # 待评估的问答数据
├── Train_DRC.json  # 训练数据（如需重新训练）
└── Valid_DRC.json  # 验证数据（如需重新训练）
```

## 功能特点

- BERT 模型评估：
  - 使用 Chinese-BERT-wwm-ext 预训练模型作为基础进行微调
  - 采用全词遮罩（Whole Word Masking）策略，更适合中文处理
  - 支持批量处理，提高推理效率
  - 自动检测并使用 GPU（如果可用）
  - 计算风格强度（Style Intensity, SI）：0-1 之间的数值

- GPT-4 评估：
  - 使用 GPT-4 模型对文本进行智能评分
  - 评分范围 0-10，反映文本与红楼梦风格的匹配度
  - 支持断点续评，自动保存评分结果
  - 实时显示评分进度和统计信息

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据准备
确保 `result.json` 文件存在于项目根目录，文件格式如下：
```json
[
    {
        "question": "问题1",
        "daiyu_answer": "待评估的文本1"
    },
    {
        "question": "问题2",
        "daiyu_answer": "待评估的文本2"
    }
]
```

### 2. BERT 模型评估
直接使用已训练好的模型进行评估：
```bash
python style_intensity_evaluator.py
```
评估结果将保存在 `si_results.json` 中。

### 3. GPT-4 评分
运行 GPT-4 评分脚本（需要设置 OpenAI API Key）：
```bash
python gpt4_style_evaluator.py
```
评分结果将保存在 `gpt4_scored_results.json` 中。

### 4. （可选）重新训练模型
如果需要重新训练 BERT 模型：
```bash
python train_style_classifier.py
```
需确保 `Train_DRC.json` 和 `Valid_DRC.json` 存在。

## 评分标准

- BERT 模型评估：
  - 输出风格强度 (SI)：0-1 之间的数值
  - 大于 0.5 判定为符合红楼梦风格
  - 计算总体风格匹配度

- GPT-4 评分标准：
  - 0-3分：几乎不符合红楼梦风格
  - 4-6分：部分符合红楼梦风格
  - 7-8分：较好地符合红楼梦风格
  - 9-10分：完全符合红楼梦风格

## 注意事项

1. 模型文件
   - `style_classifier` 文件夹包含训练好的模型，请勿删除
   - 如果重新训练，原模型会被覆盖

2. 运行环境
   - 支持 CPU 和 GPU 运行
   - 如果有 CUDA 环境会自动使用 GPU
   - 确保有足够的内存和磁盘空间

3. GPT-4 评分
   - 需要有效的 OpenAI API Key
   - 支持中断后继续评分
   - 每次评分会自动保存结果

4. 数据文件
   - 评估时需要 `result.json`
   - 训练时需要 `Train_DRC.json` 和 `Valid_DRC.json`
   - 所有文件都应使用 UTF-8 编码 