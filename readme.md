# 比赛概览

kaggle链接：https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview

## 1. 任务目标

* **任务类型：** 二元分类。
* **目标：** 预测一个Reddit评论（`body`）是否违反了其所在版块（`subreddit`）的特定规则（`rule`），输出违规的概率（`rule_violation`）。

## 2. 核心挑战：未见规则泛化 (Zero-Shot)

* **关键限制：** 训练集（`train.csv`）仅包含**2条**规则。测试集（`test.csv`）包含多条（例如4条）**未见过**的新规则。
* **问题本质：** 这不是传统的文本分类，而是**自然语言推理 (NLI)** 或**语义匹配**。模型必须学会“理解”规则，而不是“记忆”规则。

## 3. 数据集字段

* `body`: (str) 待判断的评论文本。
* `rule`: (str) 该版块的规则描述。
* `subreddit`: (str) 评论所在的版块名称（上下文）。
* `positive_example_{1,2}`: (str) **关键特征**。两个*违反*该规则的评论样例。
* `negative_example_{1,2}`: (str) **关键特征**。两个*不违反*该规则的评论样例（用于定义规则边界）。
* `rule_violation`: (int) **目标标签**（0或1）。

## 4. 评估指标

* **Column-averaged AUC (列平均AUC)**：
* 评估系统会**为每一条规则**（例如，测试集中的4条新规则）**单独计算AUC**，最后取所有规则AUC的算术平均值。
* **启示：** 模型必须在*所有*规则上都表现良好，在任何一条规则上失败都会严重拉低总分。

# 实现方案一

这是一个基于**LLM指令微调 (Instruction Finetuning)** 的SOTA方案总结。

https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/1st-place-solution

## 1. 核心思路：问答(QA)任务重构

* 将问题从“分类”重构为“指令遵循/问答”。
* **提示 (Prompt) 工程：** 使用LLM的聊天模板，构建一个包含所有上下文的提示，要求模型最后只回答 "Yes" 或 "No"。

```prompt
[USER]
You are a moderator. Here is the rule: [rule_text]
Here is a VIOLATING example: [positive_example_1]
Here is a NON-VIOLATING example: [negative_example_1]

Evaluate this comment:
[body_text]

Did this comment violate the rule? Answer with only "Yes" or "No".

[ASSISTANT]
```

## 2. 关键技术点

* **验证策略：**
    * **放弃本地验证集**（因为它不含新规则，有误导性）。
    * **完全信任 Public LB** 作为验证手段。

* **核心技巧 (数据增强)：**
    * **将`test.csv`的样例用作训练数据**。`test.csv`虽然没有`body`的标签，但它的`positive_example`和`negative_example`列本身就是“有标签”的数据。
    * **具体做法：** 创建新样本 `(body=test.positive_example, label=Yes)` 和 `(body=test.negative_example, label=No)`，并将其加入训练集。
    * **效果：** 这将问题难度从 **Zero-Shot**（零样本）**降维到了 Few-Shot**（少样本），因为模型在训练时已经“见过”了新规则的官方正反案例。

* **高效微调 (Training)：**
    * **`unsloth`：** 使用`unsloth`库进行高效的内存优化训练（如LoRA）。
    * **损失屏蔽 (Loss Masking)：** 在计算损失时，**只计算 "Yes" 和 "No" 这一个Token的损失**。忽略所有提示（Prompt）和回答模板（如`\n`）的损失，使所有梯度都集中在“做决策”这最关键的一步上。

* **高效推理 (Inference)：**
    * **`forward()`代替`generate()`：** 不使用缓慢的自回归`generate()`，而是直接调用`model.forward()`，获取最后一个Token位置上所有词的`logits`（概率得分）。
    * **稳健打分 (Robust Scoring)：** 收集 "Yes" (如`["Yes", "yes", "True"]`) 和 "No" (如`["No", "no", "False"]`) 的所有变体。
    * **最终分数：** `Score = logits(Yes_variants) - logits(No_variants)`。这是一个代表“信心”的对数几率比。

* **后处理 (Post-Processing)：**
    * **规则内归一化 (Per-rule Normalization)：** 这是针对`column-averaged AUC`指标的关键优化。
    * **做法：** 对**每条规则**内的所有预测分数，**独立**进行最小-最大归一化（Min-Max Scaling），将得分拉伸到 `[0, 1]` 区间。
    * **效果：** 消除模型对不同规则的打分偏见（bias），使所有模型的输出尺度统一，极大提升了模型融合（Ensemble）的效果。

# 实现方案二：测试时训练 (TTT) 与堆叠 (Stacking)

这是一个极其精巧的“Kaggle-Style”方案，其核心不是在“提交前”训练一个万能模型，而是在“推理时”的12小时内，利用`test.csv`中的样例（`examples`）**现场训练**一系列模型。

https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/3rd-place-solution

## 1. 核心思路：两层堆叠 (Two-Level Stacking)

本方案的本质是一个两层堆叠架构：

* **Level 1 (特征提取器)：** 包含各种模型（LLM, DeBERTa, BGE）。它们唯一的任务是“理解”文本，并将每个`body`转换为一个**嵌入向量（Embedding）**。
* **Level 2 (分类器)：** 包含传统的机器学习模型（如 **LightGBM**, **XGBoost**, **SVC**）。它们在Level 1输出的“嵌入向量”上进行训练和预测。

**最终的预测结果，是所有Level 2模型（LGBM/XGB等）输出概率的加权平均。**

## 2. 架构：快慢组合 (Fast & Slow Ensembles)

为了在12小时的推理时限内最大化性能和多样性，作者将Level 1模型分为两组：

### a) "慢"组合 (Slow Ensemble) - [LLM作为特征提取器]

这是方案的主力，使用大型语言模型（如Qwen-14b）提取高质量特征。

* **阶段 1：离线预训练 (提交前)**
    * **目的：** 让LLM“预习”所有规则。
    * **做法：** 在**所有**`examples`（来自`train.csv`和`test.csv`）上微调 (LoRA) LLM。
    * **提示：** 使用 "Zero-Shot Prompt" 格式（即提示中只包含`rule`和`body`，不包含`examples`作为上下文），但这**不代表**模型没学过`examples`，它在微调数据中已经学过了。
    * **产物：** 一套 LoRA 权重。

* **阶段 2：TTT (推理时)**
    1.  **加载模型：** 加载LLM + 阶段1的LoRA权重。
    2.  **提取特征：**
        * 用此LLM提取`train.csv`中所有`body`的嵌入，得到 `train_embeddings`。
        * 用此LLM提取`test.csv`中所有`body`的嵌入，得到 `test_embeddings`。
    3.  **训练Level 2：** **当场训练**一个LGBM/XGBoost模型：`model.fit(X=train_embeddings, y=train_labels)`。
    4.  **预测：** `predictions = model.predict_proba(test_embeddings)`。

### b) "快"组合 (Fast Ensemble) - [小模型作为特征提取器]

这些模型更小，**完全在TTT的12小时内现场训练**，用于增加模型多样性。架构与“慢”组合相同。

* **模型1: `bge-base` (Triplet Loss)**
    * **TTT训练：** 在所有`examples`上训练度量学习。
    * **数据增强：** 使用“Flipped Rules”（反转规则语义并交换正负样本）技巧。
    * **特征：** `concat([body_embed - pos_centroid, body_embed - neg_centroid])`，即评论向量到“平均违规”和“平均不违规”两个中心的距离。
    * **Level 2：** 在这些特征上训练LGBM/XGB。

* **模型2: `Qwen-Embedding` (ArcFace Loss)**
    * **TTT训练：** 在`body`上训练ArcFace Loss，使违规/不违规的嵌入分开。
    * **Level 2：** 在提取的嵌入上训练LGBM/XGB。

* **模型3: `DeBERTa-v3-small` (Cross Entropy Loss)**
    * **TTT训练：** 在`'{rule}[SEP]{body}'`输入上训练得到`[CLS]`,训练后的`[CLS]`学会总结 rule 和 body 之间的关系。也就是将DeBERTa模型作为特征提取器，把每一对 (rule, body) 文本转换成一个高维的、浓缩了语义信息的数字向量。
    * **Level 2：** 在`[CLS]` token的嵌入上训练LGBM/XGB。

## 3. 最终融合 (Final Ensemble)

* **关键点：** **不是**平均“嵌入向量”。
* **做法：** 方案中产生了许多Level 2模型的预测概率（`Preds_Slow_1`, `Preds_Fast_1`, `Preds_Fast_2`...）。
* **最终提交：** 对所有这些**最终概率**进行加权平均。