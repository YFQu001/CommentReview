import argparse
import pandas as pd
import torch
import time
import json
from tqdm import tqdm
from functools import partial
import os
import sys
from transformers import pipeline

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig # 量化
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from utils.prepare_datasets import prepare_datasets

def parse_args():
    """定义所有可配置的参数"""
    parser = argparse.ArgumentParser(description="使用 SFTTrainer 微调聊天模型")
    
    # --- 数据和列名参数 ---
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="SIMULATE", 
        help="训练数据的 CSV 文件路径。如果为 'SIMULATE'，将使用模拟数据。"
    )
    parser.add_argument("--object_col", type=str, default="评论对象名称", help="数据中包含'对象名称'的列名")
    parser.add_argument("--text_col", type=str, default="评论内容", help="数据中包含'评论内容'的列名")
    parser.add_argument("--status_col", type=str, default="审核状态", help="数据中包含'审核状态'(标签)的列名")
    parser.add_argument(
        "--train_dataset_key", 
        type=str, 
        default="B", 
        choices=['A', 'B'], 
        help="选择训练集：'A' (不平衡) 或 'B' (平衡)"
    )

    # --- 模型和 Tokenizer 参数 ---
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="Qwen/Qwen1.5-1.8B-Chat", 
        help="要加载的预训练模型地址"
    )
    parser.add_argument("--max_seq_length", type=int, default=2048, help="SFTTrainer 的最大序列长度")

    # --- LoRA 参数 ---
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 的 r (rank)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA 的 alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules", 
        type=str, 
        nargs='+', 
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "score"],
        help="要应用 LoRA 的模块列表 (空格分隔)"
    )

    # --- 训练参数 (TrainingArguments) ---
    parser.add_argument("--output_dir", type=str, default="./chat_sft_results", help="训练输出和 checkpoints 的目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练的总轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="每个设备的训练 batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="每个设备的验证 batch size")
    parser.add_argument("--do_eval", type=bool, default=True)
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--logging_steps", type=int, default=10, help="记录日志的步数间隔")
    parser.add_argument("--eval_steps", type=int, default=500, help="评估模型的步数间隔")
    parser.add_argument("--save_steps", type=int, default=500, help="保存 checkpoint 的步数间隔")
    parser.add_argument("--save_total_limit", type=int, default=3, help="最多保存的 checkpoint 数量")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="学习率预热的比例")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型 (e.g., 'linear', 'cosine')")
    

    
    return parser.parse_args()

# --- 提示词模板 (全局常量) ---
PROMPT_TEMPLATE = """
你是一位华为鸿蒙应用的评论审核助手，你的任务是对用户提交的评论进行内容安全审核。

【审核目标】
请判断该评论内容是否安全、合规，可以正常展示在应用评论区中。

【审核标准】
以下类型的内容必须判定为【不合规】：
1. **包含敏感词**： 纯血鸿蒙。
1. **辱骂、人身攻击**：包含侮辱性词汇、地域攻击、歧视、阴阳怪气、恶意讽刺等。
2. **政治及国家相关敏感内容**：涉及国家领导人、政府、军队、外交、意识形态、政权制度、民族宗教敏感话题，或可能引发争议的国际政治事件。
3. **涉及竞争对手或其他品牌攻击**：包括贬低、诋毁、造谣或对其他品牌/公司/应用的比较。
4. **违法违规内容**：涉及色情、暴力、赌博、毒品、诈骗、违法活动等。
5. **传播谣言或不实信息**：包括未经证实的负面消息或引导性言论。
6. **其他违反公共秩序或社会道德的内容**。

【允许内容】
- 可发表个人使用体验、建议、bug反馈、功能需求、普通情绪表达（如“有点卡”“挺好用的”）。
- 可出现一般性情绪词（如“生气”“失望”），但不能包含攻击对象。

【应用名称】
{object_name}

【评论内容】
{text_content}

【输出格式】
仅需输出是否通过审核（严格使用JSON格式）：
{{
  "result": "通过" 或 "不通过"
}}

请基于以上标准进行严格、客观的判断。
"""

def create_prompt_completion_format(example, args):
    """
    将数据格式化为 'prompt' 和 'completion' 
    SFTTrainer 会自动识别这种格式，并启用 completion_only_loss
    """
    
    # "prompt" 包含了完整的用户输入模板
    prompt_content = PROMPT_TEMPLATE.format(
        object_name=example[args.object_col],
        text_content=example[args.text_col]
    )
    
    # "completion" 包含了模型应该生成的标准答案
    status = str(example[args.status_col])
    answer_content = f'{{\n  "result": "{status}"\n}}'
    
    # 关键：将数据分为 "prompt" 和 "completion" 字段
    # 注意：我们仍然使用聊天格式（列表+字典），SFTTrainer会处理它
    example["prompt"] = [
        {"role": "user", "content": prompt_content}
    ]
    example["completion"] = [
        {"role": "assistant", "content": answer_content}
    ]
    return example

def main(args):
    """主训练和评估逻辑"""
    global_rank = int(os.environ.get("RANK", 0))
    is_main_process = (global_rank == 0)

    if not is_main_process:
        # 禁用所有非主进程的 print() 输出 (包括来自 prepare_datasets 的)
        sys.stdout = open(os.devnull, 'w')
    
    # --- (可选) 确保你的 GPU 支持 bfloat16 ---
    if not torch.cuda.is_bf16_supported():
        print("警告：你的 GPU 不支持 bfloat16。将回退到 float16。")
        print("如果显存不足，训练可能会失败。")
        TORCH_DTYPE = torch.float16
    else:
        TORCH_DTYPE = torch.bfloat16
        print("--- 支持 bfloat16 ---")

    # 步骤1：加载和格式化数据
    print(f"--- 正在加载数据 (路径: {args.data_path}) ---")
    if args.data_path == "SIMULATE":
        print("使用模拟数据...")
        all_data_df = pd.DataFrame({
            args.object_col: ['对象A', '对象B'] * 21614 + ['对象A'],
            args.text_col: ['内容...'] * 43229,
            args.status_col: ['通过'] * 35239 + ['不通过'] * 7990
        }).sample(frac=1).reset_index(drop=True)
    else:
        try:
            all_data_df = pd.read_csv(args.data_path)
            print(f"成功从 {args.data_path} 加载了 {len(all_data_df)} 条数据")
        except FileNotFoundError:
            print(f"错误：找不到数据文件 {args.data_path}")
            return
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return

    data_splits = prepare_datasets(
        df=all_data_df,
        object_col=args.object_col,
        status_col=args.status_col
    )

    # 将 Pandas DFs 转换为 Hugging Face Datasets 并应用格式化
    dataset_A = Dataset.from_pandas(data_splits['experiment_A']['train_set'])
    dataset_B = Dataset.from_pandas(data_splits['experiment_B']['train_set'])
    dataset_valid = Dataset.from_pandas(data_splits['valid_set'])
    
    # 使用 functools.partial 来固定 create_chat_messages 的 args 参数
    map_func = partial(create_prompt_completion_format, args=args)
    # 定义要保留的新列
    keep_columns = ['prompt', 'completion']
    
    num_proc = max(os.cpu_count() // 2, 1) # 使用一半的 CPU 核心
    print(f"--- 使用 {num_proc} 个进程进行数据预处理 ---")
    # 修改 remove_columns 以便只保留 'prompt' 和 'completion'
    dataset_A = dataset_A.map(map_func, remove_columns=[c for c in dataset_A.column_names if c not in keep_columns], num_proc=num_proc) 
    dataset_B = dataset_B.map(map_func, remove_columns=[c for c in dataset_B.column_names if c not in keep_columns], num_proc=num_proc)
    dataset_valid = dataset_valid.map(map_func, remove_columns=[c for c in dataset_valid.column_names if c not in keep_columns], num_proc=num_proc)

    print("\n--- 'prompt'/'completion' 格式数据示例 ---")
    print(dataset_A[0])

    # 步骤 2: 加载模型和 Tokenizer
    print(f"\n--- 正在加载模型: {args.model_id} (使用 {TORCH_DTYPE}) ---")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=TORCH_DTYPE,
        # device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        



    # 步骤 3: 配置 LoRA (PEFT)
    print(f"--- 配置 LoRA (r={args.lora_r}, alpha={args.lora_alpha}) ---")
    print(f"LoRA 目标模块: {args.lora_target_modules}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules, 
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 步骤 4: 配置 SFTTrainer
    sft_config = SFTConfig(
        # --- SFTTrainer 特有参数 ---
        max_length=args.max_seq_length,      # (关键) 映射到 SFTConfig 的 max_length
        packing=False,                         # 您的数据格式不需要 packing
        
        # --- 原 TrainingArguments 参数 ---
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        
        do_eval=args.do_eval,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        
        fp16=(TORCH_DTYPE == torch.float16), 
        bf16=(TORCH_DTYPE == torch.bfloat16),
        report_to="none",
        seed=args.seed
    )

    # 选择训练集
    if args.train_dataset_key == 'A':
        train_dataset = dataset_A
        print("\n--- 选择 [实验 A] (1:4.4 不平衡) 作为训练集 ---")
    else:
        train_dataset = dataset_B
        print("\n--- 选择 [实验 B] (1:1 平衡) 作为训练集 ---")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,           
        args=sft_config,               # 修正: 传入 SFTConfig 对象
        peft_config=lora_config,
        train_dataset=train_dataset,
        eval_dataset=dataset_valid,
    )


    # 步骤 5: 训练
    print(f"\n--- 开始训练 (实验 {args.train_dataset_key}) ---")
    trainer.train()

    # 保存最终的模型适配器
    print("--- 训练完成，保存最终适配器 ---")
    final_model_path = f"{args.output_dir}/final_model"
    trainer.save_model(final_model_path)
    print(f"模型已保存至: {final_model_path}")

    # 步骤 6: 在完整的验证集上推理，并计算准确率
    print("\n--- 开始在【完整验证集】上进行推理评估 (批量模式) ---")

    trained_model = trainer.model 
    trained_model.eval() 

    # 2. 获取原始的 Pandas DataFrame 和真实标签
    validation_df = data_splits['valid_set'] 
    ground_truths = validation_df[args.status_col].astype(str).str.strip().tolist()
    print(f"将在 {len(validation_df)} 条验证数据上运行推理...")

    # 3. 初始化 text-generation pipeline
    # (重要) pipeline 会自动处理设备、padding 和批量推理
    pipe = pipeline(
        "text-generation",
        model=trained_model,
        tokenizer=tokenizer,
        device=trained_model.device, # 使用模型所在的设备
        torch_dtype=TORCH_DTYPE
    )

    # 4. (一次性) 准备所有 prompts
    all_messages = []
    for index, row in validation_df.iterrows():
        prompt_text = PROMPT_TEMPLATE.format(
            object_name=row[args.object_col],
            text_content=row[args.text_col]
        )
        all_messages.append([{"role": "user", "content": prompt_text}])

    # 5. 定义生成参数
    generation_kwargs = {
        "max_new_tokens": 40,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "return_full_text": False, # (重要) 仅返回生成的部分
    }

    # 6. (核心) 运行批量推理
    predictions = []
    valid_labels = ["通过", "不通过", "解析失败"] 
    batch_size = args.per_device_eval_batch_size # 使用您在参数中定义的评估 batch_size
    
    print(f"--- 开始批量推理 (Batch Size: {batch_size}) ---")
    start_time = time.time()

    # pipe() 会自动处理 tqdm 和批量
    for out in tqdm(pipe(all_messages, batch_size=batch_size, **generation_kwargs), total=len(all_messages)):
        
        # pipeline 输出结构: [{'generated_text': '...json...'}]
        # 因为我们设置了 return_full_text=False, 'generated_text' 只包含
        # assistant 的回复
        if not out or 'generated_text' not in out[0]:
            pred_label = "解析失败"
            predictions.append(pred_label)
            continue
            
        pred_text = out[0]['generated_text']
        
        pred_label = "解析失败" # 默认值
        try:
            if "{" in pred_text and "}" in pred_text:
                json_str = pred_text[pred_text.find("{") : pred_text.rfind("}") + 1]
                data = json.loads(json_str)
                
                if "result" in data:
                    pred_label = str(data["result"]).strip()
            else:
                # 兼容解析失败但包含关键词的情况
                if "通过" in pred_text:
                    pred_label = "通过"
                elif "不通过" in pred_text:
                    pred_label = "不通过"
                    
        except Exception as e:
            pass 
            
        predictions.append(pred_label)

    # 7. 计算并打印最终指标 (与之前相同)
    end_time = time.time()
    print(f"\n--- 推理完成 ---")
    print(f"总耗时: {end_time - start_time:.2f} 秒")

    accuracy = accuracy_score(ground_truths, predictions)
    print(f"\n--- 整体准确率 (Accuracy) ---")
    print(f"{accuracy * 100:.2f} %")

    print(f"\n--- 分类报告 (Classification Report) ---")
    report = classification_report(
        ground_truths, 
        predictions, 
        labels=valid_labels,
        zero_division=0
    )
    print(report)

if __name__ == "__main__":
    args = parse_args()
    main(args)
