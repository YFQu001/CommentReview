import argparse
import pandas as pd
import torch
import time
import json
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report
from accelerate import Accelerator  # (!!!) 导入 Accelerator

from utils.prepare_datasets import prepare_datasets


# --- (!!!) 必须与训练时完全一致 (!!!) ---
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

def parse_args():
    """定义推理所需的参数"""
    parser = argparse.ArgumentParser(description="使用 Accelerate 进行多 NPU 批量推理")
    
    # --- 数据和列名参数 ---
    parser.add_argument("--data_path", type=str, required=True, help="【必需】原始训练数据xlxs 文件路径")
    parser.add_argument("--object_col", type=str, default="评论对象名称")
    parser.add_argument("--text_col", type=str, default="评论内容")
    parser.add_argument("--status_col", type=str, default="审核状态")

    # --- 模型和路径参数 ---
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen1.5-1.8B-Chat")
    parser.add_argument("--adapter_path", type=str, default="./chat_sft_results/final_model")

    # --- 推理参数 ---
    parser.add_argument(
        "--per_device_batch_size", 
        type=int, 
        default=8, 
        help="【每张NPU】的推理批量大小"
    )
    parser.add_argument("--max_new_tokens", type=int, default=40)
    
    return parser.parse_args()


# 新增：自定义数据整理器 (Collator) 
class DataCollatorForInference:
    def __init__(self, tokenizer, max_length=2048): # 增加 max_length
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """
        'batch' 是一个列表, 列表中的每一项是: [{"role": "user", "content": "..."}]
        """
        
        # 步骤 1：将 "batch" (list of conversations) 转换为 "list of strings"
        prompt_strings = []
        for conversation in batch:
            prompt_strings.append(
                self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False, # (!!!) 重点：只格式化为字符串
                    add_generation_prompt=True # 确保添加了模型的 "assistant" 提示符
                )
            )
            
        # 步骤 2：使用 tokenizer 批量处理字符串列表
        # 这将 100% 返回我们需要的字典 (BatchEncoding)
        inputs = self.tokenizer(
            prompt_strings,
            return_tensors="pt",
            padding=True,         # (!!!) Padding 在这里执行
            truncation=True,      # (!!!) 强烈建议加上截断
            max_length=self.max_length 
        )
        
        return inputs # inputs 现在保证是 {'input_ids': ..., 'attention_mask': ...}


def main(args):
    
    # --- 初始化 Accelerator ---
    # a.bf16 (bfloat16) 是 NPU 上的推荐配置
    accelerator = Accelerator(mixed_precision="bf16")
    
    # accelerator.device 会自动指向当前进程分配到的 NPU (e.g., 'npu:0', 'npu:1')
    device = accelerator.device
    
    # 使用 accelerator.print 替代 print
    # 只有主进程 (rank 0) 会实际打印，避免多卡重复输出
    accelerator.print(f"--- 运行在 {accelerator.num_processes} 个 NPU 上 ---")
    accelerator.print(f"--- 进程 {accelerator.process_index} 分配到 {device} ---")


    # --- 2. 加载数据 (所有进程都需要) ---
    accelerator.print(f"--- [进程 {accelerator.process_index}] 正在加载数据: {args.data_path} ---")
    try:
        all_data_df = pd.read_excel(args.data_path)
    except FileNotFoundError:
        accelerator.print(f"错误：找不到数据文件 {args.data_path}")
        return

    data_splits = prepare_datasets(
        df=all_data_df,
        object_col=args.object_col,
        status_col=args.status_col
    )
    validation_df = data_splits['valid_set'] 
    
    # 真实标签现在只在主进程的末尾需要，但数据在所有进程都需要准备
    ground_truths = validation_df[args.status_col].astype(str).str.strip().tolist()
    
    accelerator.print(f"--- 验证集加载完毕，共 {len(validation_df)} 条数据 ---")


# --- 3. 加载 Tokenizer ---
    accelerator.print(f"--- [进程 {accelerator.process_index}] 正在加载 Tokenizer: {args.model_id} ---")
    # (!!!) 注意：model_id 现在是 "./merged_model"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    accelerator.print(f"--- Tokenizer 设置完毕 (Padding Side: {tokenizer.padding_side}) ---")


    # --- 4. 加载 【已合并】 模型 ---
    # (!!!) 关键修改 (!!!)
    # 我们不再需要 base_model, PeftModel, merge_and_unload
    
    accelerator.print(f"--- [进程 {accelerator.process_index}] 正在加载【已合并】模型: {args.model_id} ---")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16, # 直接使用 bfloat16
        trust_remote_code=True
        # (!!!) 不要加 low_cpu_mem_usage 或 device_map
        # (!!!) 让 accelerate 自动处理模型的加载和 NPU 放置
    )
    
    model.eval() # 设置为评估模式

    # --- 5. 准备数据加载器 (DataLoader) ---
    
    # 5.1. 准备所有 Prompts (list of list of dicts)
    all_messages = []
    for index, row in validation_df.iterrows():
        prompt_text = PROMPT_TEMPLATE.format(
            object_name=row[args.object_col],
            text_content=row[args.text_col]
        )
        # 每一项都是一个独立的对话 (虽然只有一个 'user' turn)
        all_messages.append([{"role": "user", "content": prompt_text}])
        
    # 5.2. 初始化 Collator
    data_collator = DataCollatorForInference(tokenizer, max_length=1024)

    # 5.3. 创建 DataLoader
    # DataLoader 会加载 *全部* 数据。accelerator.prepare 会自动处理分发
    dataloader = DataLoader(
        all_messages,
        batch_size=args.per_device_batch_size,
        collate_fn=data_collator,
        shuffle=False # 评估时必须保持顺序
    )

    # --- 使用 Accelerator 准备模型和 DataLoader ---
    # accelerator.prepare 会自动处理 DDP/FSDP 包装和设备放置
    model, dataloader = accelerator.prepare(model, dataloader)


    # --- 手动批量推理循环 ---
    all_my_decoded_texts = [] # 存储 *当前 NPU 进程* 的预测结果
    
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    accelerator.print(f"--- [进程 {accelerator.process_index}] 开始批量推理 ---")
    start_time = time.time()
    
    # dataloader 现在只包含 *当前 NPU 进程* 应该处理的数据
    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            # 使用 accelerator.unwrap_model 获取原始模型以调用 .generate
            # 这在 FSDP 等策略下是必需的
            unwrapped_model = accelerator.unwrap_model(model)
            
            # 获取 prompt 的长度，以便稍后剥离
            prompt_len = batch['input_ids'].shape[1]
            
            outputs = unwrapped_model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                **generation_kwargs
            )
        
        # 剥离 prompt，只保留新生成的 token
        new_tokens = outputs[:, prompt_len:]
        
        # 解码当前批次
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        # 收集当前进程的所有解码文本
        all_my_decoded_texts.extend(decoded)

    accelerator.print(f"--- [进程 {accelerator.process_index}] 推理完成 ---")
    
    # --- 8. 收集所有 NPU 进程的结果 ---
    
    # accelerator.gather_object 会将 *每个进程* 的
    # `all_my_decoded_texts` 列表收集到一个大列表中
    # e.g., 在 2 个 NPU 上:
    # 结果会是 [ [NPU0的预测], [NPU1的预测] ]
    gathered_predictions_list = accelerator.gather_object(all_my_decoded_texts)

    # --- 9. (!!!) 仅在主进程上计算指标 ---
    if accelerator.is_main_process:
        
        # (!!!) 展平列表
        # [ [NPU0_1, NPU0_2], [NPU1_1, NPU1_2] ] -> 
        # [ NPU0_1, NPU0_2, NPU1_1, NPU1_2 ]
        flat_predictions_text = [
            item for sublist in gathered_predictions_list for item in sublist
        ]

        # (!!!) 重点：DDP Sampler 可能会添加重复的 "padding" 样本
        # 以确保每个 NPU 上的批次数相同。
        # 我们必须裁剪到原始验证集的大小。
        original_size = len(ground_truths)
        final_predictions_text = flat_predictions_text[:original_size]
        
        # 检查是否匹配
        if len(final_predictions_text) != original_size:
            print(f"警告：收集到的预测 ({len(final_predictions_text)}) 与")
            print(f"真实标签 ({original_size}) 数量不匹配！")

        
        # --- (以下逻辑与单卡脚本中的 JSON 解析相同) ---
        
        predictions = []
        valid_labels = ["通过", "不通过", "解析失败"] 
        
        for pred_text in final_predictions_text:
            pred_label = "解析失败" # 默认值
            try:
                if "{" in pred_text and "}" in pred_text:
                    json_str = pred_text[pred_text.find("{") : pred_text.rfind("}") + 1]
                    data = json.loads(json_str)
                    
                    if "result" in data:
                        pred_label = str(data["result"]).strip()
                else:
                    if "通过" in pred_text:
                        pred_label = "通过"
                    elif "不通过" in pred_text:
                        pred_label = "不通过"
                        
            except Exception as e:
                pass 
                
            predictions.append(pred_label)

        # --- 8. 计算并打印指标 ---
        end_time = time.time()
        # 注意: start_time 是在主进程上记录的，可能不完全准确
        # 但对于总耗时来说足够了
        print(f"\n--- 推理完成 (所有 NPU) ---")
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