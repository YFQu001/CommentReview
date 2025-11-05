import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重并保存完整模型")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen1.5-1.8B-Chat", help="基础模型 ID")
    parser.add_argument("--adapter_path", type=str, default="./chat_sft_results/final_model", help="LoRA 适配器路径")
    parser.add_argument("--output_path", type=str, default="./merged_model", help="合并后模型的保存路径")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_path, exist_ok=True)

    print(f"--- 正在加载基础模型: {args.model_id} ---")
    
    # 1. 加载基础模型 (强制在 CPU 上加载完整权重)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,  # 保持 bfloat16
        trust_remote_code=True,
        low_cpu_mem_usage=False     # (!!!) 关键：强制在 CPU 上加载
    )

    print(f"--- 正在加载 Tokenizer: {args.model_id} ---")
    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"--- 正在加载 LoRA 适配器: {args.adapter_path} ---")
    # 3. 加载 PeftModel
    # (此时 base_model 在 CPU 上是完整的, PeftModel 也会在 CPU 上加载)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("--- 正在合并 LoRA 权重 (merge_and_unload) ---")
    # 4. 在 CPU 上执行合并 (这里不会再报错)
    model = model.merge_and_unload()
    print("--- 合并完成 ---")

    print(f"--- 正在将合并后的模型保存到: {args.output_path} ---")
    # 5. 保存完整模型和 Tokenizer
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print("--- 所有操作完成 ---")

if __name__ == "__main__":
    main()