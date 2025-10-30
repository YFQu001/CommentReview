#!/bin/bash
#SBATCH --job-name=Comment
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32         # 请求 CPU 核心数（可调整）
#SBATCH -o logs/%j.log                    # 标准输出日志，%j 会被替换为任务ID
#SBATCH -e logs/%j.err                    # 错误输出日志

mkdir -p logs
# 加载 conda 环境（如果使用 bash，调整为你的 shell 配置）
source ~/.bashrc
# 激活你的环境
conda activate qwen_sft
cd /public/home/yuyan/penglei2/CommentReview/code
bash /public/home/yuyan/penglei2/CommentReview/code/train.sh