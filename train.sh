#!/bin/bash
set -ex


# --- 定义可配置的变量 ---
NUM_GPUS=4 # (重要) 设置你希望使用的 GPU 数量
MASTER_PORT=$(shuf -n 1 -i 10000-65535) # 自动选择一个空闲端口

# -- 模型和数据 --
MODEL_ID="/public/home/yuyan/penglei2/model/Qwen/Qwen2___5-0___5B-Instruct" 
# DATA_PATH="/path/to/your/review_data.csv" 
DATA_PATH="SIMULATE"

# -- 训练超参数 --
OUTPUT_DIR="/public/home/yuyan/penglei2/CommentReview/output" 
EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=2 
GRAD_ACCUM=8
LR=2e-4
WARMUP_RATIO=0.03


# -- LoRA 参数 --
LORA_R=16
LORA_ALPHA=32

# -- 其他设置 --
MAX_SEQ_LEN=2048
SAVE_STEPS=500
EVAL_STEPS=500
TRAIN_SET_KEY="B" # 使用平衡的数据集 'B'

# --- (可选) DeepSpeed ---
# 如果你要使用 DeepSpeed (如你的示例所示), 你需要：
# 1. 确保已安装 deepspeed: pip install deepspeed
# 2. 创建一个 deepspeed.json 配置文件 (例如 deepspeed_config.json)
# 3. (重要) 修改你的 train_sft.py, 在 parse_args() 中添加:
#    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed 配置文件路径")
# 4. (重要) 修改 train_sft.py 的 main() 函数中, 传递 deepspeed 参数:
#    training_args = TrainingArguments(
#        ...
#        deepspeed=args.deepspeed, # <--- 添加这一行
#        ...
#    )
# 5. 然后在下面的命令末尾取消注释并添加 --deepspeed "deepspeed_config.json" \
# DEEPSPEED_CONFIG="deepspeed_config.json"


# --- 运行 Python 训练脚本 (使用 torchrun) ---
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train.py \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size  16 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --max_seq_length $MAX_SEQ_LEN \
    --save_steps $SAVE_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "cosine" \
    --do_eval True \
    --eval_strategy "steps" \
    --eval_steps $EVAL_STEPS \
    --train_dataset_key $TRAIN_SET_KEY \
    --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj" "score" \
    --seed 42
    # --deepspeed "$DEEPSPEED_CONFIG" # <--- 如果使用 DeepSpeed，取消此行注释

echo "--- 训练完成 ---"

