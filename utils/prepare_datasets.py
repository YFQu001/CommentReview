import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_datasets(
    df: pd.DataFrame, 
    object_col: str, 
    status_col: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> dict:
    """
    为两种审核模型训练策略准备数据集。

    该函数执行以下操作：
    1.  创建 "对象-状态" 联合分层键。
    2.  (关键优化) 找出 N=1 的孤立组，将其分层键 "降级" 为仅按 "状态" 分层。
    3.  N>=2 的组保持其 "对象-状态" 联合分层键。
    4.  使用这个 "混合键" 进行 80/20 分层抽样，确保划分 100% 成功。
    5.  准备实验A (不平衡) 和 实验B (1:1 欠采样) 的训练集。

    参数:
    df (pd.DataFrame): 包含所有数据的原始DataFrame。
    object_col (str): “评论对象名称”的列名。
    status_col (str): “审核状态”的列名 (标签列)。
    test_size (float): 划分为验证集的比例 (例如 0.2 表示 20%)。
    random_state (int): 随机种子，确保结果可复现。

    返回:
    dict: 包含所有数据集和元数据的字典。
    """
    
    print("--- 开始数据处理 (v2 - 优化分层策略) ---")
    
    # 1. 准备数据和分层键
    df_processed = df.copy()
    # 填充NA，防止分层键出错
    df_processed[object_col] = df_processed[object_col].fillna("NA")
    df_processed[status_col] = df_processed[status_col].fillna("NA")
    
    # 1a. 创建联合分层键 (例如 "商品A_不通过")
    df_processed['stratify_key'] = df_processed[object_col].astype(str) + '_' + df_processed[status_col].astype(str)
    
    # 1b. (关键优化) 找出样本数 < 2 的 "孤立" 键
    key_counts = df_processed['stratify_key'].value_counts()
    # 找出所有样本数=1的键
    single_keys = key_counts[key_counts < 2].index
    
    num_single_keys = len(single_keys)
    if num_single_keys > 0:
        print(f"警告：检测到 {num_single_keys} 个 '对象-状态' 组合的样本数仅为1。")
        
        # 1c. 获取这些 "孤立" 样本的行索引
        single_rows_mask = df_processed['stratify_key'].isin(single_keys)
        
        # 1d. 将这些 "孤立" 样本的分层键“降级”为仅按 '审核状态' 分层
        #     这样，它们会被放入 '通过' 或 '不通过' 的大桶中进行80/20切分
        #     而 N>=2 的组将保持其 "对象-状态" 的精确分层
        df_processed.loc[single_rows_mask, 'stratify_key'] = df_processed.loc[single_rows_mask, status_col]
        
        print(f"已将这 {num_single_keys} 个孤立样本的分层键降级为按 '{status_col}' 分层。")
        print("N >= 2 的组合将继续使用 '对象-状态' 进行精确分层。")
    else:
        print("所有 '对象-状态' 组合 N >= 2，将进行完全的联合分层。")

    # 1e. 此时 'stratify_key' 列是一个可以安全用于划分的 "混合键"
    stratify_target = df_processed['stratify_key']

    # 2. 划分 80% 训练池 (train_pool) 和 20% 验证集 (valid_set)
    train_pool_df, valid_df = train_test_split(
        df_processed,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target # 使用优化后的混合键
    )
    
    print(f"总数据量: {len(df_processed)}")
    print(f"训练池 (Train Pool) 大小: {len(train_pool_df)} ({(1-test_size)*100}%)")
    print(f"验证集 (Validation Set) 大小: {len(valid_df)} ({test_size*100}%)")

    # 3. 准备实验A (1:4.4 不平衡训练集 + 类别权重)
    train_set_unbalanced = train_pool_df.copy()
    
    # 计算类别权重
    class_counts = train_set_unbalanced[status_col].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    weight_for_minority = class_counts[majority_class] / class_counts[minority_class]
    
    # 这个字典 {标签: 权重} 将在模型训练时传入
    class_weights_dict = {
        minority_class: weight_for_minority,
        majority_class: 1.0
    }
    print(f"不平衡训练集大小: {len(train_set_unbalanced)}")
    print(f"  少数类: '{minority_class}' (数量: {class_counts[minority_class]})")
    print(f"  多数类: '{majority_class}' (数量: {class_counts[majority_class]})")
    print(f"  计算出的类别权重 (Class Weights): {class_weights_dict}")

    # 4. 准备实验B (1:1 平衡训练集)
    print("\n步骤 4: 准备实验 B (1:1 平衡训练)")
    
    # (确保使用 train_pool_df 的 class_counts)
    neg_samples = train_pool_df[train_pool_df[status_col] == minority_class]
    pos_samples = train_pool_df[train_pool_df[status_col] == majority_class]
    
    num_neg = len(neg_samples)
    
    # 从多数类中进行欠采样 (Undersampling)
    pos_samples_undersampled = pos_samples.sample(
        n=num_neg, 
        random_state=random_state
    )
    
    # 合并并打乱
    train_set_balanced = pd.concat([neg_samples, pos_samples_undersampled])
    train_set_balanced = train_set_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"1:1 平衡训练集大小: {len(train_set_balanced)}")
    print(f"  '{minority_class}' (少数类) 样本: {num_neg}")
    print(f"  '{majority_class}' (多数类) 样本: {len(pos_samples_undersampled)}")

    # 5. 返回所有结果
    print("\n--- 数据处理完成 ---")
    
    # 清理掉临时工列
    # (注意：要从原始的 df_processed 拷贝的 df 上删除，而不是在原始 df_processed 上)
    if 'stratify_key' in valid_df.columns:
        valid_df = valid_df.drop(columns=['stratify_key'])
    if 'stratify_key' in train_set_unbalanced.columns:
        train_set_unbalanced = train_set_unbalanced.drop(columns=['stratify_key'])
    if 'stratify_key' in train_set_balanced.columns:
        train_set_balanced = train_set_balanced.drop(columns=['stratify_key'])

    results = {
        "valid_set": valid_df,
        "experiment_A": {
            "train_set": train_set_unbalanced,
            "class_weights": class_weights_dict,
            "info": "使用80%数据池 (不平衡), 训练时请使用 class_weights"
        },
        "experiment_B": {
            "train_set": train_set_balanced,
            "class_weights": None, # 1:1数据不需要类别权重
            "info": "使用 1:1 欠采样数据 (来自80%数据池), 训练时不需要 class_weights"
        }
    }
        
    return results


# # --- 1. 模拟你的数据 (请替换成你真实的数据加载) ---
# # 假设你的数据有 35239 条 '通过' 和 7990 条 '不通过'
# total_records = 43229
# pass_records = 35239
# fail_records = 7990

# data = {
#     '评论对象名称': ['对象A', '对象B', '对象C'] * (total_records // 3) + ['对象A'] * (total_records % 3),
#     '评论内容': ['评论内容示例...'] * total_records,
#     '审核状态': ['通过'] * pass_records + ['不通过'] * fail_records
# }
# all_data_df = pd.DataFrame(data)
# # 打乱模拟数据
# all_data_df = all_data_df.sample(frac=1, random_state=42).reset_index(drop=True)

# print(f"原始数据 '审核状态' 分布:\n{all_data_df['审核状态'].value_counts(normalize=True)}\n")


# # --- 2. 调用函数 ---
# # 替换成你真实的列名
# data_splits = prepare_datasets(
#     df=all_data_df,
#     object_col='评论对象名称',
#     status_col='审核状态',
#     test_size=0.2, # 20% 作为验证集
#     random_state=42
# )

# # --- 3. 获取你的数据 ---

# # 用于最终验证的、保持原始分布的 20% 验证集
# validation_data = data_splits['valid_set']

# # 实验A (不平衡数据 + 类别权重)
# train_data_A = data_splits['experiment_A']['train_set']
# class_weights_A = data_splits['experiment_A']['class_weights'] 
# # 在训练时 (如 PyTorch, Keras, XGBoost) 传入 class_weights_A

# # 实验B (1:1 平衡数据)
# train_data_B = data_splits['experiment_B']['train_set']
# # 训练时不需要 class_weights

# # --- 4. 检查结果 ---
# print("\n--- 最终数据集检查 ---")
# print(f"验证集大小: {len(validation_data)}")
# print(f"验证集 '审核状态' 分布:\n{validation_data['审核状态'].value_counts(normalize=True)}\n")

# print(f"实验A 训练集大小: {len(train_data_A)}")
# print(f"实验A '审核状态' 分布:\n{train_data_A['审核状态'].value_counts(normalize=True)}\n")

# print(f"实验B 训练集大小: {len(train_data_B)}")
# print(f"实验B '审核状态' 分布:\n{train_data_B['审核状态'].value_counts(normalize=True)}\n")