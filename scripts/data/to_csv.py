import re
import pandas as pd

# 模拟数据
raw_inputs = {
    "checkpoint-100": """
                   +    -
student=teacher: 20258 23180
teacher=ground : 19406 19295
student=ground : 17693 18785
     Total     : 24669 27642
    """,
    "checkpoint-200": """
                   +    -
student=teacher: 20204 23694
teacher=ground : 19087 19614
student=ground : 17549 19177
     Total     : 24398 27913
    """,
    "checkpoint-300": """
                   +    -
student=teacher: 20869 23334
teacher=ground : 19550 19151
student=ground : 18119 18747
     Total     : 24903 27408
    """,
    "checkpoint-400": """
                   +    -
student=teacher: 21395 22959
teacher=ground : 19986 18715
student=ground : 18576 18340
     Total     : 25385 26926
    """,
    "checkpoint-500": """
                   +    -
student=teacher: 21913 22539
teacher=ground : 20402 18299
student=ground : 19038 17934
     Total     : 25881 26430
    """,
    "checkpoint-600": """
                   +    -
student=teacher: 22131 22445
teacher=ground : 20519 18182
student=ground : 19180 17828
     Total     : 26074 26237
    """,
    "checkpoint-700": """
                   +    -
student=teacher: 22592 22061
teacher=ground : 20898 17803
student=ground : 19592 17439
     Total     : 26461 25850
    """,
    "checkpoint-800": """
                   +    -
student=teacher: 22736 21932
teacher=ground : 21036 17665
student=ground : 19757 17297
     Total     : 26579 25732
    """,
    "checkpoint-900": """
                   +    -
student=teacher: 23355 21321
teacher=ground : 21575 17126
student=ground : 20270 16771
     Total     : 27312 24999
    """,
    "checkpoint-1000": """
                   +    -
student=teacher: 23745 20948
teacher=ground : 21958 16743
student=ground : 20658 16385
     Total     : 27700 24611
    """,
    "checkpoint-1100": """
                   +    -
student=teacher: 24145 20573
teacher=ground : 22282 16419
student=ground : 21003 16065
     Total     : 28094 24217
    """,
    "checkpoint-1200": """
                   +    -
student=teacher: 24409 20329
teacher=ground : 22508 16193
student=ground : 21204 15853
     Total     : 28391 23920
    """,
}
final_data = []

for key, text_content in raw_inputs.items():
    # 1. 提取数字 Checkpoint
    ckpt_match = re.search(r'(\d+)', key)
    ckpt_step = int(ckpt_match.group(1)) if ckpt_match else 0
    
    # 临时列表，用于存储当前 Checkpoint 的所有行数据
    current_block_data = []
    
    # 变量用于记录该 block 的 total 值
    block_total_pos = 0
    block_total_neg = 0
    
    # --- 第一遍循环：解析数据并找到 Total ---
    lines = text_content.strip().split('\n')
    for line in lines:
        match = re.search(r"(.*?)\s*:\s*(\d+)\s+(\d+)", line)
        if match:
            label = match.group(1).strip()
            pos_val = int(match.group(2))
            neg_val = int(match.group(3))
            
            # 记录 Total 的值，用于后续做分母
            if label == 'Total':
                block_total_pos = pos_val
                block_total_neg = neg_val
            
            # 暂存每一行的数据
            current_block_data.append({
                "Checkpoint": ckpt_step,
                "Relation": label,
                "Positive": pos_val,
                "Negative": neg_val
            })
            
    # --- 第二遍循环：计算比例并添加到最终列表 ---
    for row in current_block_data:
        # 防止除以 0 的保护
        if block_total_pos > 0:
            row['Pos_Share'] = row['Positive'] / block_total_pos
        else:
            row['Pos_Share'] = 0.0
            
        if block_total_neg > 0:
            row['Neg_Share'] = row['Negative'] / block_total_neg
        else:
            row['Neg_Share'] = 0.0
            
        final_data.append(row)

# 生成 DataFrame
df = pd.DataFrame(final_data)

# 排序：按 Checkpoint 升序，Total 放在每组的最后（可选）
# 这里的逻辑是：先按 Checkpoint 排，再按 Relation 排
df = df.sort_values(by=["Checkpoint", "Relation"])

# print(df)
df.to_csv("analysis_with_ratios.csv", index=False)