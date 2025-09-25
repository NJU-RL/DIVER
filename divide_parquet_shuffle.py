import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd

# 读取全部数据
table = pq.read_table('dataset/valid.mmlu_pro.parquet')
total_rows = table.num_rows
print(f"原始文件总行数: {total_rows}")

# 转换为pandas DataFrame以便进行shuffle
df = table.to_pandas()

# 打乱数据
df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
print("数据已打乱")

# 将pandas DataFrame转回PyArrow Table
table_shuffled = pa.Table.from_pandas(df_shuffled)

# 计算每个分片应有的行数
rows_per_chunk = total_rows // 20
remainder = total_rows % 20

# 创建20个写入器
writers = [
    pq.ParquetWriter(f'dataset/mmlu_shuffle/mmlu_pro_{i}.parquet', schema=table.schema)
    for i in range(20)
]

# 分割并写入数据
start_idx = 0
for i in range(20):
    # 最后一个分片可能会多一些行
    chunk_size = rows_per_chunk + (1 if i < remainder else 0)
    end_idx = start_idx + chunk_size
    
    # 切片获取当前分片的数据
    chunk_table = table_shuffled.slice(start_idx, chunk_size)
    writers[i].write_table(chunk_table)
    
    print(f"分片 {i}: 行数 = {chunk_table.num_rows} (从 {start_idx} 到 {end_idx-1})")
    start_idx = end_idx

# 关闭所有写入器
for writer in writers:
    writer.close()

# 验证分割结果
for i in range(20):
    file_path = f'dataset/mmlu_shuffle/mmlu_pro_{i}.parquet'
    table = pq.read_table(file_path)
    print(f"验证 {file_path}: {table.num_rows} 行")