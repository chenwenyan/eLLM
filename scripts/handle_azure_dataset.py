import pandas as pd
import io


input_path = '/nfs/dataset/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv'
output_path = '/nfs/dataset/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week_milliseconds.csv'
output_path = '/nfs/dataset/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week_count.csv'
# df = pd.read_csv(input_path)

# df['TIMESTAMP'] = pd.to_datetime(
#         df['TIMESTAMP'],
#         errors='coerce',      # 解析失败的置为 NaT
#         utc=True              # 直接转换为 UTC
# )

# # 3. 计算相邻记录之间的毫秒差
# df['Delta_ms'] = df['TIMESTAMP'].diff().dt.total_seconds() * 1000

# print(df.head())

# df.to_csv(output_path, index=False)

df = pd.read_csv(
    input_path,
    parse_dates=['TIMESTAMP'],      # 或 0，用列号也行
    date_parser=lambda x: pd.to_datetime(x, errors='coerce', utc=True)
)

# 把 Timestamp 向下取整到秒
df['Timestamp_sec'] = df.iloc[:, 0].dt.floor('S')

# 按秒分组计数
result = (
    df.groupby('Timestamp_sec')
      .size()
      .reset_index(name='Count')
)

# 保存
result.to_csv(output_path, index=False)