import pandas as pd

# 加载Excel文件
file_path = 'y_null.xlsx'  # 文件路径
data = pd.read_excel(file_path)

# 计算每列缺失的数量
missing_counts = data.isnull().sum()

# 按从高到低的顺序排序
missing_counts_sorted = missing_counts.sort_values(ascending=False)

# 创建一个包含缺失数量信息的新DataFrame
missing_counts_df = pd.DataFrame({'Column_Name': missing_counts_sorted.index, 'Missing_Count': missing_counts_sorted.values})

# 将新的DataFrame导出到Excel文件
output_file = 'missing_counts_sorted.xlsx'  # 导出的文件路径
missing_counts_df.to_excel(output_file, index=False)
