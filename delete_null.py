import pandas as pd

# 加载Excel文件
file_path = 'cleaned_data_final_ykx1.xlsx'  # 文件路径
data = pd.read_excel(file_path)

# 删除存在空值的行
cleaned_data = data.dropna()

# 将清洗后的数据导出到新的Excel文件
output_file = 'data_all.xlsx'  # 保存的路径
cleaned_data.to_excel(output_file, index=False)
