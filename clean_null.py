import pandas as pd

# 加载Excel文件
file_path = 'data_final.xlsx'  # 文件路径
data = pd.read_excel(file_path)

# 保留'资不抵债'为1的行，即使其他列有空值；对于'资不抵债'不为1的行，如果有空值则删除
cleaned_data = data[(data['资不抵债'] == 1) | (data.notna().all(axis=1))]
# 找出存在空值的列
columns_with_null = cleaned_data.columns[cleaned_data.isnull().any()].tolist()

# 打印存在空值的列名
print("存在空值的列名：", columns_with_null)
# # 将清洗后的数据导出到新的Excel文件
# output_file = 'cleaned_data_final_ykx1.xlsx'  # 保存的路径
# cleaned_data.to_excel(output_file, index=False)
