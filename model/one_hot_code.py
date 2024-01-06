import pandas as pd

# 加载Excel文件
file_path = 'data.xlsx'  # 用您的文件路径替换这里
data = pd.read_excel(file_path)

# 对'Industry2'和'市场类型'应用一热编码
one_hot_encoded_data = pd.get_dummies(data, columns=['Industry2', '市场类型'])

# 将一热编码后的数据导出到新的Excel文件
output_file = 'one_hot_encoded_data.xlsx'  # 用您希望保存的路径替换这里
one_hot_encoded_data.to_excel(output_file, index=False)
