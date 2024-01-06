import pandas as pd

# 加载Excel文件
file_path = 'one_hot_encoded_data.xlsx'  # 文件路径，请替换为你的文件路径
data = pd.read_excel(file_path)

# 指定需要替换的列名列表
columns_to_replace = [
    'Industry2_C13', 'Industry2_C14', 'Industry2_C15', 'Industry2_C17', 'Industry2_C18',
    'Industry2_C19', 'Industry2_C20', 'Industry2_C21', 'Industry2_C22', 'Industry2_C23',
    'Industry2_C24', 'Industry2_C25', 'Industry2_C26', 'Industry2_C27', 'Industry2_C28',
    'Industry2_C29', 'Industry2_C30', 'Industry2_C31', 'Industry2_C32', 'Industry2_C33',
    'Industry2_C34', 'Industry2_C35', 'Industry2_C36', 'Industry2_C37', 'Industry2_C38',
    'Industry2_C39', 'Industry2_C40', 'Industry2_C41', 'Industry2_C42', 'Industry2_C43',
    '市场类型_上证A股', '市场类型_深证A股'
]

# 替换指定列中的FALSE为0，TRUE为1
data[columns_to_replace] = data[columns_to_replace].replace({False: 0, True: 1})

# 将修改后的数据导出到新的Excel文件
output_file = 'modified_data.xlsx'  # 保存的路径，请替换为你希望保存的路径
data.to_excel(output_file, index=False)
