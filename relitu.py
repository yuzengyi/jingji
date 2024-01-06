import pandas as pd

# Load the Excel file
file_path = 'relitu.xlsx'
data = pd.read_excel(file_path)

# Revised code to calculate the proportion of '资不抵债' == 1 as a percentage of all records

# Calculating the proportion of '资不抵债' == 1 for each industry and year
# Group by year and industry, then calculate the sum of '资不抵债'
sum_data = data.groupby(['year', 'IndustryName'])['资不抵债'].sum().reset_index()
print(sum_data)
# Calculating the total number of records for each industry and year
count_data = data.groupby(['year', 'IndustryName'])['资不抵债'].count().reset_index()
print(count_data)
# Merging sum and count data
merged_data = pd.merge(sum_data, count_data, on=['year', 'IndustryName'], suffixes=('_sum', '_count'))

# Calculating the proportion as a percentage
merged_data['proportion'] = (merged_data['资不抵债_sum'] / merged_data['资不抵债_count']) * 100

# Pivot the data to get years as rows and industries as columns
pivot_data = merged_data.pivot(index='year', columns='IndustryName', values='proportion')

# Filling NaN values with 0, assuming NaN means no data which can be considered as 0 proportion
pivot_data = pivot_data.fillna(0)

# Saving the pivot table to an Excel file
# output_file = 'industry_insolvency_proportions_percentage.xlsx'
# pivot_data.to_excel(output_file)

# pivot_data.head(), output_file

