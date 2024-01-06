import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Reading the Excel file
file_path = 'Size.xlsx'
data = pd.read_excel(file_path)

# Step 2: Cleaning the data (removing NaN values)
data_cleaned = data.dropna(subset=['Size'])

# Step 3: Applying KMeans clustering
X = data_cleaned[['Size']]
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
data_cleaned['Cluster'] = kmeans.labels_
print(kmeans.cluster_centers_)
# Step 4: Finding the most common cluster
most_common_cluster = data_cleaned['Cluster'].value_counts().idxmax()
# most_common_cluster = 3
print(most_common_cluster)
cluster_center = kmeans.cluster_centers_[most_common_cluster]
print(cluster_center)
# Step 5: Identifying companies in the most common cluster
companies_in_cluster = data_cleaned[data_cleaned['Cluster'] == most_common_cluster]['id'].unique()

# Selecting all records of these companies
all_records_of_selected_companies = data_cleaned[data_cleaned['id'].isin(companies_in_cluster)]

# Define the number of bins for the histogram
bin_count = 20

# Step 6: Plotting the histogram with the identified cluster
plt.figure(figsize=(12, 6))
plt.hist(data_cleaned['Size'], bins=bin_count, color='lightblue', edgecolor='black')
plt.axvline(x=cluster_center, color='red', linestyle='--', label=f'Cluster Center: {cluster_center[0]:.2f}')
plt.title('Frequency Distribution of Company Size with Identified Cluster')
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Calculating and printing the final size range for the exported data
final_min_size = all_records_of_selected_companies['Size'].min()
final_max_size = all_records_of_selected_companies['Size'].max()

print("Final Size Range for Exported Data:")
print(f"Minimum Size: {final_min_size:.2f}")
print(f"Maximum Size: {final_max_size:.2f}")

# Step 7: Exporting the filtered data to a new Excel file
output_file_path = 'Kmeans_Filtered_Size_Data.xlsx'
all_records_of_selected_companies.to_excel(output_file_path, index=False)

output_file_path, all_records_of_selected_companies.shape, all_records_of_selected_companies['id'].nunique()

