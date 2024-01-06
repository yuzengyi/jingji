import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read and clean the data
data = pd.read_excel('Size.xlsx')
data_cleaned = data.dropna(subset=['Size'])

# Generate a range of potential intervals with different interval_step values
min_size = data_cleaned['Size'].min()
max_size = data_cleaned['Size'].max()
best_interval = None

# Define interval_step_values from 5 to 10 with a step of 0.1
interval_step_values = np.arange(10.1, 15.1, 0.1)

for interval_step in interval_step_values:
    print(interval_step)
    potential_intervals = [(start, start + interval_step) for start in np.arange(min_size, max_size, interval_step)]

    # Function to evaluate each interval
    def evaluate_interval(interval, data):
        min_val, max_val = interval
        # Filter companies within the interval
        companies_in_interval = data[(data['Size'] >= min_val) & (data['Size'] <= max_val)]['id'].unique()
        # Filter records of these companies
        all_records = data[data['id'].isin(companies_in_interval)]
        # Check if all years are present for these companies
        complete_records = all([len(data[data['id'] == company]['year'].unique()) == len(data['year'].unique()) for company in companies_in_interval])
        return (max_val - min_val, len(companies_in_interval), complete_records, interval)

    # Evaluate each interval
    evaluated_intervals = [evaluate_interval(interval, data_cleaned) for interval in potential_intervals]

    # Filter intervals that meet the completeness criterion
    valid_intervals = [interval for interval in evaluated_intervals if interval[2]]

    # Sort intervals by number of companies covered and then by interval width
    sorted_valid_intervals = sorted(valid_intervals, key=lambda x: (-x[1], x[0]))

    # Choose the best interval based on the criteria (if any valid interval exists)
    if sorted_valid_intervals:
        best_interval = sorted_valid_intervals[0]
        break  # Stop searching if a valid interval is found

# Print and save the best interval
if best_interval:
    print(f"Best Interval: {best_interval[3]}")
    print(f"Interval Width: {best_interval[0]}")
    print(f"Number of Companies Covered: {best_interval[1]}")
    min_val, max_val = best_interval[3]
    filtered_data = data_cleaned[(data_cleaned['Size'] >= min_val) & (data_cleaned['Size'] <= max_val)]
    output_file_path = 'Filtered_Size_Data.xlsx'
    filtered_data.to_excel(output_file_path, index=False)
    print(f"Filtered data saved to {output_file_path}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(data_cleaned['Size'], bins=50, color='lightblue', edgecolor='black')
    plt.axvline(x=best_interval[3][0], color='green', linestyle='--', label='Interval Start')
    plt.axvline(x=best_interval[3][1], color='red', linestyle='--', label='Interval End')
    plt.title('Size Distribution with Best Interval')
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
else:
    print("No valid interval found that meets all criteria.")
