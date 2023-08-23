import pandas as pd
import matplotlib.pyplot as plt
# Sample data for demonstration
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 32, 28, 21, 29],
    'Gender': ['F', 'M', 'M', 'M', 'F'],
    'Salary': [50000, 60000, 55000, 48000, 52000],
    'Hire_Date': pd.to_datetime(['2020-01-15', '2019-07-10', '2022-03-05', '2023-02-20', '2021-11-30']),
    'Is_Employee': [True, True, True, False, True],
    'Ratings': [4.5, 3.8, 4.9, None, 4.2],
    'Department': ['HR', 'Finance', 'Engineering', 'Marketing', 'HR']
}

# Creating a DataFrame
df = pd.DataFrame(data)


# Define different data types of columns
df['Name'] = df['Name'].astype('string')            # String data type (Pandas StringDtype)
df['Age'] = df['Age'].astype('int64')               # Integer data type (int64)
df['Gender'] = df['Gender'].astype('category')      # Categorical data type (Pandas CategoricalDtype)
df['Salary'] = df['Salary'].astype('float64')       # Floating-point data type (float64)
df['Hire_Date'] = pd.to_datetime(df['Hire_Date'])   # Datetime data type (datetime64[ns])
df['Is_Employee'] = df['Is_Employee'].astype('bool')  # Boolean data type (bool)
df['Ratings'] = df['Ratings'].astype('float64')     # Floating-point data type (float64)
df['Department'] = df['Department'].astype('category')  # Categorical data type (Pandas CategoricalDtype)

# Operations with pandas

# 1. Data Exploration
# ---------------------
# Get the number of rows and columns
print("Shape:", df.shape)

# Get basic info about the DataFrame
print("Info:")
print(df.info())

# Get the data types of columns
print("Data Types:")
print(df.dtypes)

# 2. Data Selection
# ---------------------
# Select specific columns
print("Selected Columns:")
print(df[['Name', 'Age', 'Salary']])

# Select rows using loc and iloc
print("Filtered Rows:")
print(df.loc[df['Age'] > 25])
print(df.iloc[1:4])
#
# 3. Data Filtering
# ---------------------
# Filter rows based on multiple conditions (use '&' for 'and', '|' for 'or')
print("Filtered Rows with Multiple Conditions:")
print(df[(df['Age'] > 25) & (df['Gender'] == 'M')])

# 4. Data Sorting
# ---------------------
# Sort DataFrame by column(s)
print("Sorted DataFrame:")
print(df.sort_values(by=['Age', 'Name'], ascending=[True, False]))

# 5. Data Aggregation
# ---------------------
# Get the maximum salary
print("Maximum Salary:", df['Salary'].max())
#
# Get the average age
print("Average Age:", df['Age'].mean())
#
# 6. Data Cleaning
# ---------------------
# Check for missing values
print("Missing Values:")
print(df.isnull().sum())
#
# Fill missing values with a default value
# df['Ratings'].fillna(0, inplace=True)
# print(df)
# # 7. Data Manipulation
# ---------------------
# Add a new column
df['Bonus'] = df['Salary'] * 0.1

# Apply a function to a column
df['Gender'] = df['Gender'].apply(lambda x: 'Male' if x == 'M' else 'Female')

print(df)
# 8. Data Grouping
# ---------------------
# Group data and perform aggregate functions
print("Average Salary by Department:")
print(df.groupby('Department')['Salary'].mean())

#9. Data Joining and Merging
#---------------------
# # #Merge two DataFrames based on a common column
# df1 = df[["Name","Age"]]
# df2 = df[["Name",'Gender']]
# merged_df = pd.merge(df1, df2, on='Name')
# print(merged_df)
# 10. Data Reshaping
# ---------------------
# Melt the DataFrame
# melted_df = pd.melt(df, id_vars='Name', value_vars=['Age', 'Salary'], var_name='Attributes', value_name='Values')
# print(melted_df)
# # 11. Pivot Tables
# # ---------------------
# # Create a pivot table
pivot_table = df.pivot_table(index='Gender', columns='Department', values='Salary', aggfunc='mean')
print(pivot_table)
# 12. Data Visualization
# ---------------------
# import matplotlib.pyplot as plt
# df.plot(kind='bar', x='Name', y='Salary')
# plt.show()
#
# Save the DataFrame back to a CSV file
# df.to_csv('new_file.csv', index=False)
#
# # Print the DataFrame after all operations
# print("\nFinal DataFrame:")
# print(df)
#
# import pandas as pd
#
#
# # More Complex Operations with pandas
#
# # 1. Advanced Filtering
# # ---------------------
# # Filter rows using complex conditions (use '|' for 'or' and '&' for 'and' with parentheses)
print("Complex Filtered Rows:")
print(df[(df['Age'] > 25) & ((df['Gender'] == 'M') | (df['Department'] == 'HR'))])
#
# # 2. Data Aggregation with Multiple Functions
# # ---------------------
# # Get multiple aggregate statistics using `agg`
agg_stats = df.groupby('Department')['Salary'].agg(['mean', 'median', 'min', 'max'])
print("Aggregate Statistics by Department:")
print(agg_stats)
#
# # 3. Handling Datetime Data
# # ---------------------
# # Extract year, month, and day from the Hire_Date column
df['Year_Hired'] = df['Hire_Date'].dt.year
df['Month_Hired'] = df['Hire_Date'].dt.month
df['Day_Hired'] = df['Hire_Date'].dt.day
print(df)
#
# # 4. Data Imputation with Fillna
# # ---------------------
# # Fill missing values with mean of Ratings
# df["new_col"] = df['Ratings'].fillna(df['Ratings'].mean(), inplace=False)
# print(df)
#
# # 5. Data Ranking
# # ---------------------
# # Rank employees based on their Salary
# df['Salary_Rank'] = df['Salary'].rank()
# print(df[["Salary","Salary_Rank"]])
# # 6. String Operations
# # ---------------------
# Extract the first character of each name
# df['First_Character'] = df['Name'].str[0]
# print(df)
#
# # Check if a name contains a specific substring
# df['Has_Substr'] = df['Name'].str.contains('e', case=False)
# print(df)
# # 7. Data Reshaping with Pivot Table
# # ---------------------
# # Create a pivot table with multi-level indexing
# pivot_table_multi_index = df.pivot_table(index=['Department', 'Gender'], columns='Is_Employee', values='Salary', aggfunc='mean')
# print("Multi-Index Pivot Table:")
# print(pivot_table_multi_index)
#
# # 8. Data Concatenation
# # ---------------------
# # Concatenate two DataFrames vertically
# df2 = pd.DataFrame({'Name': ['Frank'], 'Age': [27], 'Gender': ['M'], 'Salary': [62000], 'Department': ['Finance']})
# df_concat = pd.concat([df, df2], ignore_index=True)
# print(df_concat)
#
# # 9. Data Transformation using apply()
# # ---------------------
# # Convert Salary to a string with '$' symbol
df['Salary'] = df['Salary'].apply(lambda x: f"${x}")
print(df)
#
# # 10. Custom Function for Data Transformation
# # ---------------------
# # Define a custom function to categorize ages
def age_category(age):
    if age < 25:
        return 'Young'
    elif age >= 25 and age < 35:
        return 'Middle-aged'
    else:
        return 'Senior'

# Apply the custom function to create a new column
df['Age_Category'] = df['Age'].apply(age_category)
print(df)
#
# # Print the DataFrame after all operations
# print("\nFinal DataFrame:")
# print(df)
#
# # More Complex Operations with pandas (Continued)
#
# # 11. Handling Categorical Data
# # ---------------------
# # Get the unique categories of the 'Department' column
# print("Unique Departments:", df['Department'].unique())
#
# # Convert categorical data to numerical codes
# df['Department_Code'] = df['Department'].cat.codes

# # 12. Handling Boolean Data
# # ---------------------
# # Count the number of employees and non-employees
# print("Employee Counts:")
# print(df['Is_Employee'].value_counts())
#
# # 13. Data Transformation using Map
# # ---------------------
# # Map the 'Gender' column to numerical values (0 for Female, 1 for Male)
# df['Gender_Code'] = df['Gender'].map({'F': 0, 'M': 1})
#
# # 14. Data Grouping with Aggregation
# # ---------------------
# # Group data by 'Gender' and 'Is_Employee' and calculate the mean salary and maximum age in each group
# grouped_data = df.groupby(['Gender', 'Is_Employee']).agg({'Salary': 'mean', 'Age': 'max'})
# print("Grouped Data with Aggregation:")
# print(grouped_data)
#
# # 15. Rolling Window Operations
# # ---------------------
# # Calculate the rolling average of Salary over a window of 2 periods
# df['Rolling_Average_Salary'] = df['Salary'].rolling(window=2).mean()
#
# # 16. Data Shifting
# # ---------------------
# # Shift the 'Age' column one step forward
# df['Age_Shifted'] = df['Age'].shift(-1)
#
# # 17. Data Sampling
# # ---------------------
# # Sample a random subset of data
# sampled_data = df.sample(n=3, random_state=42)
# print("Sampled Data:")
# print(sampled_data)
#
# # 18. Data Reshaping using melt()
# # ---------------------
# # Melt the DataFrame to convert columns into rows
# melted_df = pd.melt(df, id_vars=['Name', 'Age'], value_vars=['Salary', 'Ratings'], var_name='Attributes', value_name='Values')
# print("Melted DataFrame:")
# print(melted_df)
#
# # 19. Custom Data Aggregation with pivot_table()
# # ---------------------
# # Define a custom aggregation function
# def custom_aggregation(x):
#     return x.max() - x.min()
#
# # Create a pivot table with custom aggregation
# pivot_table_custom_agg = df.pivot_table(index='Department', values='Salary', aggfunc=custom_aggregation)
# print("Custom Aggregation with Pivot Table:")
# print(pivot_table_custom_agg)
#
# # Print the DataFrame after all operations
# print("\nFinal DataFrame:")
# print(df)
