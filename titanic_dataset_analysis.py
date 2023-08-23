import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pprint import pprint
# Load the Titanic dataset using pandas
df = pd.read_csv('/Users/srepala/PycharmProjects/datascience/datasets/data_analysis/titanic/train.csv')
print(df.columns)


# Data Exploration and Descriptive Statistics
print("Data Exploration and Descriptive Statistics:")
print(df.info())
print(df.describe())
print(df.head())

# Data Analysis - Survival Statistics
survived_count = df['Survived'].sum()
total_passengers = len(df)
survival_rate = survived_count / total_passengers * 100
print(f"\nSurvival Rate: {survival_rate:.2f}%")
print(f"Total Passengers: {total_passengers}")
print(f"Survived: {survived_count}")
print(f"Not Survived: {total_passengers - survived_count}")

# Data Analysis - Passengers by Class
passengers_by_class = df['Pclass'].value_counts()
print("\nPassengers by Class:")
print(passengers_by_class)

# Data Analysis - Passengers by Age Group
age_groups = pd.cut(df['Age'], bins=[0, 18, 30, 50, 100])
passengers_by_age_group = df.groupby(age_groups)['Age'].count()
print("\nPassengers by Age Group:")
print(passengers_by_age_group)

# Data Analysis - Passengers by Embarkation Port
passengers_by_embarkation_port = df['Embarked'].value_counts()
print("\nPassengers by Embarkation Port:")
print(passengers_by_embarkation_port)

# Data Visualization - Survival Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Survived', palette='pastel')
plt.title('Survival Distribution')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Data Visualization - Passengers by Class
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Pclass', palette='pastel')
plt.title('Passengers by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# Data Visualization - Passengers by Age Group
plt.figure(figsize=(8, 6))
passengers_by_age_group.plot(kind='bar', color='lightblue', edgecolor='black')
plt.title('Passengers by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

# Data Visualization - Passengers by Embarkation Port
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Embarked', palette='pastel')
plt.title('Passengers by Embarkation Port')
plt.xlabel('Embarkation Port')
plt.ylabel('Count')
plt.show()

# Statistical Analysis
# Example: Pearson correlation coefficient between age and fare
correlation_coefficient, _ = stats.pearsonr(df['Age'], df['Fare'])
print(f"\nPearson correlation coefficient between Age and Fare: {correlation_coefficient:.2f}")
