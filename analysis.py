import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = 'https://p16-bot-sign-sg.ciciai.com/tos-alisg-i-b2l6bve69y-sg/0a542996a2e74931b9a17cc3fb593feb.csv~tplv-b2l6bve69y-image.image?rk3s=68e6b6b5&x-expires=1719816096&x-signature=hvoff8lKbCLE4lp7Nfk5Cir9yn4%3D'
data = pd.read_csv(url)

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Gender Distribution
plt.figure(figsize=(6, 6))
sns.countplot(x='Sex', data=data)
plt.title('Gender Distribution')
plt.xlabel('Sex (0: Female, 1: Male)')
plt.ylabel('Count')
plt.show()

# Prevalence of High Cholesterol
high_cholesterol_prevalence = data['HighChol'].mean() * 100
print(f'Prevalence of High Cholesterol: {high_cholesterol_prevalence:.2f}%')

# Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Relationship between BMI and Health Conditions
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes', y='BMI', data=data)
plt.title('BMI vs Diabetes')
plt.xlabel('Diabetes (0: No, 1: Yes)')
plt.ylabel('BMI')
plt.show()

# Impact of Physical Activity on General Health
plt.figure(figsize=(10, 6))
sns.boxplot(x='PhysActivity', y='GenHlth', data=data)
plt.title('Physical Activity vs General Health')
plt.xlabel('Physical Activity (0: No, 1: Yes)')
plt.ylabel('General Health (1: Excellent, 5: Poor)')
plt.show()