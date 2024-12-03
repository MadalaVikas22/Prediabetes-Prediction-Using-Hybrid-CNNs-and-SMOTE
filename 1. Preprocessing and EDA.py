# Exploratory Data Analysis (EDA) and Preprocessing for Diabetes Dataset
# Dataset: https://data.mendeley.com/datasets/wj9rwkp9c2/1

'''
Ensemble Methods: Consider using ensemble methods like Random Forests or XGBoost, which can be robust to class imbalance even without SMOTE.
However, combining these with SMOTE can further enhance performance.
'''

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Set Random Seed for Reproducibility
np.random.seed(42)

# Load Dataset
diabetes_data = pd.read_csv("/content/drive/MyDrive/Exposys/Dataset of Diabetes .csv")

# Initial Dataset Exploration
print("First 6 rows of the dataset:")
print(diabetes_data.head(6))
print("\nDataset shape:", diabetes_data.shape)

# Drop Irrelevant Columns
diabetes_data.drop(['No_Pation', 'ID'], axis=1, inplace=True)

# Dataset Information
print("\nDataset Info:")
print(diabetes_data.info())
print("\nStatistical Summary:")
print(diabetes_data.describe())
print("\nMissing Values Count:")
print(diabetes_data.isnull().sum())

# Distribution of Categorical Variables
print("\nGender Distribution:")
print(diabetes_data['Gender'].value_counts())

# Clean and Process Target Variable
diabetes_data['CLASS'] = diabetes_data['CLASS'].str.strip()
print("\nTarget Variable Distribution:")
print(diabetes_data['CLASS'].value_counts())

# Visualize Target Variable Distribution
plt.figure(figsize=(8, 6))
class_counts = diabetes_data['CLASS'].value_counts()
ax = class_counts.plot(kind='bar', color=['#007acc', '#ff6666', '#66cc99'])

# Annotate Each Bar
for p in ax.patches:
    ax.annotate(str(p.get_height()),
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='center', xytext=(0, 5),
                textcoords='offset points')

# Customize Plot
ax.set_title('Count of Each Class')
ax.set_xticks(range(len(class_counts)))
ax.set_xticklabels(['Diabetic', 'Non-Diabetic', 'Pre-Diabetic'], rotation=0, ha='center')
ax.set_xlabel('')
ax.set_ylabel('')
ax.get_yaxis().set_ticks([])
for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(False)
plt.show()

# Encode Categorical Variables
diabetes_data['Gender'] = diabetes_data['Gender'].replace('f', 'F')
diabetes_data['Gender'] = diabetes_data['Gender'].map({'F': 1, 'M': 0})
diabetes_data['CLASS'] = diabetes_data['CLASS'].map({'Y': 0, 'N': 1, 'P': 2})

# Visualize Numerical Features
diabetes_data.hist(figsize=(12, 10), color='#86bf91', alpha=0.9)
plt.suptitle('Histograms of Numerical Features', fontsize=16)
plt.show()

# Boxplot for Outliers
plt.figure(figsize=(12, 10))
diabetes_data.boxplot(grid=False)
plt.title('Boxplot of Numerical Features')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = diabetes_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Drop Highly Correlated Features (Threshold > 0.8)
high_corr_features = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_features.append(corr_matrix.columns[i])
diabetes_data.drop(high_corr_features, axis=1, inplace=True)

# Final Dataset Shape and Head
print("\nFinal Dataset Shape:", diabetes_data.shape)
print("\nFirst 5 rows of the final dataset:")
print(diabetes_data.head())
