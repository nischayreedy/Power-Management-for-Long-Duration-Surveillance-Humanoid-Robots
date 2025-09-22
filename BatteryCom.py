
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('/content/Battery_Comparison_Dataset.csv')


display(df.head())
df.fillna(method='ffill', inplace=True)
plt.figure(figsize=(12, 6))
sns.histplot(df['Energy_Density_Wh_kg'], bins=20, kde=True)
plt.title('Energy Density Distribution')
plt.show()


plt.figure(figsize=(10, 6))
numerical_df = df.select_dtypes(include=np.number)
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


df_plot = df.set_index('Battery_Type')
plt.figure(figsize=(12, 6))
df_plot[['Efficiency_%']].plot(kind='area', alpha=0.6, linewidth=2, color='mediumseagreen')
plt.title('Battery Efficiency Comparison', fontsize=14)
plt.xlabel('Battery Type')
plt.ylabel('Efficiency (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend(title='Efficiency')
plt.show()



plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Efficiency_%', hue='Battery_Type', fill=True, alpha=0.5)
plt.title('KDE Area Chart: Battery Efficiency Distribution')
plt.xlabel('Efficiency (%)')
plt.ylabel('Density')
plt.grid(True)
plt.yticks([])
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Capacity_Ah', hue='Battery_Type', fill=True, alpha=0.5)
plt.title('KDE Area Chart: Battery Capacity Distribution')
plt.xlabel('Capacity (Ah)')
plt.ylabel('Density')
plt.grid(True)
plt.yticks([])
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Weight_kg', hue='Battery_Type', fill=True, alpha=0.5)
plt.title('KDE Area Chart: Battery Weight Distribution')
plt.xlabel('Weight (kg)')
plt.ylabel('Density')
plt.grid(True)
plt.yticks([])
plt.tight_layout()
plt.show()



