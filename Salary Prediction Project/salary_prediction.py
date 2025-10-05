# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load Dataset
df = pd.read_csv("your_dataset.csv")

# 3. Basic Exploration
print(df.head())
print(df.info())
print(df.describe())
print(df['Salary'].describe())

# Drop index column if needed
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# 4. Check Missing Values
print(df.isnull().sum())

# Optional: Drop or fill missing
# df = df.dropna()  # or use imputation for numerical/categorical columns

# 5. Data Visualization
plt.figure(figsize=(10,6))
sns.histplot(df['Salary'], kde=True)
plt.title('Salary Distribution')
plt.show()

# Salary by Country
plt.figure(figsize=(12,6))
sns.boxplot(x='Country', y='Salary', data=df)
plt.xticks(rotation=90)
plt.title('Salary by Country')
plt.show()

# Salary by Race
plt.figure(figsize=(10,6))
sns.boxplot(x='Race', y='Salary', data=df)
plt.xticks(rotation=90)
plt.title('Salary by Race')
plt.show()

# 6. Feature Selection
X = df.drop('Salary', axis=1)
y = df['Salary']

# 7. Identify Numerical and Categorical Columns
numeric_features = ['Age', 'Years of Experience']
categorical_features = ['Education Level', 'Job Title', 'Country', 'Race']

# 8. Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 9. Create Regression Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 10. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 11. Train the Model
model.fit(X_train, y_train)

# 12. Predict and Evaluate
y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Optional: Try Linear Regression for comparison
linear_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)

print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
