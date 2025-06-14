import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

import os, psutil
import gc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load and sample the dataset (reduce size for dev)
try:
    df = pd.read_csv("data/Hackathon_bureau_data_50000.csv", encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv("data/Hackathon_bureau_data_50000.csv", encoding='latin1')


SAMPLE_SIZE = 20000
df = df.sample(SAMPLE_SIZE, random_state=42)

mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
print(f"📊 Sampled {SAMPLE_SIZE} rows using ~{mem:.2f} MB RAM")

available_mem = psutil.virtual_memory().available / (1024 ** 2)
if mem > 0.3 * available_mem:
    raise MemoryError(f"❌ Not enough memory to process {SAMPLE_SIZE} rows safely.")

df = df.dropna(subset=['target_income'])
df = df.drop_duplicates()
df = df.apply(lambda x: x.str.strip().str.lower() if x.dtype == 'object' else x)
gc.collect()

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(8, 4))
sns.histplot(df['target_income'], kde=True)
plt.title('Distribution of Target Income')
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()

for col in ['age', 'income', 'credit_score']:
    if col in df.columns:
        sns.scatterplot(x=col, y='target_income', data=df)
        plt.title(f'{col} vs Target Income')
        plt.show()

sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

# Feature reduction based on correlation with target
corr_matrix = df.corr(numeric_only=True)
target_corr = corr_matrix['target_income'].abs().sort_values(ascending=False)
print("\n🔍 Top correlated features with target:")
print(target_corr.head(10))

# Drop features with low correlation (< 0.01)
low_corr_cols = target_corr[target_corr < 0.01].index.tolist()
df.drop(columns=low_corr_cols, inplace=True)

# %%
def reduce_cardinality(df, threshold=0.01):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        freqs = df[col].value_counts(normalize=True)
        rare_labels = freqs[freqs < threshold].index
        df[col] = df[col].apply(lambda x: 'other' if x in rare_labels else x)
    return df

from itertools import combinations

def engineer_features(df):
    df = df.copy()
    important_numeric = ['age', 'income', 'credit_score']
    for col in important_numeric:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2

    combos = list(combinations(important_numeric, 2))
    for col1, col2 in combos:
        if col1 in df.columns and col2 in df.columns:
            df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]

    return df

X = df.drop(columns=['target_income'])
y = df['target_income']  # Removed log1p transformation
X = reduce_cardinality(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = engineer_features(X_train)
X_test = engineer_features(X_test)

categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
numeric_cols = X_train.select_dtypes(include='number').columns.tolist()

from sklearn.impute import SimpleImputer

# Impute numeric columns
imputer_num = SimpleImputer(strategy='median')
X_train[numeric_cols] = imputer_num.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = imputer_num.transform(X_test[numeric_cols])

# Impute categorical columns
imputer_cat = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = imputer_cat.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = imputer_cat.transform(X_test[categorical_cols])

from sklearn.feature_selection import SelectKBest, mutual_info_regression
selector = SelectKBest(score_func=mutual_info_regression, k='all')
selector.fit(X_train[numeric_cols], y_train)
feature_scores = pd.Series(selector.scores_, index=numeric_cols).sort_values(ascending=False)

# Visualize top features
feature_scores.head(30).plot(kind='barh', figsize=(10, 8), title='Top 30 Features')
plt.gca().invert_yaxis()
plt.show()

# Keep only top 30 features
top_n = 30
selected_numeric_cols = feature_scores.head(top_n).index.tolist()

X_train = X_train[selected_numeric_cols + categorical_cols]
X_test = X_test[selected_numeric_cols + categorical_cols]

# %%
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import time

X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=18,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

start = time.time()
model.fit(X_train_final, y_train_final)
print("✅ Model training completed in", time.time() - start, "seconds")

# Evaluate on test set
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae}")
print(f"R2: {r2}")
print(f"RMSE: {rmse}")
print("✅ Training complete.")
