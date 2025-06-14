import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import joblib
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("data/Hackathon_bureau_data_400.csv")
df = df.dropna(subset=['target_income'])
df = df.drop_duplicates()
df = df.apply(lambda x: x.str.strip().str.lower() if x.dtype == 'object' else x)

# Feature engineering
def engineer_features(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
    important_numeric = ['age', 'income', 'credit_score']
    for col in important_numeric:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2
    return df

# Reduce cardinality
def reduce_cardinality(df, threshold=0.01):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        freqs = df[col].value_counts(normalize=True)
        rare_labels = freqs[freqs < threshold].index
        df[col] = df[col].apply(lambda x: 'other' if x in rare_labels else x)
    return df

# Prepare data
df = engineer_features(df)
X = df.drop(columns=['target_income'])
y = df['target_income']
X = reduce_cardinality(X)

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(include='number').columns.tolist()

# Pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Model
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectKBest, f_regression

lgbm = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)

xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)

stacking_regressor = StackingRegressor(
    estimators=[
        ('lgbm', lgbm),
        ('xgb', xgb)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess for importance
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names
ohe_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
all_feature_names = numeric_cols + list(ohe_feature_names)

# Train LGBM on full processed data
lgbm.fit(X_train_processed, y_train)

# Feature importance
importances = lgbm.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top features
plt.figure(figsize=(12, 8))
sns.barplot(data=feat_imp_df.head(30), y='Feature', x='Importance', palette='magma')
plt.title("Top 30 Important Features from LightGBM")
plt.tight_layout()
plt.show()

# Select top N important features
top_n = 100
selected_features = feat_imp_df.head(top_n)['Feature'].tolist()

# Custom transformer to keep only selected features
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=100):
        self.k = k
        self.selector = SelectKBest(score_func=f_regression, k=self.k)
        
    def fit(self, X, y=None):
        self.selector.fit(X, y)
        return self
        
    def transform(self, X):
        return self.selector.transform(X)

# Create the pipeline with the proper feature selector
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selector", FeatureSelector(k=100)),  # Select top 100 features
    ("model", stacking_regressor)
])

# Train
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ MAE: {mae}")
print(f"✅ R2: {r2}")
print(f"✅ RMSE: {rmse}")
print("✅ Training complete.")