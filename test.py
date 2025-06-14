import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("data/Hackathon_bureau_data_50000.csv")
df = df.dropna(subset=['target_income'])
df = df.drop_duplicates()
df = df.apply(lambda x: x.str.strip().str.lower() if x.dtype == 'object' else x)

categorical_cols = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership',
    'device_model', 'device_category', 'platform', 'device_manufacturer', 'score_type', 'score_comments'
]
exclude_cols = ['unique_id', 'target'] + categorical_cols
numeric_cols = [col for col in df.columns if col not in exclude_cols]


print(df)
def engineer_features(df):
    df = df.copy()
    important_numeric = ['age'] + [col for col in df.columns if col.startswith('var_')]
    print("imp",important_numeric)
    # Only create interactions between important features
    for i in range(len(important_numeric)):
        for j in range(i+1, len(important_numeric)):
            col1, col2 = important_numeric[i], important_numeric[j]
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
    # Polynomial features
    for col in important_numeric:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2
    return df

df = engineer_features(df)

# --- Remove Outliers in Target ---
q_low = df['target_income'].quantile(0.01)
q_hi  = df['target_income'].quantile(0.99)
df = df[(df['target_income'] > q_low) & (df['target_income'] < q_hi)]

# --- Log-transform the Target ---
import numpy as np
y = np.log1p(df['target_income'])
X = df.drop(columns=['target_income'])

# Replace rare categories with 'other'
def reduce_cardinality(df, threshold=0.01):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        freqs = df[col].value_counts(normalize=True)
        rare_labels = freqs[freqs < threshold].index
        df[col] = df[col].apply(lambda x: 'other' if x in rare_labels else x)
    return df

X = df.drop(columns=['target_income'])
y = df['target_income']

X = reduce_cardinality(X)
df = engineer_features(df)


# categorical_cols = X.select_dtypes(include='object').columns.tolist()
# numeric_cols = X.select_dtypes(include='number').columns.tolist()   #0.5s

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


from sklearn.feature_selection import SelectKBest, f_regression
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

from sklearn.linear_model import Lasso , Ridge
from sklearn.feature_selection import SelectFromModel

lasso = Lasso(alpha=0.01)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", SelectFromModel(lasso))
])

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.feature_selection import SelectKBest, f_regression , mutual_info_regression


# feature_selector = SelectKBest(f_regression, k=1000)  # Select top 1000 features


# model = LGBMRegressor(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=4,
#     n_jobs=-1,
#     random_state=42
# )
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

# Create stacking model
stacking_regressor = StackingRegressor(
    estimators=[
        ('lgbm', lgbm),
        ('xgb', xgb)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    # ("model", model)
    ("feature_selector", SelectKBest(mutual_info_regression, k=2000)),  # Using mutual_info_regression for better feature selection
    ("model", stacking_regressor)

])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Calculate metrics
from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"R2: {r2}")

print("✅ Training complete.")
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

y_pred = pipeline.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))