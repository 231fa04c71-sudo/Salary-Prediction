#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U pandas openpyxl numpy matplotlib scikit-learn')
get_ipython().system('pip install -U ydata-profiling')
get_ipython().system('pip install -U pandera')
get_ipython().system('pip install -U dvc')
get_ipython().system('pip install -U mlflow')


# In[2]:


import os

folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "reports",
    "models",
    "plots"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Project structure created successfully!")


# In[2]:


import pandas as pd
import os

excel_file = "Salary Data.xlsx"

df = pd.read_excel(excel_file)

print("Dataset loaded successfully")
print("Shape:", df.shape)

df.head()


# In[3]:


print("Columns:")
print(df.columns.tolist())


# In[4]:


print("Data Types:")
print(df.dtypes)


# In[5]:


df.info


# In[6]:


raw_path = "salary_mlops/data/raw/salary_data.xlsx"
df.to_excel(raw_path,index=False)
print("Raw dataset saved at:")
print(raw_path)


# In[7]:


print("Missing values:")
print(df.isnull().sum())


# In[8]:


df.describe()


# In[24]:


import os

os.makedirs("data/raw", exist_ok=True)

df.to_excel(
    "data/raw/Salary Data.xlsx",
    index=False
)

print("Raw dataset saved successfully!")


# In[25]:


df.describe(include="all")


# In[26]:


for column in df.columns:
    print(column, ":", df[column].nunique())


# In[27]:


print("Duplicate rows:")
display(df[df.duplicated(keep=False)])


# In[28]:


for column in df.columns:
    duplicate_values = df[column][df[column].duplicated()]

    if not duplicate_values.empty:
        print("\nDuplicate values in", column)
        print(duplicate_values.unique())


# In[29]:


df_before = df.copy()

print("Rows before removing duplicates:", len(df))

df = df.drop_duplicates()

print("Rows after removing duplicates:", len(df))

print("Duplicates removed:",
      len(df_before) - len(df))


# In[30]:


get_ipython().run_line_magic('pip', 'install ipywidgets')


# In[31]:


get_ipython().run_line_magic('pip', 'install -U ipywidgets')



# In[32]:


import ipywidgets as widgets

print("ipywidgets installed successfully!")


# In[34]:


import sys

print(sys.version)
print(sys.executable)


# In[1]:


from ydata_profiling import ProfileReport

print("YData Profiling imported successfully!")


# In[13]:


numerical_columns = [
    "Age",
    "Years of Experience"
]


# In[14]:


categorical_columns = [
    "Gender",
    "Education Level",
    "Job Title"
]


# In[16]:


print(df.shape)


# In[17]:


encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# In[20]:


models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=10
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}

print("Models created:")
for model in models:
    print(model)


# In[21]:


# ============================================================
# EMPLOYEE SALARY PREDICTION
# COMPLETE MODEL TRAINING + EVALUATION
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

file_path = "Salary Data.xlsx"

df = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# ------------------------------------------------------------
# 2. CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = df.columns.str.strip()

# ------------------------------------------------------------
# 3. REMOVE DUPLICATES
# ------------------------------------------------------------

df = df.drop_duplicates()

# ------------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ------------------------------------------------------------

# Numerical columns
df["Age"] = df["Age"].fillna(df["Age"].median())

df["Years of Experience"] = df["Years of Experience"].fillna(
    df["Years of Experience"].median()
)

df["Salary"] = df["Salary"].fillna(
    df["Salary"].median()
)

# Categorical columns
df["Gender"] = df["Gender"].fillna(
    df["Gender"].mode()[0]
)

df["Education Level"] = df["Education Level"].fillna(
    df["Education Level"].mode()[0]
)

df["Job Title"] = df["Job Title"].fillna(
    df["Job Title"].mode()[0]
)

# ------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ------------------------------------------------------------

df["Experience_Ratio"] = (
    df["Years of Experience"] / df["Age"]
)

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 45, 60, 100],
    labels=[
        "Young",
        "Early Career",
        "Mid Career",
        "Senior",
        "Late Career"
    ]
)

# ------------------------------------------------------------
# 6. FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop("Salary", axis=1)

y = df["Salary"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget: Salary")

# ------------------------------------------------------------
# 7. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# ------------------------------------------------------------
# 8. PREPROCESSING
# ------------------------------------------------------------

numeric_features = [
    "Age",
    "Years of Experience",
    "Experience_Ratio"
]

categorical_features = [
    "Gender",
    "Education Level",
    "Job Title",
    "Age_Group"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# ------------------------------------------------------------
# 9. DEFINE MODELS
# ------------------------------------------------------------

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=10
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}

# ------------------------------------------------------------
# 10. TRAIN MODELS
# ------------------------------------------------------------

trained_models = {}

results = []

print("\n================ MODEL TRAINING ================\n")

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Store trained model
    trained_models[name] = pipeline

    # Predict
    predictions = pipeline.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"{name}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")
    print()

# ------------------------------------------------------------
# 11. RESULTS TABLE
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
).reset_index(drop=True)

print("=============== MODEL COMPARISON ===============")

display(results_df)

# ------------------------------------------------------------
# 12. BEST MODEL
# ------------------------------------------------------------

best_model_name = results_df.loc[0, "Model"]

best_model = trained_models[best_model_name]

best_r2 = results_df.loc[0, "R2"]

print("\n================================================")
print("BEST MODEL")
print("================================================")

print("Model:", best_model_name)
print("R² Score:", round(best_r2, 4))

print("\nAll models trained successfully!")


# In[22]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("Employee Salary Prediction - Model Comparison")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# In[23]:


import joblib

joblib.dump(
    best_model,
    "salary_prediction_model.pkl"
)

print("Best model saved successfully!")
print("Model:", best_model_name)


# In[3]:


get_ipython().run_line_magic('pip', 'install -U scikit-learn joblib')


# In[1]:


import sklearn
import joblib

print("scikit-learn:", sklearn.__version__)
print("joblib: OK")


# In[2]:


import sys
print(sys.executable)


# In[4]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("All 4 algorithms imported successfully!")


# In[5]:


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=10
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

print("Algorithms selected:")
for name in models:
    print("-", name)


# In[8]:


get_ipython().run_line_magic('pip', 'install -U openpyxl')


# In[9]:


import pandas as pd

df = pd.read_excel("Salary Data.xlsx")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())


# In[10]:


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Remove duplicates and missing values
df = df.drop_duplicates()
df = df.dropna()

# Features and target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Numerical and categorical columns
numerical_columns = [
    "Age",
    "Years of Experience"
]

categorical_columns = [
    "Gender",
    "Education Level",
    "Job Title"
]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_columns),
    ("cat", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ), categorical_columns)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("ML setup completed!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# In[11]:


models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=10
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Algorithm": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    print(name, "trained successfully!")

results_df = pd.DataFrame(results)

display(results_df)


# In[17]:


get_ipython().run_line_magic('pip', 'install -U mlflow')


# In[19]:


import mlflow
import mlflow.sklearn

print("MLflow:", mlflow.__version__)
print("MLflow imported successfully!")


# In[20]:


import pandas as pd

df = pd.read_excel("Salary Data.xlsx")
df = df.drop_duplicates()
df = df.dropna()

print(df.shape)


# In[21]:


X = df.drop("Salary", axis=1)
y = df["Salary"]

numerical_columns = ["Age", "Years of Experience"]
categorical_columns = ["Gender", "Education Level", "Job Title"]


# In[22]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])


# In[23]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# In[24]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=10
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

trained_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    trained_models[name] = pipeline

    print(name, "trained successfully!")


# In[25]:


results = []

for name, pipeline in trained_models.items():

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Algorithm": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

results_df = pd.DataFrame(results)

display(results_df)


# In[27]:


import mlflow
import mlflow.sklearn
import numpy as np

mlflow.set_experiment("Salary Prediction")

for name, pipeline in trained_models.items():

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    with mlflow.start_run(run_name=name):

        mlflow.log_param("Algorithm", name)
        mlflow.log_param("Test_Size", 0.20)
        mlflow.log_param("Random_State", 42)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2_Score", r2)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
        )

        print(name, "→ logged successfully")


# In[28]:


# Compare all trained models

results = []

for name, pipeline in trained_models.items():

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Algorithm": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2
    })

results_df = pd.DataFrame(results)

# Sort by best R2 score
results_df = results_df.sort_values(
    by="R2_Score",
    ascending=False
)

print("Model Performance:")
display(results_df)


# In[29]:


best_model_name = results_df.iloc[0]["Algorithm"]
best_model = trained_models[best_model_name]

print("Best Model:", best_model_name)
print("Best R2 Score:", results_df.iloc[0]["R2_Score"])


# In[30]:


import joblib

joblib.dump(best_model, "best_salary_model.pkl")

print("Best model saved successfully!")


# In[31]:


import joblib

joblib.dump(best_model, "best_salary_model.pkl")

print("Best model saved successfully!")


# In[32]:


# ==========================================
# SALARY PREDICTION - COMPLETE ML PIPELINE
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------

df = pd.read_excel("Salary Data.xlsx")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ------------------------------------------
# 2. Data Cleaning
# ------------------------------------------

df = df.drop_duplicates()
df = df.dropna()

print("After cleaning:", df.shape)


# ------------------------------------------
# 3. Features and Target
# ------------------------------------------

X = df.drop("Salary", axis=1)
y = df["Salary"]

numerical_columns = [
    "Age",
    "Years of Experience"
]

categorical_columns = [
    "Gender",
    "Education Level",
    "Job Title"
]


# ------------------------------------------
# 4. Preprocessing
# ------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


# ------------------------------------------
# 5. Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------
# 6. Algorithms
# ------------------------------------------

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=10
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


print("\nAlgorithms selected:")

for name in models:
    print("-", name)


# ------------------------------------------
# 7. Train Models and Evaluate
# ------------------------------------------

trained_models = {}
results = []

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    trained_models[name] = pipeline

    results.append({
        "Algorithm": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2
    })

    print(
        f"{name} -> "
        f"MAE: {mae:.2f}, "
        f"RMSE: {rmse:.2f}, "
        f"R2: {r2:.4f}"
    )


# ------------------------------------------
# 8. Comparison Table
# ------------------------------------------

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df)


# ------------------------------------------
# 9. Select Best Model
# ------------------------------------------

best_model_name = results_df.loc[
    results_df["R2_Score"].idxmax(),
    "Algorithm"
]

best_model = trained_models[best_model_name]

best_r2 = results_df.loc[
    results_df["R2_Score"].idxmax(),
    "R2_Score"
]

print("\nBest Model:", best_model_name)
print("Best R2 Score:", best_r2)


# ------------------------------------------
# 10. Save Best Model
# ------------------------------------------

joblib.dump(
    best_model,
    "best_salary_model.pkl"
)

print("\nBest model saved successfully!")


# ------------------------------------------
# 11. Save Model Results
# ------------------------------------------

results_df.to_csv(
    "model_results.csv",
    index=False
)

print("Model results saved successfully!")


# ------------------------------------------
# 12. Final Summary
# ------------------------------------------

print("\n===================================")
print("SALARY PREDICTION COMPLETED")
print("===================================")

print("Algorithms:")
for name in models:
    print(" -", name)

print("\nBest Model:", best_model_name)
print("Best R2 Score:", round(best_r2, 4))


# In[ ]:




