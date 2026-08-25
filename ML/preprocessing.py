import pandas as pd  

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import(
    OneHotEncoder,
    StandardScaler
)

def create_preprocessort(X):

    numerical_columns=X.select_dtypes(
        include=["int64","float64","int32","float32"]
    ).columns.tolist()

    categorical_columns=X.select_dtypes(
        include=["category","object","bool"]
    ).columns.tolist()

    numerical_pipeline=Pipeline(
        steps=[
            (
            "imputer",
            SimpleImputer(strategy="median")
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )
    categorical_pipeline = Pipeline(
        steps= [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown ="ignore",
                    sparse_output=False
                )
            )

        ]   
    )

    transformers = []

    if numerical_columns:
        transformers.append(
            numerical_pipeline,
            numerical_columns
        )

    if categorical_columns:
        transformers.append(
            categorical_pipeline,
            categorical_columns
        )

    
    preprocessor =ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor