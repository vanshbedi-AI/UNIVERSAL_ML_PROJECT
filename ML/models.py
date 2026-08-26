from sklearn.linear_model import(
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso
)

from sklearn.tree import(
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor
)

from sklearn.svm import(
    SVC,
    SVR
)

from sklearn.naive_bayes import(
    GaussianNB
)

from sklearn.pipeline import Pipeline

CLASSIFICATION_MODELS = {

    "Logisitic":
        LogisticRegression(max_iter=1000),

    "Decision":
        DecisionTreeClassifier(
            random_state=42
        ),

    'Random':
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
    "gradient":
        GradientBoostingClassifier(
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        ),

    "Naive":
        GaussianNB()

        
}

REGRESSION_MODELS = {

    "Linear":
        LinearRegression(),

    "Decision":
        DecisionTreeRegressor(
            random_state=42
        ),

    'Random':
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        
    "gradient":
        GradientBoostingRegressor(
            random_state=42
        ),

    "KNN":
        KNeighborsRegressor(),

    "lasso":
        Lasso(),

    "Ridge":
        Ridge()
      
}

def get_models(problem_type):

    if problem_type == "CLASSIFICATION":
        return CLASSIFICATION_MODELS
    else:
        return REGRESSION_MODELS

