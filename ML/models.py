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

from sklear.ensemble import(
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBosstingClassifier,
    GradientBosstingRegressor
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor
)

from sklearn.svm import(
    SVC,
    SVM
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
        GradientBosstingClassifier(
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

    "Linear"
        LinearRegression(),

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
        GradientBosstingClassifier(
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

