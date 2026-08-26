import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def evaluate_classification(model,X_test,y_test):

    start_time =time.perf_counter()

    predictions = model.predict(X_test)

    latency = (time.perf_counter() - start_time)

    results = {

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions,
                average = "weighted",
                zero_division=0
            ),
        "recall":
            recall_score(
                y_test,
                predictions,
                average = "weighted",
                zero_division=0
            ),
        "F1 Score":
            f1_score(
                y_test,
                predictions,
                average = "weighted",
                zero_division=0
            ),    

        "Latency": latencyatency

    }

    return results


def evaluate_regression(model,x_test,y_test):
        
    start_time =time.perf_counter()

    predictions = model.predict(x_test)

    latency = (time.perf_counter() - start_time)

    results = {

        "MAE":
            mean_absolute_error(
                y_test,
                predictions
            ),

        "MSE":
            mean_squared_error(
                y_test,
                predictions
            ),

        "R2":
            r2_score(
                y_test,
                predictions
            ),

        "Latency": 
            latency
    }

    return results




