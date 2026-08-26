import pandas as pd 

from ML.models import get_models
from ML.trainer import train_model

from ML.evaluator import(
    evaluate_classification,
    evaluate_regression
)

def benchmark_models(
    df,
    target_column,
    problem_type,
    selected_models,
    test_size=0.2
):
        models = get_models(problem_type)

        results = []

        trained_models = {}

        for model_name in selected_models:

            model = models[model_name]

            (
                pipeline,
                X_train,
                X_test,
                y_train,
                y_test,
                training_time
            ) = train_model(
                  df=df,
                  target_column=target_column,
                  model=model,
                  test_size=test_size
            )

            if problem_type == "CLASSIFICATION":

                metrics = evaluate_classification(
                    pipeline,
                    X_test,
                    y_test
                )

            else:

                metrics = evaluate_regression(
                    pipeline,
                    X_test,
                    y_test
                )

            metrics["Model"] = model_name

            metrics[
                "Training Time"
            ] = training_time

            results.append(metrics)


            trained_models[model_name] = pipeline

        results_df = pd.DataFrame(results)

        return results_df, trained_models

