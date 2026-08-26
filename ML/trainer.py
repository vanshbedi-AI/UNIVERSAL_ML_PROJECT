import time

from sklearn.pipeline import Pipeline

from sklearn.model_selection  import train_test_split

from ML.preprocessing import create_preprocessor

def train_model(
        df,
        target_column,
        model,
        test_size=0.2,
        random_state=42
):  

        X=df.drop(columns=[target_column])
        y=df[target_column]

        X_train,X_test,y_train,y_test= train_test_split(
                X,y,test_size=test_size,random_state=random_state
        )

        preprocesor =create_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                    (
                        "preprocessor",
                        preprocesor
                    ),
                    (
                        "model",
                        model    
                    )
                ]

            )

        start_time = time.perf_counter()

        pipeline.fit(
                X_train,
                y_train
        )

        training_time = (time.perf_counter() - start_time)

        return(
                pipeline,
                X_train,
                X_test,
                y_train,
                y_test,
                training_time
        )

        

