import pandas as pd 

def profile_data(df):

    profile={
        "row": df.shape[0],
        "columns":df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicated":int(df.duplicated().sum()),
        "numerical":df.select_dtypes(
            include="number"
        ).columns.tolist(),
        "categorical":df.select_dtypes(
            include=["bool","object","category"]
        ).columns.tolist(),

    }

    return profile