import pandas as pd 

def load_data(file):

    file_name = file.name.lower()

    if file_name.endswith(".csv"):
        df=pd.read_csv(file)

    elif file_name.endswith(".xls",".xlsx"):
        df=pd.read_excel(file)

    elif file_name.endswith(".parquet"):
        df=pd.read_parquet(file)

    else:
        raise ValueError(
            "please upload right file"
        )

    return df