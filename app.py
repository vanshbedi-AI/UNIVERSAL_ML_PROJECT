import streamlit as st 
import andas as pd

from ML.data_loader import load_data
from ML.profiler import profile_data

st.set_page_config(
    page_title="UNIVERSAL ML",
    layout="wide"
)

st.title("UNIVERSAL ML")


st.write(
    "upload tabular dataset and analyze it"
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload",
    type=["csv","xlx","xlsx"]
)

if uploaded_file:

    try:

        df=load_data(uploaded_file)

        st.success("dataset loaded")

        st.subheader("Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        profile =profile_data(df)

        st.subheader("Datset overview")

        col1,col2,col3,col4 = st.columns()

        col1.metric(
            "row",
            profile["rows"]
        )
        col2.metric(
            "columns",
            profile["columns"]
        )
        col3.metric(
            "missing",
            profile["missing_values"]
        )
        col4.metric(
            "duplicated",
            profile["duplicated"]
        )

        st.subheader("column info")

        column_info= pd.DataFrame({
            "data_type": df.dtypes.astypes(str),
            "unique":[
                df[col].nunique()
                for col in df.columns
            ]

        })

        st.dataframe(
            column_info,
            use_container_width=True
        )


        st.subheader("ML Config")

        target_column=st.selectbox(
            "Select Target column",
            df.columns
        )

        problem_type= st.radio(
            "select",
            ["classification","regression"],
            horizontal=True
        )

        st.write(
            f"**Problem:** {problem_type}"
        )

    except Exception as e:

        st.error(f"Error: {e}")