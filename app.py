import streamlit as st 
import pandas as pd

from ML.data_loader import load_data
from ML.profiler import profile_data
from ML.models import get_models
from ML.benchmark import benchmark_models

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
            width='stretch'
        )

        profile =profile_data(df)

        st.subheader("Datset overview")

        col1,col2,col3,col4 = st.columns(4)

        col1.metric(
            "row",
            profile["row"]
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
            "data_type": df.dtypes.astype(str),
            "unique":[
                df[col].nunique()
                for col in df.columns
            ]

        })

        st.dataframe(
            column_info,
            width='stretch'
        )


        st.subheader("ML Config")

        target_column=st.selectbox(
            "Select Target column",
            df.columns
        )

        problem_type= st.radio(
            "select problem type",
            ["CLASSIFICATION","REGRESSION"],
            horizontal=True
        )

        st.write(
            f"**Problem:** {problem_type}"
        )

        st.divider()

        st.subheader("Model Selection")

        available_models = get_models(problem_type)

        selected_models = st.multiselect(
            "Select Algo",
            options = list(available_models.keys()),
            default= list(available_models.keys())
        )

        test_size =st.slider(
            "test Size",
            min_value =0.1,
            max_value=0.4,
            value=0.2
        )

        if st.button(
            "Train and benchmark",
            type= "primary"
        ):

            if not selected_models:

                st.warning(
                    "Please select at least one algo"
                )

            else:

                with st.spinner(
                    "training"
                ):
                    results_df,trained_models =(
                        benchmark_models(
                            df=df,
                            target_column = target_column,
                            problem_type=problem_type,
                            selected_models = selected_models,
                            test_size =test_size
                        )

                    )
                st.success(
                    "benchmarking"
                )

                st.subheader(
                    "Model results"
                )

                st.dataframe(
                    results_df,
                    width='stretch'
                )

    except Exception as e:

        st.error(f"Error: {e}")

        st.exception(e)