import streamlit as st
import plotly.express as px


def render_overview(filtered_summary):
    col1, col2 = st.columns(2)
    with col1:
        metric_choice = st.selectbox(
            "Select Primary Performance Metric",
            ["Accuracy", "F1 Score (Weighted Average)", "F1 Score (Macro Average)"],
        )
    with col2:
        sort_by = st.selectbox(
            "Sort Models By", ["Reduction (%)", "Accuracy", "Parameter Count"]
        )

    metric_mapping = {
        "Accuracy": "accuracy",
        "F1 Score (Weighted Average)": "f1_weighted_avg",
        "F1 Score (Macro Average)": "f1_macro_avg",
    }
    sort_mapping = {
        "Reduction (%)": "reduction_percent",
        "Accuracy": "accuracy",
        "Parameter Count": "parameter_count",
    }

    sorted_df = filtered_summary.sort_values(sort_mapping[sort_by])

    st.subheader("Architecture Configuration")
    architecture_table = sorted_df[
        [
            "model_name",
            "reduction_percent",
            "attention_heads",
            "hidden_dim",
            "ffn",
            "parameter_count",
            "trained_epochs",
        ]
    ].rename(
        columns={
            "model_name": "Model Name",
            "reduction_percent": "Reduction (%)",
            "attention_heads": "Attention Heads",
            "hidden_dim": "Hidden Dimension",
            "ffn": "FFN Size",
            "parameter_count": "Parameter Count",
            "trained_epochs": "Trained Epochs",
        }
    )
    st.dataframe(architecture_table, use_container_width=True)

    st.subheader("Performance vs Model Size")
    overview_df = sorted_df.rename(
        columns={
            "accuracy": "Accuracy",
            "parameter_count": "Parameter Count",
            "model_name": "Model",
        }
    )
    fig = px.line(
        overview_df,
        x="Parameter Count",
        y="Accuracy",
        text="Model",
        markers=True,
        title=f"{metric_choice} vs Parameter Count",
    )
    fig.update_traces(textposition="top center", texttemplate="%{y:.2%}")
    fig.update_layout(
        xaxis_title="Parameter Count", yaxis_title=metric_choice, yaxis_tickformat=".0%"
    )

    st.plotly_chart(fig, use_container_width=True)
