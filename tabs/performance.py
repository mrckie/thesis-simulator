import streamlit as st
import plotly.express as px


def render_performance(filtered_summary):
    metric_options = {
        "Accuracy": "accuracy",
        "Precision (Weighted Average)": "precision_weighted_avg",
        "Precision (Macro Average)": "precision_macro_avg",
        "Recall (Weighted Average)": "recall_weighted_avg",
        "Recall (Macro Average)": "recall_macro_avg",
        "F1 Score (Weighted Average)": "f1_weighted_avg",
        "F1 Score (Macro Average)": "f1_macro_avg",
    }

    selected_metric_label = st.selectbox(
        "Select Performance Metric", list(metric_options.keys())
    )
    metric_choice = metric_options[selected_metric_label]

    st.subheader(f"{selected_metric_label} Comparison Across Models")

    perf_df = filtered_summary.rename(columns={"model_name": "Model"})
    fig = px.bar(perf_df, x="Model", y=metric_choice, color="Model", text_auto=True)
    fig.update_traces(texttemplate="%{y:.2%}")
    fig.update_layout(
        xaxis_title="Model", yaxis_title=selected_metric_label, yaxis_tickformat=".0%"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Classification Metrics")

    classification_table = filtered_summary[
        [
            "model_name",
            "accuracy",
            "precision_weighted_avg",
            "precision_macro_avg",
            "recall_weighted_avg",
            "recall_macro_avg",
            "f1_weighted_avg",
            "f1_macro_avg",
        ]
    ].rename(
        columns={
            "model_name": "Model Name",
            "accuracy": "Accuracy",
            "precision_weighted_avg": "Precision (Weighted)",
            "precision_macro_avg": "Precision (Macro)",
            "recall_weighted_avg": "Recall (Weighted)",
            "recall_macro_avg": "Recall (Macro)",
            "f1_weighted_avg": "F1 Score (Weighted)",
            "f1_macro_avg": "F1 Score (Macro)",
        }
    )

    classification_table_pct = classification_table.copy()
    for col in classification_table.columns[1:]:
        classification_table_pct[col] = (classification_table_pct[col] * 100).round(
            2
        ).astype(str) + "%"

    st.dataframe(classification_table_pct, use_container_width=True)
