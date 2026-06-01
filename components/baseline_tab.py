import streamlit as st
import pandas as pd
import plotly.express as px

def render_baseline_section(summary_df, curves_df, confusion_df, dataset_info_df, class_imbalance_df):
    st.markdown("<h4 style='text-align: center;'>Baseline Analysis</h4>", unsafe_allow_html=True)

    baseline_names = ["baseline 58k sample analysis", "baseline 12.5k sample analysis"]
    baseline_summary = summary_df[summary_df["model_name"].isin(baseline_names)].copy()

    base_tab1, base_tab2, base_tab3, base_tab4 = st.tabs([
        "Dataset Configuration",
        "Classification Performance",
        "Training Curves",
        "Confusion Matrix"
    ])

    # --- TAB 1: DATASET CONFIGURATION & IMBALANCE ---
    with base_tab1:
        # Create two columns for the charts
        col_data1, col_data2 = st.columns(2, gap="large")

        with col_data1:
            st.markdown("<h5 style='text-align: center;'>Dataset Configuration</h5>", unsafe_allow_html=True)
            dataset_df = dataset_info_df[dataset_info_df["dataset"].isin(baseline_names)].copy()
            dataset_df = dataset_df.rename(columns={"dataset": "Dataset", "total": "Total Samples", "train": "Train", "val": "Validation", "test": "Test"})
            
            plot_df = dataset_df.melt(id_vars="Dataset", value_vars=["Train", "Validation", "Test"], var_name="Split", value_name="Samples")
            plot_df["Samples"] = pd.to_numeric(plot_df["Samples"], errors='coerce')
            plot_df = plot_df.dropna(subset=["Samples"])

            # Applied neutral/informative colors for dataset splits
            fig_dataset = px.bar(
                plot_df, x="Dataset", y="Samples", color="Split", barmode="group", text="Samples",
                color_discrete_map={
                    "Train": "#1f77b4",       # Deep Corporate Blue
                    "Validation": "#ff7f0e",  # Vibrant Orange
                    "Test": "#17becf"         # Teal
                }
            )
            fig_dataset.update_traces(textposition='outside')
            fig_dataset.update_layout(xaxis_title="Dataset", yaxis_title="Number of Samples", yaxis_range=[0, plot_df["Samples"].max() * 1.2])
            st.plotly_chart(fig_dataset, use_container_width=True)

        with col_data2:
            st.markdown("<h5 style='text-align: center;'>Class Imbalance</h5>", unsafe_allow_html=True)
            imbalance_df = class_imbalance_df[class_imbalance_df["dataset"].isin(baseline_names)].copy()
            imbalance_df['Dataset'] = imbalance_df['dataset']
            imbalance_df = imbalance_df.dropna(subset=['Dataset'])
            imbalance_df['Class'] = imbalance_df['target'].map({0: "Negative", 1: "Positive"})
            imbalance_df['Proportion (%)'] = (imbalance_df['proportion'] * 100).round(1)

            # Applied strict Green/Red mapping for Positive/Negative classes
            fig_imbalance = px.bar(
                imbalance_df, x="Dataset", y="Proportion (%)", color="Class", barmode="group", text="Proportion (%)",
                color_discrete_map={
                    "Positive": "#28a745",    # Success Green
                    "Negative": "#dc3545"     # Warning Red
                }
            )
            fig_imbalance.update_traces(textposition='outside')
            fig_imbalance.update_layout(xaxis_title="Dataset", yaxis_title="Proportion (%)", yaxis_range=[0, 115])
            st.plotly_chart(fig_imbalance, use_container_width=True)

    # --- TAB 2: CLASSIFICATION REPORTS ---
    with base_tab2:
        st.markdown("<h5 style='text-align: center;'>Classification Performance</h5>", unsafe_allow_html=True)
        class_report_table = baseline_summary[["model_name", "accuracy", "precision_macro_avg", "recall_macro_avg", "f1_macro_avg", "precision_weighted_avg", "recall_weighted_avg", "f1_weighted_avg"]].rename(columns={"model_name": "Model", "accuracy": "Accuracy", "precision_macro_avg": "Precision (Macro)", "recall_macro_avg": "Recall (Macro)", "f1_macro_avg": "F1 (Macro)", "precision_weighted_avg": "Precision (Weighted)", "recall_weighted_avg": "Recall (Weighted)", "f1_weighted_avg": "F1 (Weighted)"})
        
        for col in class_report_table.columns[1:]:
            class_report_table[col] = (pd.to_numeric(class_report_table[col], errors='coerce') * 100).round(2).astype(str) + "%"
        st.dataframe(class_report_table, use_container_width=True)

    # --- TAB 3: TRAINING CURVES ---
    with base_tab3:
        st.markdown("<h5 style='text-align: center;'>Training Curves</h5>", unsafe_allow_html=True)
        base_curves = curves_df[curves_df["model_name"].isin(baseline_names)].copy()
        base_curves["training_loss"] = pd.to_numeric(base_curves["training_loss"], errors='coerce')
        base_curves["validation_loss"] = pd.to_numeric(base_curves["validation_loss"], errors='coerce')

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("**58k Baseline**")
            curve_58k = base_curves[base_curves["model_name"] == "baseline 58k sample analysis"]
            if not curve_58k.empty:
                # Map the line color
                fig_c1 = px.line(
                    curve_58k, x="epoch", y=["training_loss"], markers=True,
                    color_discrete_map={"training_loss": "#1f77b4"} # Blue
                )
                fig_c1.update_layout(xaxis_title="Epoch", yaxis_title="Loss", showlegend=True)
                st.plotly_chart(fig_c1, use_container_width=True)
                
        with col_c2:
            st.write("**12.5k Baseline**")
            curve_12k = base_curves[base_curves["model_name"] == "baseline 12.5k sample analysis"]
            if not curve_12k.empty:
                # Map both line colors
                fig_c2 = px.line(
                    curve_12k, x="epoch", y=["training_loss", "validation_loss"], markers=True,
                    color_discrete_map={
                        "training_loss": "#1f77b4",      # Blue
                        "validation_loss": "#ff7f0e"     # Orange
                    }
                )
                fig_c2.update_layout(xaxis_title="Epoch", yaxis_title="Loss", showlegend=True)
                st.plotly_chart(fig_c2, use_container_width=True)

    # --- TAB 4: CONFUSION MATRICES ---
    with base_tab4:
        st.markdown("<h5 style='text-align: center;'>Confusion Matrices</h5>", unsafe_allow_html=True)
        cm_base = confusion_df[confusion_df["model_name"].isin(baseline_names)].copy()
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.write("**58k baseline confusion matrix**")
            cm_58k = cm_base[cm_base["model_name"] == "baseline 58k sample analysis"]
            if not cm_58k.empty:
                matrix_58k = cm_58k.pivot(index="true_label", columns="predicted_label", values="count")
                fig_m1 = px.imshow(matrix_58k, text_auto=True, color_continuous_scale="Blues")
                fig_m1.update_layout(xaxis_title="Predicted", yaxis_title="True")
                st.plotly_chart(fig_m1, use_container_width=True)

        with col_m2:
            st.write("**12.5k baseline confusion matrix**")
            cm_12k = cm_base[cm_base["model_name"] == "baseline 12.5k sample analysis"]
            if not cm_12k.empty:
                matrix_12k = cm_12k.pivot(index="true_label", columns="predicted_label", values="count")
                fig_m2 = px.imshow(matrix_12k, text_auto=True, color_continuous_scale="Blues")
                fig_m2.update_layout(xaxis_title="Predicted", yaxis_title="True")
                st.plotly_chart(fig_m2, use_container_width=True)