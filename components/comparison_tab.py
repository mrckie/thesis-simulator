import streamlit as st
import pandas as pd
import plotly.express as px
import re

def render_comparison_section(summary_df, curves_df, confusion_df, load_sentiment_models_func):
    st.markdown("<h1 style='text-align: center;'>Baseline vs Modified Architectures</h1>", unsafe_allow_html=True)

    # --- GLOBAL ENVIRONMENT & MODEL SELECTION ---
    st.markdown("### Environment & Model Selection")
    env_col, mod_col = st.columns(2)

    # Mapping for environment display names
    environment_map = {
        "Environment with GPU": "gpu_t4",
        "Environment with only CPU": "cpu_default"
    }

    with env_col:
        selected_env_display = st.selectbox("Select Environment", options=list(environment_map.keys()), index=0)
        selected_environment = environment_map[selected_env_display]

    # Filter Data by Environment immediately
    env_summary = summary_df[summary_df["environment"].isin([selected_environment, "N/A"])]
    all_env_models = env_summary["model_name"].unique()
    models_to_hide = ["baseline 58k sample analysis", "baseline 12.5k sample analysis"]
    available_models = [m for m in all_env_models if m not in models_to_hide]
    
    # Mapping for model display names
    model_display_map = {
        "baseline": "baseline",
        "33.33% reduction": "33.33% reduction"
    }
    # Reverse mapping to get display names from actual model names
    model_reverse_map = {v: k for k, v in model_display_map.items()}
    
    # Get available display names
    available_display_models = [model_reverse_map.get(m, m) for m in available_models]
    valid_defaults_display = [m for m in ["baseline", "33.33% reduction"] if model_display_map.get(m) in available_models]

    with mod_col:
        selected_display_models = st.multiselect("Select Models to Compare / Show", options=available_display_models, default=valid_defaults_display)
        # Convert display names back to actual model names
        selected_models = [model_display_map.get(m, m) for m in selected_display_models]

    # --- GLOBAL DATA FILTERING ---
    filtered_summary = env_summary[env_summary["model_name"].isin(selected_models)].copy()
    filtered_summary["parameter_count"] = filtered_summary["parameter_count"].astype(str).str.replace(',', '', regex=False)
    filtered_summary["parameter_count"] = pd.to_numeric(filtered_summary["parameter_count"], errors='coerce').fillna(0)
    filtered_summary["accuracy"] = pd.to_numeric(filtered_summary["accuracy"], errors='coerce').fillna(0)
    filtered_summary["reduction_percent"] = pd.to_numeric(filtered_summary["reduction_percent"], errors='coerce').fillna(0)

    env_curves = curves_df[curves_df["environment"].isin([selected_environment, "N/A"])]
    filtered_curves = env_curves[env_curves["model_name"].isin(selected_models)]

    env_confusion = confusion_df[confusion_df["environment"].isin([selected_environment, "N/A"])]
    filtered_confusion = env_confusion[env_confusion["model_name"].isin(selected_models)]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Architecture & Size", "Predictive Performance", "Computational Efficiency", 
        "Training Curves", "Confusion Matrix", "Sentiment Analysis (Live)" 
    ])

    # --- TAB 1: ARCHITECTURE ---
    with tab1:
        st.markdown("### Architecture Configuration")
        sort_by_arch = st.selectbox("Sort Table By", ["Parameter Count", "Reduction (%)"], key="arch_sort")
        sort_col = "parameter_count" if sort_by_arch == "Parameter Count" else "reduction_percent"
        arch_df = filtered_summary.sort_values(sort_col, ascending=False)

        architecture_table = arch_df[["model_name", "reduction_percent", "attention_heads", "hidden_dim", "ffn", "parameter_count", "trained_epochs"]].rename(columns={"model_name": "Model", "reduction_percent": "Reduction (%)", "attention_heads": "Heads", "hidden_dim": "Hidden", "ffn": "FFN", "parameter_count": "Parameters", "trained_epochs": "Epochs"})
        st.dataframe(architecture_table.fillna("N/A"), use_container_width=True)

        st.markdown("### Model Parameter Count")
        fig_params = px.bar(architecture_table, x="Model", y="Parameters", text="Parameters", color="Model")
        fig_params.update_traces(texttemplate='%{text:,}', textposition='outside') 
        fig_params.update_layout(xaxis_title="Model", yaxis_title="Total Parameters", yaxis_range=[0, architecture_table["Parameters"].max() * 1.2])
        st.plotly_chart(fig_params, use_container_width=True)

    # --- TAB 2: PREDICTIVE PERFORMANCE ---
    with tab2:
        st.markdown("### Predictive Performance")
        metric_options = {
            "Accuracy": "accuracy", "Precision (Weighted Average)": "precision_weighted_avg",
            "Precision (Macro Average)": "precision_macro_avg", "Recall (Weighted Average)": "recall_weighted_avg",
            "Recall (Macro Average)": "recall_macro_avg", "F1 Score (Weighted Average)": "f1_weighted_avg",
            "F1 Score (Macro Average)": "f1_macro_avg"
        }
        selected_metric_label = st.selectbox("Select Metric to Visualize", list(metric_options.keys()))
        detailed_metric_choice = metric_options[selected_metric_label]

        perf_df = filtered_summary.rename(columns={"model_name": "Model"}).copy()
        perf_df[detailed_metric_choice] = pd.to_numeric(perf_df[detailed_metric_choice], errors='coerce').fillna(0)
        
        # PERMANENT SORT: Highest Score to Lowest Score
        perf_df = perf_df.sort_values(detailed_metric_choice, ascending=False)

        fig_detailed = px.bar(perf_df, x="Model", y=detailed_metric_choice, color="Model", text_auto=True)
        fig_detailed.update_traces(texttemplate='%{y:.2%}', textposition='outside')
        fig_detailed.update_layout(xaxis_title="Model", yaxis_title=selected_metric_label, yaxis_tickformat=".0%", yaxis_range=[0, 1.1])
        st.plotly_chart(fig_detailed, use_container_width=True)

        st.markdown("#### Detailed Classification Metrics Table")
        classification_table = filtered_summary[["model_name", "accuracy", "precision_weighted_avg", "precision_macro_avg", "recall_weighted_avg", "recall_macro_avg", "f1_weighted_avg", "f1_macro_avg"]].rename(columns={"model_name": "Model Name", "accuracy": "Accuracy", "precision_weighted_avg": "Precision (Weighted)", "precision_macro_avg": "Precision (Macro)", "recall_weighted_avg": "Recall (Weighted)", "recall_macro_avg": "Recall (Macro)", "f1_weighted_avg": "F1 Score (Weighted)", "f1_macro_avg": "F1 Score (Macro)"})
        
        classification_table_pct = classification_table.copy()
        for col in classification_table.columns[1:]:
            classification_table_pct[col] = (pd.to_numeric(classification_table_pct[col], errors='coerce') * 100).round(2).astype(str) + "%"
        st.dataframe(classification_table_pct, use_container_width=True)

    # --- TAB 3: COMPUTATIONAL EFFICIENCY ---
    with tab3:
        col1, col2 = st.columns(2)
        with col1: 
            time_unit = st.selectbox("Training Time Unit", ["Seconds", "Minutes"])
        with col2: 
            memory_unit = st.selectbox("Memory Unit", ["MB", "GB"])

        efficiency_df = filtered_summary.copy()
        efficiency_df["train_time"] = pd.to_numeric(efficiency_df["train_time"], errors='coerce').fillna(0)
        efficiency_df["peak_gpu_usage"] = pd.to_numeric(efficiency_df["peak_gpu_usage"], errors='coerce').fillna(0)
        efficiency_df["peak_ram_usage"] = pd.to_numeric(efficiency_df["peak_ram_usage"], errors='coerce').fillna(0)
        
        if time_unit == "Minutes":
            efficiency_df["train_time_display"] = efficiency_df["train_time"] / 60
            time_label = "Training Time (Minutes)"
        else:
            efficiency_df["train_time_display"] = efficiency_df["train_time"]
            time_label = "Training Time (Seconds)"

        if memory_unit == "MB":
            efficiency_df["gpu_memory_display"] = efficiency_df["peak_gpu_usage"] * 1000
            efficiency_df["cpu_memory_display"] = efficiency_df["peak_ram_usage"] * 1000
            gpu_memory_label = "Peak GPU Memory Usage (MB)"
            cpu_memory_label = "Peak CPU Memory Usage (MB)"
        else:
            efficiency_df["gpu_memory_display"] = efficiency_df["peak_gpu_usage"]
            efficiency_df["cpu_memory_display"] = efficiency_df["peak_ram_usage"]
            gpu_memory_label = "Peak GPU Memory Usage (GB)"
            cpu_memory_label = "Peak CPU Memory Usage (GB)"

        efficiency_display = efficiency_df.rename(columns={"model_name": "Model", "train_time_display": "Training Time", "gpu_memory_display": gpu_memory_label, "cpu_memory_display": cpu_memory_label})

        st.subheader("Training Time Comparison")
        fig = px.bar(efficiency_display, x="Model", y="Training Time", color="Model", text_auto=True, title=time_label)
        fig.update_traces(texttemplate='%{y:.2f}')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Peak GPU Memory Usage")
        fig2 = px.bar(efficiency_display, x="Model", y=gpu_memory_label, color="Model", text_auto=True, title=gpu_memory_label)
        fig2.update_traces(texttemplate='%{y:.2f}')
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Peak CPU Memory Usage")
        fig3 = px.bar(efficiency_display, x="Model", y=cpu_memory_label, color="Model", text_auto=True, title=cpu_memory_label)
        fig3.update_traces(texttemplate='%{y:.2f}')
        st.plotly_chart(fig3, use_container_width=True)

    # --- TAB 4: TRAINING CURVES ---
    with tab4:
        st.subheader("Epochs Configuration")
        epochs_info = filtered_summary[["model_name", "epochs", "trained_epochs"]].rename(columns={"model_name": "Model Name", "epochs": "Hyperparameter Epochs", "trained_epochs": "Actual Epochs (with Early Stopping)"}).drop_duplicates()
        for col in ["Actual Epochs (with Early Stopping)", "Hyperparameter Epochs"]:
            epochs_info[col] = pd.to_numeric(epochs_info[col], errors='coerce').fillna(0)
        
        num_models = len(epochs_info)
        if num_models > 0:
            cols = st.columns(num_models)
            for idx, row in epochs_info.reset_index(drop=True).iterrows():
                actual = int(row['Actual Epochs (with Early Stopping)'])
                hyper = int(row['Hyperparameter Epochs'])
                delta_val = actual - hyper
                with cols[idx]:
                    st.metric(row["Model Name"], f"{actual}/{hyper}", delta=f"{delta_val} epochs")
        
        st.info("ⓘ Early stopping is configured with patience of 3 epochs.")

        curves_df_display = filtered_curves.copy()
        curves_df_display["training_loss"] = pd.to_numeric(curves_df_display["training_loss"], errors='coerce').fillna(0)
        curves_df_display["validation_loss"] = pd.to_numeric(curves_df_display["validation_loss"], errors='coerce').fillna(0)
        curves_df_display = curves_df_display.rename(columns={"model_name": "Model", "epoch": "Epoch", "training_loss": "Training Loss", "validation_loss": "Validation Loss"})

        st.subheader("Training Loss")
        fig = px.line(curves_df_display, x="Epoch", y="Training Loss", color="Model", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Validation Loss")
        fig2 = px.line(curves_df_display, x="Epoch", y="Validation Loss", color="Model", markers=True)
        st.plotly_chart(fig2, use_container_width=True)

    # --- TAB 5: CONFUSION MATRIX ---
    with tab5:
        if selected_models:
            for row_start in range(0, len(selected_models), 2):
                cols = st.columns(2)
                for idx in range(2):
                    model_index = row_start + idx
                    if model_index >= len(selected_models):
                        continue
                    model_name = selected_models[model_index]
                    with cols[idx]:
                        cm_df_model = filtered_confusion[filtered_confusion["model_name"] == model_name].copy()
                        cm_df_model["count"] = pd.to_numeric(cm_df_model["count"], errors='coerce').fillna(0)
                        cm_display = cm_df_model.rename(columns={"true_label": "True Label", "predicted_label": "Predicted Label"})

                        if not cm_display.empty:
                            matrix = cm_display.pivot(index="True Label", columns="Predicted Label", values="count")
                            fig = px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", title=f"{model_name} confusion matrix")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No confusion matrix data available for {model_name}.")
        else:
            st.warning("No models selected. Please choose models in the Comparison selector above.")

    # --- TAB 6: SENTIMENT ANALYSIS ---
    with tab6:
        st.markdown("### Emotion-Based Sentiment Inference")
        st.info(
            "This tool compares the real-world predictive behavior of the "
            "**Baseline Model** against the **Optimized Architecture (41.67% Reduction)**. "
            "Both models are hosted on and retrieved directly from the **Hugging Face Model Hub**"
        )

        user_input = st.text_area("Enter a short sentence expressing emotion or sentiment:", placeholder="e.g., I feel happy today.", height=100)

        if st.button("Analyze Sentiment", type="primary"):
            
            # ==========================================
            # INPUT VALIDATION (BETA TESTING CRITERIA)
            # ==========================================
            is_valid = True
            
            # 1. Check if empty
            if not user_input.strip():
                st.warning("Please enter some text to analyze.")
                is_valid = False
                
            # 2. Check for minimum length (Cannot be just "a" or "ok")
            elif len(user_input.strip()) < 3:
                st.warning("Input is too short. Please enter a meaningful sentence expressing an emotion.")
                is_valid = False
                
            # 3. Check for maximum length (Prevents DistilBERT token overflow / crashes)
            elif len(user_input) > 1000:
                st.warning("Input is too long! Please limit your text to a short paragraph (under 1000 characters).")
                is_valid = False
                
            # 4. Check for Gibberish/Symbols (Must contain actual alphabetical letters)
            elif not re.search('[a-zA-Z]', user_input):
                st.error("Invalid Input: The inputted data consists only of symbols or numbers and does not express emotions.")
                is_valid = False

            # ==========================================
            # INFERENCE EXECUTION
            # ==========================================
            if is_valid:
                try:
                    baseline_model, compressed_model = load_sentiment_models_func()
                    
                    base_result = baseline_model(user_input)[0]
                    comp_result = compressed_model(user_input)[0]
                    
                    label_map = {"LABEL_0": "Negative 🔴", "LABEL_1": "Positive 🟢"}
                    
                    base_label = label_map.get(base_result['label'], base_result['label'])
                    base_score = base_result['score']
                    comp_label = label_map.get(comp_result['label'], comp_result['label'])
                    comp_score = comp_result['score']
                    
                    st.markdown("#### Inference Results")
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.markdown("**Baseline Model**")
                        st.metric(label="Prediction", value=base_label)
                        st.progress(base_score, text=f"Confidence: {base_score:.2%}")
                        
                    with res_col2:
                        st.markdown("**Optimized Model (41.67% Architecture Reduction)**")
                        st.metric(label="Prediction", value=comp_label)
                        st.progress(comp_score, text=f"Confidence: {comp_score:.2%}")
                        
                    if base_label != comp_label:
                        st.warning("The models disagree on this sentiment!")
                    else:
                        st.success("Both models agree on the sentiment!")
                        
                except Exception as e:
                    st.error(f"Error loading or running models: {e}")