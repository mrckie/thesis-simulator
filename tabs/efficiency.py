import streamlit as st
import plotly.express as px

def render_efficiency(filtered_summary):
    col1, col2 = st.columns(2)
    with col1:
        time_unit = st.selectbox("Training Time Unit", ["Seconds", "Minutes"])
    with col2:
        memory_unit = st.selectbox("Memory Unit", ["MB", "GB"])

    efficiency_df = filtered_summary.copy()
    
    if time_unit == "Minutes":
        efficiency_df["train_time_display"] = efficiency_df["train_time"] / 60
        time_label = "Training Time (Minutes)"
    else:
        efficiency_df["train_time_display"] = efficiency_df["train_time"]
        time_label = "Training Time (Seconds)"

    if memory_unit == "MB":
        efficiency_df["gpu_memory_display"] = efficiency_df["peak_gpu_usage"] * 1000
        efficiency_df["cpu_memory_display"] = efficiency_df["peak_cpu_usage"] * 1000
        gpu_memory_label = "Peak GPU Memory Usage (MB)"
        cpu_memory_label = "Peak CPU Memory Usage (MB)"
    else:
        efficiency_df["gpu_memory_display"] = efficiency_df["peak_gpu_usage"]
        efficiency_df["cpu_memory_display"] = efficiency_df["peak_cpu_usage"]
        gpu_memory_label = "Peak GPU Memory Usage (GB)"
        cpu_memory_label = "Peak CPU Memory Usage (GB)"

    efficiency_display = efficiency_df.rename(columns={
        "model_name": "Model", "train_time_display": "Training Time",
        "gpu_memory_display": gpu_memory_label, "cpu_memory_display": cpu_memory_label
    })

    st.subheader("Training Time Comparison")
    fig = px.bar(efficiency_display, x="Model", y="Training Time", color="Model", text_auto=True, title=time_label)
    fig.update_traces(texttemplate='%{y:.2f}')
    fig.update_layout(xaxis_title="Model", yaxis_title="Training Time")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Peak GPU Memory Usage")
    fig2 = px.bar(efficiency_display, x="Model", y=gpu_memory_label, color="Model", text_auto=True, title=gpu_memory_label)
    fig2.update_traces(texttemplate='%{y:.2f}')
    fig2.update_layout(xaxis_title="Model", yaxis_title=gpu_memory_label)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Peak CPU Memory Usage")
    fig3 = px.bar(efficiency_display, x="Model", y=cpu_memory_label, color="Model", text_auto=True, title=cpu_memory_label)
    fig3.update_traces(texttemplate='%{y:.2f}')
    fig3.update_layout(xaxis_title="Model", yaxis_title=cpu_memory_label)
    st.plotly_chart(fig3, use_container_width=True)