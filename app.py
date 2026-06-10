import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ml_core import (
    calculate_threshold_metrics,
    classify_failure_by_threshold,
    get_classification_report_from_predictions,
    get_data_quality_report,
    prepare_data,
    train_model,
    train_and_compare_models,
    get_feature_importance,
    save_model,
    load_model,
    build_single_sample,
    predict_single_sample,
    get_risk_level,
    get_maintenance_advice
)


DATA_PATH = "data/ai4i2020.csv"


st.set_page_config(
    page_title="工业设备故障预测系统",
    layout="wide"
)

failure_threshold = st.sidebar.slider(
    "故障判定阈值",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.info(
    "阈值越低，模型越容易判定为故障，更偏向减少漏报；"
    "阈值越高，模型越谨慎，更偏向减少误报。"
)


st.title("基于机器学习的工业设备故障预测系统")

st.markdown("""
本项目面向工业设备预测性维护场景，使用设备运行传感器数据，
训练机器学习模型预测设备是否可能发生故障。

当前系统功能：

1. 读取设备运行数据
2. 展示故障样本分布
3. 训练随机森林故障预测模型
4. 输出模型评估指标
5. 展示混淆矩阵和特征重要性
6. 支持单条设备故障风险预测
7. 输出故障概率、风险等级和维护建议
""")


st.header("1. 数据加载")

uploaded_file = st.file_uploader(
    "请上传设备运行数据 CSV 文件",
    type=["csv"]
)


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV 文件上传成功。")

elif os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.success(f"已从本地加载数据：{DATA_PATH}")

else:
    df = None
    st.warning("请上传 CSV 文件，或将 ai4i2020.csv 放入 data 文件夹。")


if df is not None:
    st.header("2. 数据预览")

    st.write("数据前 5 行：")
    st.dataframe(df.head())

    st.write("数据维度：")
    st.write(f"共 {df.shape[0]} 行，{df.shape[1]} 列")

    st.header("3. 数据质量分析")

    quality_report = get_data_quality_report(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("数据行数", quality_report["shape"][0])
    col2.metric("数据列数", quality_report["shape"][1])

    if quality_report["failure_rate"] is not None:
        col3.metric("故障样本占比", f"{quality_report['failure_rate']:.2%}")
    else:
        col3.metric("故障样本占比", "N/A")

    st.subheader("字段名称和字段类型")
    dtype_df = pd.DataFrame(
        quality_report["dtypes"].items(),
        columns=["字段名称", "字段类型"]
    )
    st.dataframe(dtype_df)

    st.subheader("每列缺失值数量")
    missing_df = pd.DataFrame(
        quality_report["missing_values"].items(),
        columns=["字段名称", "缺失值数量"]
    )
    st.dataframe(missing_df)

    if quality_report["target_distribution"]:
        st.subheader("Machine failure 类别分布")
        target_distribution_df = pd.DataFrame(
            quality_report["target_distribution"].items(),
            columns=["Machine failure", "样本数量"]
        )
        st.dataframe(target_distribution_df)
        st.bar_chart(target_distribution_df.set_index("Machine failure")["样本数量"])
    else:
        st.warning("数据中没有找到目标字段：Machine failure")

    st.header("4. 故障样本分布")

    if "Machine failure" in df.columns:
        failure_counts = df["Machine failure"].value_counts().sort_index()

        col1, col2, col3 = st.columns(3)

        normal_count = failure_counts.get(0, 0)
        failure_count = failure_counts.get(1, 0)
        total_count = normal_count + failure_count

        col1.metric("正常样本数量", normal_count)
        col2.metric("故障样本数量", failure_count)

        if total_count > 0:
            col3.metric("故障样本占比", f"{failure_count / total_count:.2%}")

        st.bar_chart(failure_counts)

    else:
        st.error("数据中没有找到目标字段：Machine failure")

    st.header("5. 模型训练与评估")

    st.markdown("""
    点击下方按钮后，系统会完成数据预处理、多模型对比、随机森林模型训练、模型评估和模型保存。
    """)

    if st.button("训练随机森林模型"):
        try:
            with st.spinner("正在进行数据预处理..."):
                X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

            st.success("数据预处理完成。")

            col1, col2 = st.columns(2)

            col1.write("训练集大小：")
            col1.write(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

            col2.write("测试集大小：")
            col2.write(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

            with st.spinner("正在训练并对比多个模型..."):
                model_comparison_df = train_and_compare_models(
                    X_train,
                    X_test,
                    y_train,
                    y_test
                )

            st.subheader("多模型评估结果")
            st.dataframe(
                model_comparison_df.style.format({
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-score": "{:.4f}",
                })
            )

            with st.spinner("正在训练随机森林模型..."):
                model = train_model(X_train, y_train)

            st.success("模型训练完成。")

            with st.spinner("正在保存模型..."):
                save_model(model, feature_names)

            st.success("模型已保存到 models 文件夹。")

            with st.spinner("正在按当前阈值评估模型..."):
                metrics, cm, y_pred, _ = calculate_threshold_metrics(
                    model,
                    X_test,
                    y_test,
                    threshold=failure_threshold
                )
                report = get_classification_report_from_predictions(
                    y_test,
                    y_pred
                )

            st.header("6. Random Forest 最终模型评估指标")

            st.info(
                f"当前评估指标基于故障判定阈值 {failure_threshold:.2f} 计算："
                "测试集故障概率大于或等于该阈值时判定为故障。"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy 准确率", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision 精确率", f"{metrics['precision']:.4f}")
            col3.metric("Recall 召回率", f"{metrics['recall']:.4f}")
            col4.metric("F1-score", f"{metrics['f1']:.4f}")

            st.info(
                "在工业故障预测场景中，Recall 召回率很重要，"
                "因为漏报故障可能导致设备停机或生产损失。"
            )

            st.subheader("分类报告")

            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            st.subheader("混淆矩阵")

            fig, ax = plt.subplots()

            ax.imshow(cm)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred Normal", "Pred Failure"])
            ax.set_yticklabels(["True Normal", "True Failure"])

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center")

            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix")

            st.pyplot(fig)

            st.subheader("特征重要性")

            importance_df = get_feature_importance(model, feature_names)

            st.dataframe(importance_df)

            st.bar_chart(
                importance_df.set_index("feature")["importance"]
            )

        except Exception as e:
            st.error("模型训练或评估失败。")
            st.write("错误原因：")
            st.code(str(e))


st.header("7. 单条设备故障预测")

st.markdown("""
在这里可以手动输入一台设备的运行参数，系统会加载已保存的随机森林模型，
预测该设备是否存在故障风险，并给出故障概率、风险等级和维护建议。
""")

with st.form("single_prediction_form"):
    product_type = st.selectbox(
        "产品类型 Type",
        options=["L", "M", "H"],
        index=0
    )

    air_temperature = st.number_input(
        "空气温度 Air temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "工艺温度 Process temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "转速 Rotational speed [rpm]",
        min_value=0,
        max_value=5000,
        value=1500,
        step=10
    )

    torque = st.number_input(
        "扭矩 Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "工具磨损 Tool wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )

    submitted = st.form_submit_button("预测设备故障风险")


if submitted:
    try:
        model, feature_names = load_model()

        sample_df = build_single_sample(
            product_type=product_type,
            air_temperature=air_temperature,
            process_temperature=process_temperature,
            rotational_speed=rotational_speed,
            torque=torque,
            tool_wear=tool_wear,
            feature_names=feature_names
        )

        prediction, probability = predict_single_sample(model, sample_df)

        st.subheader("预测结果")

        if probability is not None:
            risk_level, risk_text = get_risk_level(probability)
            advice = get_maintenance_advice(risk_level)
            threshold_prediction = classify_failure_by_threshold(
                probability,
                failure_threshold
            )
        else:
            risk_level = "未知风险"
            risk_text = "模型未返回故障概率。"
            advice = "维护建议：请检查模型配置。"
            threshold_prediction = prediction

        st.write(f"当前故障判定阈值：{failure_threshold:.2f}")

        if threshold_prediction == 1:
            st.error("预测结果：设备存在故障风险")
        else:
            st.success("预测结果：暂未发现明显故障风险")

        st.metric("故障概率", f"{probability:.2%}" if probability is not None else "未知")

        if probability is not None:
            st.progress(float(probability))

        if risk_level == "低风险":
            st.success(f"风险等级：{risk_level}")
        elif risk_level == "中风险":
            st.warning(f"风险等级：{risk_level}")
        elif risk_level == "高风险":
            st.error(f"风险等级：{risk_level}")
        else:
            st.info(f"风险等级：{risk_level}")

        st.write(risk_text)

        st.subheader("维护建议")
        st.info(advice)

        st.subheader("输入样本特征")
        st.dataframe(sample_df)

    except Exception as e:
        st.error("单条预测失败。")
        st.write("错误原因：")
        st.code(str(e))
        st.warning("请先在上方点击“训练随机森林模型”，保存模型后再进行单条预测。")


st.header("8. 项目说明")

st.info("""
当前项目已经形成完整机器学习应用流程：

设备数据读取 → 数据预处理 → 模型训练 → 模型评估 → 模型保存 → 单条设备故障风险预测 → 风险等级与维护建议输出

该项目适合作为机器学习、数据分析、智能制造 AI 方向的实习项目展示。
""")
