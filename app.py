import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ml_core import (
    batch_predict,
    build_single_sample,
    calculate_threshold_metrics,
    classify_failure_by_threshold,
    get_classification_report_from_predictions,
    get_data_quality_report,
    get_feature_importance,
    get_maintenance_advice,
    get_risk_level,
    load_model,
    predict_single_sample,
    prepare_data,
    save_model,
    train_and_compare_models,
    train_model,
)


DATA_PATH = "data/ai4i2020.csv"


st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
)


def init_session_state():
    defaults = {
        "trained_model": None,
        "feature_names": None,
        "model_comparison_df": None,
        "X_train_shape": None,
        "X_test_shape": None,
        "y_train_shape": None,
        "y_test_shape": None,
        "X_test": None,
        "y_test": None,
        "feature_importance_df": None,
        "last_training_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_dataset(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "已从上传文件加载数据。"

    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH), f"已从本地加载数据：{DATA_PATH}"

    return None, "请上传 CSV 文件，或将 ai4i2020.csv 放入 data 文件夹。"


def draw_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")

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


def show_risk_message(risk_level, message):
    if risk_level == "低风险":
        st.success(message)
    elif risk_level == "中风险":
        st.warning(message)
    elif risk_level == "高风险":
        st.error(message)
    else:
        st.info(message)


init_session_state()


st.title("工业设备故障预测与风险评估系统")
st.caption("基于设备运行数据进行故障概率预测、风险等级评估和维护建议生成。")


with st.sidebar:
    st.header("控制面板")

    uploaded_file = st.file_uploader(
        "上传训练 / 分析数据集 CSV",
        type=["csv"],
        key="dataset_upload",
    )

    df, data_message = load_dataset(uploaded_file)

    if df is not None:
        st.success(data_message)
    else:
        st.warning(data_message)

    st.divider()

    failure_threshold = st.slider(
        "故障判定阈值",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    st.info(
        "阈值越低，模型越容易判定为故障，更偏向减少漏报；"
        "阈值越高，模型越谨慎，更偏向减少误报。"
    )

    train_clicked = st.button(
        "训练随机森林模型",
        type="primary",
        use_container_width=True,
        disabled=df is None,
    )

    st.divider()

    st.subheader("使用步骤")
    st.markdown(
        """
        1. 上传数据或使用本地 AI4I 数据集
        2. 查看数据质量与故障样本分布
        3. 设置故障判定阈值
        4. 训练模型并查看评估结果
        5. 执行单条或批量设备预测
        """
    )

    st.subheader("当前功能")
    st.markdown(
        """
        - 数据质量分析
        - 多模型对比
        - 阈值化模型评估
        - 混淆矩阵与特征重要性
        - 单条设备预测
        - 批量 CSV 预测与下载
        """
    )


if train_clicked and df is not None:
    try:
        with st.spinner("正在进行数据预处理与模型训练..."):
            X_train, X_test, y_train, y_test, feature_names = prepare_data(df)
            model_comparison_df = train_and_compare_models(
                X_train,
                X_test,
                y_train,
                y_test,
            )
            model = train_model(X_train, y_train)
            save_model(model, feature_names)
            feature_importance_df = get_feature_importance(model, feature_names)

        st.session_state.trained_model = model
        st.session_state.feature_names = feature_names
        st.session_state.model_comparison_df = model_comparison_df
        st.session_state.X_train_shape = X_train.shape
        st.session_state.X_test_shape = X_test.shape
        st.session_state.y_train_shape = y_train.shape
        st.session_state.y_test_shape = y_test.shape
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.feature_importance_df = feature_importance_df
        st.session_state.last_training_error = None
        st.success("模型训练完成，并已保存到 models 文件夹。")

    except Exception as e:
        st.session_state.last_training_error = str(e)
        st.error("模型训练失败。")
        st.code(str(e))


quality_tab, training_tab, evaluation_tab, importance_tab, single_tab, batch_tab = st.tabs(
    [
        "数据质量分析",
        "模型训练与对比",
        "模型评估结果",
        "特征重要性分析",
        "单条设备预测",
        "批量预测与结果下载",
    ]
)


with quality_tab:
    st.subheader("数据概览")

    if df is None:
        st.warning("暂无可用数据，请在侧边栏上传 CSV 文件或准备本地数据集。")
    else:
        quality_report = get_data_quality_report(df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("数据行数", quality_report["shape"][0])
        col2.metric("数据列数", quality_report["shape"][1])

        target_distribution = quality_report["target_distribution"]
        normal_count = target_distribution.get(0, 0)
        failure_count = target_distribution.get(1, 0)
        col3.metric("故障样本数", failure_count)

        if quality_report["failure_rate"] is not None:
            col4.metric("故障样本占比", f"{quality_report['failure_rate']:.2%}")
        else:
            col4.metric("故障样本占比", "N/A")

        st.markdown("#### 数据预览")
        st.dataframe(df.head(), use_container_width=True)

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("#### 字段类型")
            dtype_df = pd.DataFrame(
                quality_report["dtypes"].items(),
                columns=["字段名称", "字段类型"],
            )
            st.dataframe(dtype_df, use_container_width=True)

        with right_col:
            st.markdown("#### 缺失值统计")
            missing_df = pd.DataFrame(
                quality_report["missing_values"].items(),
                columns=["字段名称", "缺失值数量"],
            )
            st.dataframe(missing_df, use_container_width=True)

        if target_distribution:
            st.markdown("#### Machine failure 类别分布")
            distribution_df = pd.DataFrame(
                target_distribution.items(),
                columns=["Machine failure", "样本数量"],
            )
            st.bar_chart(distribution_df.set_index("Machine failure")["样本数量"])
        else:
            st.warning("数据中没有找到目标字段：Machine failure")


with training_tab:
    st.subheader("模型训练与多模型对比")

    if st.session_state.last_training_error:
        st.error("最近一次训练失败。")
        st.code(st.session_state.last_training_error)

    if st.session_state.model_comparison_df is None:
        st.info("请在侧边栏点击“训练随机森林模型”，完成数据预处理、多模型对比和随机森林训练。")
    else:
        col1, col2 = st.columns(2)
        col1.metric("训练集 X", str(st.session_state.X_train_shape))
        col1.metric("训练集 y", str(st.session_state.y_train_shape))
        col2.metric("测试集 X", str(st.session_state.X_test_shape))
        col2.metric("测试集 y", str(st.session_state.y_test_shape))

        st.markdown("#### 多模型评估结果")
        st.dataframe(
            st.session_state.model_comparison_df.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-score": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

        st.success("Random Forest 已作为最终模型保存，用于评估、特征重要性和预测。")


with evaluation_tab:
    st.subheader("基于当前阈值的 Random Forest 评估")

    if st.session_state.trained_model is None:
        st.info("请先在侧边栏训练模型。评估指标会根据当前故障判定阈值实时计算。")
    else:
        metrics, cm, y_pred, _ = calculate_threshold_metrics(
            st.session_state.trained_model,
            st.session_state.X_test,
            st.session_state.y_test,
            threshold=failure_threshold,
        )
        report = get_classification_report_from_predictions(
            st.session_state.y_test,
            y_pred,
        )

        st.info(
            f"当前指标基于故障判定阈值 {failure_threshold:.2f} 计算。"
            "测试集故障概率大于或等于该阈值时判定为故障。"
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("Precision", f"{metrics['precision']:.4f}")
        col3.metric("Recall", f"{metrics['recall']:.4f}")
        col4.metric("F1-score", f"{metrics['f1']:.4f}")

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.markdown("#### 混淆矩阵")
            draw_confusion_matrix(cm)

        with right_col:
            st.markdown("#### 分类报告")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)


with importance_tab:
    st.subheader("特征重要性分析")

    if st.session_state.feature_importance_df is None:
        st.info("请先在侧边栏训练模型，系统将展示 Random Forest 的特征重要性。")
    else:
        importance_df = st.session_state.feature_importance_df

        st.dataframe(importance_df, use_container_width=True)
        st.bar_chart(importance_df.set_index("feature")["importance"])


with single_tab:
    st.subheader("单条设备故障预测")

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            product_type = st.selectbox("产品类型 Type", options=["L", "M", "H"], index=0)
            air_temperature = st.number_input(
                "空气温度 Air temperature [K]",
                min_value=250.0,
                max_value=350.0,
                value=300.0,
                step=0.1,
            )

        with col2:
            process_temperature = st.number_input(
                "工艺温度 Process temperature [K]",
                min_value=250.0,
                max_value=400.0,
                value=310.0,
                step=0.1,
            )
            rotational_speed = st.number_input(
                "转速 Rotational speed [rpm]",
                min_value=0,
                max_value=5000,
                value=1500,
                step=10,
            )

        with col3:
            torque = st.number_input(
                "扭矩 Torque [Nm]",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=0.1,
            )
            tool_wear = st.number_input(
                "工具磨损 Tool wear [min]",
                min_value=0,
                max_value=300,
                value=100,
                step=1,
            )

        submitted = st.form_submit_button("预测设备故障风险", use_container_width=True)

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
                feature_names=feature_names,
            )

            _, probability = predict_single_sample(model, sample_df)

            if probability is None:
                risk_level = "未知风险"
                risk_text = "模型未返回故障概率。"
                advice = "维护建议：请检查模型配置。"
                threshold_prediction = 0
            else:
                risk_level, risk_text = get_risk_level(probability)
                advice = get_maintenance_advice(risk_level)
                threshold_prediction = classify_failure_by_threshold(
                    probability,
                    failure_threshold,
                )

            result_text = (
                "存在故障风险"
                if threshold_prediction == 1
                else "暂未发现明显故障风险"
            )

            st.markdown("#### 预测结果")
            col1, col2, col3 = st.columns(3)
            col1.metric("故障概率", f"{probability:.2%}" if probability is not None else "未知")
            col2.metric("故障判定阈值", f"{failure_threshold:.2f}")
            col3.metric("风险等级", risk_level)

            if threshold_prediction == 1:
                st.error(f"预测结果：{result_text}")
            else:
                st.success(f"预测结果：{result_text}")

            show_risk_message(risk_level, f"风险解释：{risk_text}")
            st.info(advice)

            st.markdown("#### 模型输入特征")
            st.dataframe(sample_df, use_container_width=True)

        except Exception as e:
            st.error("单条预测失败。")
            st.code(str(e))
            st.warning("请先在侧边栏训练并保存模型，再进行单条预测。")


with batch_tab:
    st.subheader("批量 CSV 预测与结果下载")

    batch_file = st.file_uploader(
        "上传需要批量预测的设备数据 CSV 文件",
        type=["csv"],
        key="batch_prediction_upload",
    )

    if batch_file is None:
        st.info("上传 CSV 后，系统将输出每台设备的故障概率、预测结果、风险等级和维护建议。")
    else:
        try:
            batch_df = pd.read_csv(batch_file)
            model, feature_names = load_model()

            with st.spinner("正在进行批量预测..."):
                batch_result_df = batch_predict(
                    batch_df,
                    model,
                    feature_names,
                    threshold=failure_threshold,
                )

            st.success("批量预测完成。")
            st.write(f"当前故障判定阈值：{failure_threshold:.2f}")
            st.dataframe(batch_result_df, use_container_width=True)

            csv_data = batch_result_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="下载批量预测结果 CSV",
                data=csv_data,
                file_name="batch_prediction_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as e:
            st.error("批量预测失败。")
            st.code(str(e))
            st.warning("请确认已训练并保存随机森林模型，且上传 CSV 包含设备运行特征字段。")
