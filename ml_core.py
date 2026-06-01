import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")


def prepare_data(df: pd.DataFrame):
    """
    数据预处理函数。

    输入：
        df: 原始设备运行数据

    输出：
        X_train, X_test, y_train, y_test, feature_names
    """

    data = df.copy()

    # 删除无效编号字段
    drop_cols = ["UDI", "Product ID"]

    for col in drop_cols:
        if col in data.columns:
            data = data.drop(columns=[col])

    # 目标字段
    target_col = "Machine failure"

    if target_col not in data.columns:
        raise ValueError("数据中没有找到目标字段：Machine failure")

    # 类别特征编码
    if "Type" in data.columns:
        data = pd.get_dummies(data, columns=["Type"], drop_first=False)

    # 特征和标签
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # 只保留数值型和布尔型特征
    X = X.select_dtypes(include=[np.number, "bool"])

    feature_names = X.columns.tolist()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names


def train_model(X_train, y_train):
    """
    训练随机森林模型。
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    评估模型效果。
    """

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
        output_dict=True
    )

    cm = confusion_matrix(y_test, y_pred)

    return metrics, report, cm, y_pred


def get_feature_importance(model, feature_names):
    """
    获取随机森林模型的特征重要性。
    """

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return importance_df


def save_model(model, feature_names):
    """
    保存训练好的模型和特征名。
    """

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, FEATURE_PATH)


def load_model():
    """
    加载已经保存的模型和特征名。
    """

    if not os.path.exists(MODEL_PATH):
        raise ValueError("没有找到已保存的模型，请先训练模型。")

    if not os.path.exists(FEATURE_PATH):
        raise ValueError("没有找到特征名文件，请先训练模型。")

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    return model, feature_names


def build_single_sample(
    product_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    feature_names
):
    """
    构造单条设备样本。

    输入的是用户在页面上填写的设备参数。
    输出的是和训练集特征结构一致的一行 DataFrame。
    """

    sample = {
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_H": 0,
        "Type_L": 0,
        "Type_M": 0,
    }

    type_col = f"Type_{product_type}"

    if type_col in sample:
        sample[type_col] = 1

    sample_df = pd.DataFrame([sample])

    # 确保单条样本的列和训练时完全一致
    for col in feature_names:
        if col not in sample_df.columns:
            sample_df[col] = 0

    sample_df = sample_df[feature_names]

    return sample_df


def predict_single_sample(model, sample_df):
    """
    对单条设备数据进行故障预测。

    输出：
        prediction: 0 或 1
        failure_probability: 故障概率
    """

    prediction = model.predict(sample_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(sample_df)[0][1]
    else:
        probability = None

    return prediction, probability


def get_risk_level(probability):
    """
    根据故障概率判断风险等级。

    参数：
        probability: 故障概率，范围 0~1

    返回：
        risk_level: 风险等级
        risk_text: 风险解释
    """

    if probability is None:
        return "未知风险", "当前模型无法输出故障概率。"

    if probability < 0.3:
        return "低风险", "设备当前运行状态较稳定，可继续正常运行。"

    elif probability < 0.7:
        return "中风险", "设备存在一定故障风险，建议加强巡检并关注关键运行参数。"

    else:
        return "高风险", "设备故障风险较高，建议尽快安排检修或停机检查。"


def get_maintenance_advice(risk_level):
    """
    根据风险等级给出维护建议。
    """

    if risk_level == "低风险":
        return (
            "维护建议：设备当前风险较低，可继续运行。"
            "建议保持常规巡检，定期记录温度、转速、扭矩和工具磨损等参数。"
        )

    elif risk_level == "中风险":
        return (
            "维护建议：设备存在一定风险。"
            "建议重点关注空气温度、工艺温度、转速、扭矩和工具磨损变化，"
            "适当增加巡检频率，并准备维护计划。"
        )

    elif risk_level == "高风险":
        return (
            "维护建议：设备故障风险较高。"
            "建议尽快安排停机检查，重点排查工具磨损、负载过大、转速异常、"
            "温度异常等问题，避免发生生产中断。"
        )

    else:
        return "维护建议：当前风险等级未知，请检查模型输出。"