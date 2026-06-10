import unittest

import numpy as np
import pandas as pd

from ml_core import (
    batch_predict,
    calculate_threshold_metrics,
    classify_failure_by_threshold,
    get_classification_report_from_predictions,
    get_data_quality_report,
    get_maintenance_advice,
    get_risk_level,
    prepare_prediction_data,
    prepare_data,
    train_and_compare_models,
)


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.array(probabilities)

    def predict_proba(self, X):
        failure_probs = self.probabilities[:len(X)]
        return np.column_stack([1 - failure_probs, failure_probs])


class MlCoreUpgradeTests(unittest.TestCase):
    def setUp(self):
        rows = 12
        self.df = pd.DataFrame(
            {
                "UDI": list(range(1, rows + 1)),
                "Product ID": [f"PID-{i}" for i in range(1, rows + 1)],
                "Type": ["L", "M", "H"] * 4,
                "Air temperature [K]": [300.0 + i for i in range(rows)],
                "Process temperature [K]": [
                    310.0,
                    311.0,
                    312.0,
                    313.0,
                    314.0,
                    None,
                    316.0,
                    317.0,
                    318.0,
                    319.0,
                    320.0,
                    321.0,
                ],
                "Rotational speed [rpm]": [1400 + i * 20 for i in range(rows)],
                "Torque [Nm]": [35.0 + i for i in range(rows)],
                "Tool wear [min]": [10 + i * 5 for i in range(rows)],
                "Machine failure": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            }
        )

    def test_data_quality_report_contains_expected_sections(self):
        report = get_data_quality_report(self.df)

        self.assertEqual(report["shape"], (12, 9))
        self.assertIn("Type", report["dtypes"])
        self.assertEqual(report["missing_values"]["Process temperature [K]"], 1)
        self.assertEqual(report["target_distribution"][0], 6)
        self.assertEqual(report["target_distribution"][1], 6)
        self.assertEqual(report["failure_rate"], 0.5)

    def test_train_and_compare_models_returns_metrics_for_three_models(self):
        clean_df = self.df.dropna()
        X_train, X_test, y_train, y_test, _ = prepare_data(clean_df)

        results_df = train_and_compare_models(X_train, X_test, y_train, y_test)

        self.assertEqual(
            results_df["Model"].tolist(),
            ["Logistic Regression", "Decision Tree", "Random Forest"],
        )
        self.assertEqual(
            results_df.columns.tolist(),
            ["Model", "Accuracy", "Precision", "Recall", "F1-score"],
        )
        self.assertEqual(len(results_df), 3)

    def test_calculate_threshold_metrics_uses_failure_probability_threshold(self):
        model = FixedProbabilityModel([0.2, 0.4, 0.8, 0.9])
        X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})
        y_test = pd.Series([0, 1, 1, 0])

        metrics, cm, y_pred, probabilities = calculate_threshold_metrics(
            model,
            X_test,
            y_test,
            threshold=0.7,
        )

        self.assertEqual(y_pred.tolist(), [0, 0, 1, 1])
        self.assertEqual(probabilities.tolist(), [0.2, 0.4, 0.8, 0.9])
        self.assertEqual(cm.tolist(), [[1, 1], [1, 1]])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_classify_failure_by_threshold_uses_selected_threshold(self):
        self.assertEqual(classify_failure_by_threshold(0.65, 0.7), 0)
        self.assertEqual(classify_failure_by_threshold(0.70, 0.7), 1)

    def test_classification_report_can_be_built_from_threshold_predictions(self):
        report = get_classification_report_from_predictions(
            pd.Series([0, 1, 1, 0]),
            np.array([0, 0, 1, 1]),
        )

        self.assertIn("0", report)
        self.assertIn("1", report)
        self.assertIn("accuracy", report)

    def test_risk_level_and_maintenance_advice_use_updated_rules(self):
        low_level, low_text = get_risk_level(0.29)
        medium_level, medium_text = get_risk_level(0.30)
        high_level, high_text = get_risk_level(0.70)

        self.assertEqual(low_level, "低风险")
        self.assertIn("较稳定", low_text)
        self.assertEqual(medium_level, "中风险")
        self.assertIn("潜在异常", medium_text)
        self.assertEqual(high_level, "高风险")
        self.assertIn("故障概率较高", high_text)

        self.assertIn("常规巡检", get_maintenance_advice(low_level))
        self.assertIn("转速、扭矩、温度和工具磨损", get_maintenance_advice(medium_level))
        self.assertIn("尽快停机检查或安排维护", get_maintenance_advice(high_level))

    def test_prepare_prediction_data_aligns_columns_to_training_features(self):
        prediction_df = pd.DataFrame(
            {
                "UDI": [101],
                "Product ID": ["P-101"],
                "Type": ["M"],
                "Air temperature [K]": [300.0],
                "Process temperature [K]": [310.0],
                "Rotational speed [rpm]": [1500],
                "Torque [Nm]": [40.0],
                "Machine failure": [1],
                "extra_column": [999],
            }
        )
        feature_names = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Type_H",
            "Type_L",
            "Type_M",
        ]

        prepared_df = prepare_prediction_data(prediction_df, feature_names)

        self.assertEqual(prepared_df.columns.tolist(), feature_names)
        self.assertEqual(prepared_df.loc[0, "Tool wear [min]"], 0)
        self.assertEqual(prepared_df.loc[0, "Type_M"], 1)
        self.assertEqual(prepared_df.loc[0, "Type_H"], 0)
        self.assertNotIn("UDI", prepared_df.columns)
        self.assertNotIn("Product ID", prepared_df.columns)
        self.assertNotIn("Machine failure", prepared_df.columns)
        self.assertNotIn("extra_column", prepared_df.columns)

    def test_batch_predict_returns_original_fields_and_threshold_results(self):
        batch_df = pd.DataFrame(
            {
                "UDI": [101, 102],
                "Product ID": ["P-101", "P-102"],
                "Type": ["L", "H"],
                "Air temperature [K]": [300.0, 320.0],
                "Process temperature [K]": [310.0, 330.0],
                "Rotational speed [rpm]": [1500, 2100],
                "Torque [Nm]": [40.0, 70.0],
                "Tool wear [min]": [100, 220],
                "Machine failure": [0, 1],
            }
        )
        feature_names = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Type_H",
            "Type_L",
            "Type_M",
        ]
        model = FixedProbabilityModel([0.25, 0.75])

        result_df = batch_predict(batch_df, model, feature_names, threshold=0.5)

        self.assertIn("UDI", result_df.columns)
        self.assertIn("Product ID", result_df.columns)
        self.assertIn("failure_probability", result_df.columns)
        self.assertIn("prediction_result", result_df.columns)
        self.assertIn("risk_level", result_df.columns)
        self.assertIn("maintenance_advice", result_df.columns)
        self.assertEqual(result_df["failure_probability"].tolist(), [0.25, 0.75])
        self.assertEqual(
            result_df["prediction_result"].tolist(),
            ["暂未发现明显故障风险", "存在故障风险"],
        )
        self.assertEqual(result_df["risk_level"].tolist(), ["低风险", "高风险"])
        self.assertIn("常规巡检", result_df.loc[0, "maintenance_advice"])
        self.assertIn("尽快停机检查或安排维护", result_df.loc[1, "maintenance_advice"])


if __name__ == "__main__":
    unittest.main()
