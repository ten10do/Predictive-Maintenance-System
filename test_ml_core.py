import unittest

import pandas as pd

from ml_core import get_data_quality_report, prepare_data, train_and_compare_models


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


if __name__ == "__main__":
    unittest.main()
