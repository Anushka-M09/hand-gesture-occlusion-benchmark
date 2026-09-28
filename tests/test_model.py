import unittest
import numpy as np
import tempfile
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class TestModelBundle(unittest.TestCase):
    def test_model_bundle_structure(self):
        # Create mock 63-d normalized features
        np.random.seed(42)
        X = np.random.uniform(-1.0, 1.0, (20, 63)).astype(np.float32)
        y = np.array(["fist", "peace"] * 10)

        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y_enc)

        bundle = {
            "model": clf,
            "label_encoder": encoder,
            "classes": encoder.classes_.tolist(),
            "model_type": "random_forest",
            "feature_dim": 63,
            "baseline_accuracy": 1.0
        }

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            joblib.dump(bundle, tmp_path)
            loaded = joblib.load(tmp_path)
            self.assertEqual(loaded["feature_dim"], 63)
            self.assertEqual(loaded["model_type"], "random_forest")
            self.assertEqual(loaded["classes"], ["fist", "peace"])
            preds = loaded["model"].predict(X[:2])
            decoded = loaded["label_encoder"].inverse_transform(preds)
            self.assertEqual(len(decoded), 2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()
