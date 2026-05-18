import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
models_dir = BASE_DIR / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

stats = {
    "available": True,
    "model_type": "Offline Demo Classifier",
    "task": "Binary Text Classification",
    "max_sequence_length": 256,
    "classes": ["fake_news", "real_news"],
    "accuracy": 0.87,
    "precision": 0.88,
    "recall": 0.85,
    "f1_score": 0.865,
    "roc_auc": 0.92,
    "test_samples": 1200,
    "confusion_matrix": [[520, 80], [100, 500]],
    "message": "Demo metrics generated for presentation purposes."
}

out_path = models_dir / 'model_stats.pkl'
joblib.dump(stats, out_path)
print(f"Wrote demo stats to {out_path}")
