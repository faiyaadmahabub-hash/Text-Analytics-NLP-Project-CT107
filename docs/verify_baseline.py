"""Rerun the original ComplementNB baseline, without its expensive grid search."""
from pathlib import Path
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
train = pd.read_csv(ROOT / "Data/train_set.csv")
test = pd.read_csv(ROOT / "Data/test_set.csv")
assert train.shape == (107768, 2) and test.shape == (26945, 2)
assert not train.isna().any().any() and not test.isna().any().any()
assert set(train.Sentiment) == set(test.Sentiment) == {"negative", "neutral", "positive"}
overlap = int(test.Summary_clean.isin(set(train.Summary_clean)).sum())
notebook = json.loads((ROOT / "Notebooks/Salsabila_PartB.ipynb").read_text(encoding="utf-8"))
source = next("".join(c["source"]) for c in notebook["cells"] if "nb_pipeline = Pipeline([" in "".join(c.get("source", [])))
scope = {"Pipeline": Pipeline, "TfidfVectorizer": TfidfVectorizer, "ComplementNB": ComplementNB,
         "X_train": train.Summary_clean, "y_train": train.Sentiment}
exec(compile(source, "Salsabila_PartB-baseline-cell", "exec"), scope)
prediction = scope["nb_pipeline"].predict(test.Summary_clean)
metrics = {"date": "2026-10-17", "python": platform.python_version(), "sklearn": sklearn.__version__,
           "model": "Original ComplementNB baseline cell; no tuning", "train_rows": len(train),
           "test_rows": len(test), "test_rows_with_text_seen_in_train": overlap,
           "accuracy": accuracy_score(test.Sentiment, prediction),
           "macro_f1": f1_score(test.Sentiment, prediction, average="macro"),
           "classification_report": classification_report(test.Sentiment, prediction, output_dict=True)}
assets = ROOT / "docs/assets"
assets.mkdir(parents=True, exist_ok=True)
(ROOT / "docs/baseline-results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(test.Sentiment, prediction, labels=["negative", "neutral", "positive"], ax=ax, cmap="Blues", colorbar=False)
ax.set_title("ComplementNB baseline: rerun 17 Oct 2026\nCommitted test split; repeated texts remain across splits")
fig.tight_layout()
fig.savefig(assets / "rerun-baseline-confusion.png", dpi=150)
plt.close(fig)
print(f"PASS: {len(train):,} training / {len(test):,} test rows; accuracy={metrics['accuracy']:.6f}; macro F1={metrics['macro_f1']:.6f}")
print(f"LIMITATION: {overlap:,}/{len(test):,} test rows have identical cleaned text in training ({overlap/len(test):.2%}).")
