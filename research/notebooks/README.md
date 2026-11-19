# Optional DistilBERT fine-tune (free GPU)

If `make train` validation F1 is below your target:

1. Export unified splits from `research/train.py` (same `seed=42`) or reuse parquet in `data/raw`.
2. Open Kaggle / Colab with GPU runtime.
3. Fine-tune `distilbert-base-uncased` for binary harmful classification on `text`.
4. Export weights to `models/distilbert/v1/` (TorchScript or ONNX optional).
5. Add a swap-in layer in `engine/transformer_layer.py` (not enabled by default).

No notebook is checked in to avoid stale dependencies; copy training hyperparameters from `models/lr/v1/metrics.json` for comparable splits.