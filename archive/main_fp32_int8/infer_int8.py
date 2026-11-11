"""PC-side inference for the QAT INT8 artefact.  numpy only, no torch needed.

    python infer_int8.py model/<dir>            # full-dataset recompute
    python infer_int8.py model/<dir> --compare  # also report the FP32 checkpoint

The arithmetic mirrors what train.py's fake quantisation simulated: int8
per-channel weights, uint8 asymmetric activations at the four operator
boundaries, everything else in float.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import joblib
from math import erf, sqrt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def gelu(x):
    from scipy.special import erf as _erf
    return 0.5 * x * (1.0 + _erf(x / sqrt(2.0)))


def relu(x):
    return np.maximum(x, 0.0)


class Int8Model:
    def __init__(self, directory: Path):
        self.d = dict(np.load(directory / "model_int8.npz"))
        self.meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.scaler = joblib.load(directory / "scaler.pkl")

    def _w(self, name):
        return self.d[f"{name}_w_int"].astype(np.float64) * self.d[f"{name}_w_scale"].reshape(-1, 1)

    def _q(self, name, value):
        s = float(self.d[f"act_{name}_scale"]); zp = float(self.d[f"act_{name}_zero_point"])
        levels = int(self.d[f"act_{name}_levels"])
        return (np.clip(np.round(value / s) + zp, 0, levels) - zp) * s

    def standardise(self, features):
        return (np.asarray(features, dtype=np.float64) - self.scaler["mean"]) / self.scaler["scale"]

    def logits(self, standardised):
        x = self._q("input", standardised)
        h0 = self._q("stem", gelu(x @ self._w("stem").T + self.d["stem_b"]))
        a1 = self._q("inner", relu(h0 @ self._w("res1").T + self.d["res1_b"]))
        feat = self._q("feature", relu(h0 + (a1 @ self._w("res2").T + self.d["res2_b"])))
        w = self._w("head")
        alpha = float(self.d["alpha"]); scale = float(self.d["head_scale"])
        linear = feat @ w.T + self.d["head_b"]
        fn = np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
        wn = np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
        angular = scale * ((feat / fn) @ (w / wn).T)
        return (1.0 - alpha) * linear + alpha * angular

    def predict(self, standardised, batch=16384):
        out = [self.logits(standardised[s:s + batch]).argmax(1) for s in range(0, len(standardised), batch)]
        return np.concatenate(out)


def main():
    from iatopt.core import load_dataset, build_features, metrics
    directory = Path(sys.argv[1])
    model = Int8Model(directory)
    bundle = load_dataset(ROOT.parent / "esp32p4_iat_demo/input/data/x_train.csv",
                          ROOT.parent / "esp32p4_iat_demo/input/data/y_train.csv")
    x = model.standardise(build_features(bundle.raw, model.meta["feature_spec"]))
    prediction = model.predict(x)
    m = metrics(bundle.labels, prediction)
    print(f"INT8 全量复算  acc={m['accuracy']*100:.4f}%  bal={m['balanced_accuracy']*100:.4f}%  "
          f"F1={m['macro_f1']*100:.4f}%  最差类={m['minimum_class_recall']*100:.4f}%")
    weight_bytes = sum(v.size for k, v in model.d.items() if k.endswith("_w_int"))
    bias_bytes = 4 * sum(v.size for k, v in model.d.items() if k.endswith("_b"))
    scale_bytes = 4 * sum(v.size for k, v in model.d.items() if k.endswith("_w_scale"))
    print(f"INT8 权重 {weight_bytes} B + 偏置 {bias_bytes} B + 尺度 {scale_bytes} B "
          f"= {(weight_bytes+bias_bytes+scale_bytes)/1024:.3f} KiB   "
          f"(FP32 检查点 {model.meta['state_dict_kib']:.3f} KiB)")
    if "--compare" in sys.argv:
        import torch
        from iatopt.core import IATMLP, predict_logits
        net = IATMLP(model.meta["input_dim"], model.meta["architecture"], dropout=model.meta["dropout"],
                     head_scale=model.meta["head_scale"], hybrid_alpha=model.meta["hybrid_alpha_init"])
        state = torch.load(directory / "model_fp32.pth", map_location="cpu")
        state = {k: v for k, v in state.items() if not k.startswith("activation_observers")}
        # drop the quantiser's own state (LSQ carries log_step/offset/ready/frozen; the
        # min/max quantiser carries nothing), keeping only the underlying float weight
        state = {k: v for k, v in state.items()
                 if ".parametrizations." not in k or k.endswith("parametrizations.weight.original")}
        state = {k.replace("parametrizations.weight.original", "weight"): v for k, v in state.items()}
        net.load_state_dict(state); net.eval()
        fp = predict_logits(net, x.astype(np.float32)).argmax(1)
        mf = metrics(bundle.labels, fp)
        print(f"FP32 权重同一模型 acc={mf['accuracy']*100:.4f}%   "
              f"INT8 - FP32 = {(m['accuracy']-mf['accuracy'])*100:+.4f} pp   "
              f"预测一致率 {float((prediction==fp).mean())*100:.4f}%")


if __name__ == "__main__":
    main()
