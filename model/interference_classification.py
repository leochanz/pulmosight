import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
import pydicom
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


# =========================================================
# 1) 改呢幾個 path
# =========================================================
# CKPT_PATH = r"C:\Users\hkpuadmin\Desktop\CTM\fine-tune\CLASSIFICATION\TRIAL1\best_model_fold5.pth"
# LOCAL_DIR = r"D:\hf_models\medsiglip-448"   # 你訓練時用嘅 HF backbone folder
# DICOM_PATH = r"C:\Users\hkpuadmin\Desktop\your_selected_ct_image.dcm"  # 你自己揀嘅 CT slice

CKPT_PATH = r"/Users/leochan/Downloads/ctm/backend/best_model_fold.pth"   # <-- 你訓練好嘅 classification model checkpoint
LOCAL_DIR = r"/Users/leochan/Downloads/ctm/backend"   # 你訓練時用嘅 HF backbone folder
DICOM_PATH = r"/Users/leochan/Downloads/ctm/backend/test.dcm"  # 你自己揀嘅 CT slice

THRESHOLD = 0.7


# =========================================================
# 2) 跟返你原本 training script 嘅 function
# =========================================================
def window_and_norm(hu: np.ndarray, win=(-1000, 400)) -> np.ndarray:
    lo, hi = win
    x = np.clip(hu, lo, hi)
    x = (x - lo) / (hi - lo)
    return x.astype(np.float32)

def safe_read_dicom_hu(dcm_path: Path, verbose: bool = False) -> Optional[np.ndarray]:
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        return arr * slope + intercept
    except Exception as e:
        if verbose:
            print(f"[skip] bad dicom: {dcm_path} | {type(e).__name__}: {e}")
        return None

def normalize_to_tensor(img01: np.ndarray, mean: List[float], std: List[float]) -> torch.Tensor:
    x = np.stack([img01, img01, img01], axis=0).astype(np.float32)
    m = np.array(mean, dtype=np.float32)[:, None, None]
    s = np.array(std, dtype=np.float32)[:, None, None]
    x = (x - m) / s
    return torch.from_numpy(x)

def make_sliding_coords(H: int, W: int, win: int, stride: int):
    ys = list(range(0, max(1, H - win + 1), stride))
    xs = list(range(0, max(1, W - win + 1), stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if ys[-1] != max(0, H - win):
        ys.append(max(0, H - win))
    if xs[-1] != max(0, W - win):
        xs.append(max(0, W - win))
    return ys, xs

def _pad_if_needed(img: np.ndarray, top: int, left: int, win: int):
    H, W = img.shape
    bottom = top + win
    right = left + win
    pad_t = max(0, -top)
    pad_l = max(0, -left)
    pad_b = max(0, bottom - H)
    pad_r = max(0, right - W)

    if pad_t or pad_l or pad_b or pad_r:
        img = np.pad(
            img,
            ((pad_t, pad_b), (pad_l, pad_r)),
            mode="constant",
            constant_values=0.0
        )
        top += pad_t
        left += pad_l

    return img, top, left

def pool_scores(scores: List[float], mode: str = "topk_mean", topk: int = 2) -> float:
    if not scores:
        return 0.0
    if mode == "mean":
        return float(np.mean(scores))
    if mode == "max":
        return float(np.max(scores))
    if mode == "topk_mean":
        k = min(int(topk), len(scores))
        s = np.sort(np.array(scores, dtype=np.float32))[::-1]
        return float(np.mean(s[:k]))
    raise ValueError(f"Unknown pool mode: {mode}")


# =========================================================
# 3) model structure
# =========================================================
def get_image_embedding(backbone, pixel_values: torch.Tensor) -> torch.Tensor:
    if hasattr(backbone, "get_image_features"):
        try:
            out = backbone.get_image_features(pixel_values=pixel_values)
            if isinstance(out, torch.Tensor):
                return out
        except TypeError:
            out = backbone.get_image_features(pixel_values)
            if isinstance(out, torch.Tensor):
                return out
        except Exception:
            pass

    if hasattr(backbone, "vision_model"):
        vout = backbone.vision_model(pixel_values=pixel_values, return_dict=True)
        if hasattr(vout, "pooler_output") and isinstance(vout.pooler_output, torch.Tensor):
            z = vout.pooler_output
        else:
            z = vout.last_hidden_state[:, 0, :]

        for proj_name in ["vision_projection", "visual_projection", "image_projection"]:
            if hasattr(backbone, proj_name):
                proj = getattr(backbone, proj_name)
                if isinstance(proj, nn.Module):
                    z = proj(z)
                    break
        return z

    raise RuntimeError("Cannot extract image embedding safely.")

class BackboneClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, emb_dim: int, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        z = get_image_embedding(self.backbone, pixel_values)
        return self.head(z)


# =========================================================
# 4) load checkpoint
# =========================================================
def load_model(ckpt_path: str, local_dir: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta", {})

    win = int(meta.get("win", 448))
    stride_val = int(meta.get("stride_val", 224))
    pool_mode = meta.get("pool", "topk_mean")
    topk = int(meta.get("topk", 2))
    emb_dim = int(meta["emb_dim"]) if "emb_dim" in meta else None

    processor = AutoImageProcessor.from_pretrained(local_dir, local_files_only=True, use_fast=False)
    mean = meta.get("mean", getattr(processor, "image_mean", [0.5, 0.5, 0.5]))
    std = meta.get("std", getattr(processor, "image_std", [0.5, 0.5, 0.5]))

    try:
        backbone = AutoModel.from_pretrained(local_dir, local_files_only=True, dtype=torch.float32)
    except TypeError:
        backbone = AutoModel.from_pretrained(local_dir, local_files_only=True, torch_dtype=torch.float32)

    backbone = backbone.to(device)

    if emb_dim is None:
        with torch.no_grad():
            dummy = torch.randn(1, 3, win, win, device=device, dtype=torch.float32)
            emb = get_image_embedding(backbone, dummy)
            emb_dim = int(emb.shape[-1])

    model = BackboneClassifier(backbone=backbone, emb_dim=emb_dim, num_classes=2).to(device)

    # 優先 load 成個 model
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        # fallback
        if "backbone" in ckpt:
            model.backbone.load_state_dict(ckpt["backbone"], strict=False)
        if "head" in ckpt:
            model.head.load_state_dict(ckpt["head"], strict=True)

    model.eval()

    return model, mean, std, win, stride_val, pool_mode, topk, meta


# =========================================================
# 5) 單張 CT slice inference
# =========================================================
@torch.no_grad()
def infer_one_dicom(
    model,
    dcm_path: str,
    device: torch.device,
    mean: List[float],
    std: List[float],
    win: int,
    stride: int,
    pool_mode: str,
    topk: int,
    threshold: float = 0.7,
):
    hu = safe_read_dicom_hu(Path(dcm_path), verbose=True)
    if hu is None:
        raise RuntimeError(f"Cannot read dicom: {dcm_path}")

    img01 = window_and_norm(hu, win=(-1000, 400))
    H, W = img01.shape

    ys, xs = make_sliding_coords(H, W, win, stride)

    window_probs = []
    window_infos = []

    for wy in ys:
        for wx in xs:
            img_pad, yy, xx = _pad_if_needed(img01, wy, wx, win)
            patch = img_pad[yy:yy + win, xx:xx + win]

            x = normalize_to_tensor(patch, mean, std).unsqueeze(0).to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0, 1].item()   # class 1 = malignant

            window_probs.append(prob)
            window_infos.append((wy, wx, prob))

    pooled_score = pool_scores(window_probs, mode=pool_mode, topk=topk)
    pred_label = 1 if pooled_score >= threshold else 0
    pred_name = "malignant" if pred_label == 1 else "benign"

    # 排序睇最危險幾個 window
    window_infos = sorted(window_infos, key=lambda x: x[2], reverse=True)

    return {
        "dicom_path": dcm_path,
        "image_shape": (H, W),
        "n_windows": len(window_probs),
        "threshold": threshold,
        "pooled_score": pooled_score,
        "pred_label": pred_label,
        "pred_name": pred_name,
        "top_windows": window_infos[:10],
    }


# =========================================================
# 6) main
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    model, mean, std, win, stride_val, pool_mode, topk, meta = load_model(
        ckpt_path=CKPT_PATH,
        local_dir=LOCAL_DIR,
        device=device
    )

    result = infer_one_dicom(
        model=model,
        dcm_path=DICOM_PATH,
        device=device,
        mean=mean,
        std=std,
        win=win,
        stride=stride_val,
        pool_mode=pool_mode,
        topk=topk,
        threshold=THRESHOLD,
    )

    print("\n================ RESULT ================")
    print(f"DICOM path        : {result['dicom_path']}")
    print(f"Image shape       : {result['image_shape']}")
    print(f"Num windows       : {result['n_windows']}")
    print(f"Threshold         : {result['threshold']}")
    print(f"Pooled score      : {result['pooled_score']:.6f}")
    print(f"Prediction label  : {result['pred_label']}")
    print(f"Prediction class  : {result['pred_name']}")

    print("\nTop suspicious windows (wy, wx, p_malignant):")
    for i, (wy, wx, p) in enumerate(result["top_windows"], 1):
        print(f"  {i:02d}. ({wy:4d}, {wx:4d}) -> {p:.6f}")

    # 額外顯示 training meta
    print("\n================ CKPT META ================")
    for k, v in meta.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()