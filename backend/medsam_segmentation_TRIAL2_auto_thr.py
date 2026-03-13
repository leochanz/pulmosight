# medsam_segmentation.py
# ============================================================
# LIDC-IDRI 5-Fold CV (SEGMENTATION)
# - ✅ Using: MedSAM (SAM ViT) image encoder as backbone
# - ✅ Freeze whole MedSAM image encoder, except last N transformer blocks (default N=2)
# - ✅ Add: lightweight UNet-ish upsampling head to predict 1-channel mask logits
# - ✅ ROI curriculum (roi_prob: start -> end) for TRAIN
# - ✅ TRAIN: pos/neg window intent balance (deterministic option)
# - ✅ VAL: full sliding windows (win/stride)
# - ✅ Automatic threshold sweep (SEGMENTATION / DICE) on VAL via histograms
# - ✅ Anti-crash checkpointing:
#       - save "last" every epoch (and optionally every N train steps)
#       - resume from checkpoint with optimizer/scheduler/best_dice/bad_epochs/roi_prob/RNG state
#
# Install deps:
#   python -m pip install pydicom opencv-python matplotlib tqdm
#   python -m pip install segment-anything-py
#
# Run (example):
#   python medsam_segmentation.py --root_dir "C:\Users\hkpuadmin\Desktop\dataset\CT" --medsam_ckpt "C:\Users\hkpuadmin\weights\medsam_vit_b.pth" --sam_type vit_b
#
# Resume (example):
#   python medsam_segmentation.py --resume "C:\Users\hkpuadmin\Desktop\CTM\fine-tune\SEGMENTATION\medsam_seg_fold1_last.pt"
# ============================================================

import os
import time
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import cv2
import pydicom
import xml.etree.ElementTree as ET

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated.*",
    category=FutureWarning,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Default paths (edit to your own)
# -----------------------
# DEFAULT_ROOT_DIR = r"C:\Users\hkpuadmin\Desktop\dataset\CT"
# DEFAULT_MEDSAM_CKPT = r"C:\Users\hkpuadmin\weights\medsam_vit_b.pth"  # <-- MedSAM checkpoint path
# DEFAULT_SAVE_DIR = r"C:\Users\hkpuadmin\Desktop\CTM\fine-tune\SEGMENTATION"

DEFAULT_ROOT_DIR = r"/Users/leochan/Downloads/ctm/backend"
DEFAULT_MEDSAM_CKPT = r"/Users/leochan/Downloads/ctm/backend/medsam_seg_fold2_best.pt"  # <-- MedSAM checkpoint path
DEFAULT_SAVE_DIR = r"/Users/leochan/Downloads/ctm/backend/segmentation"

# SAM / MedSAM pixel normalization (from SAM code)
SAM_PIXEL_MEAN = [123.675, 116.28, 103.53]
SAM_PIXEL_STD  = [58.395, 57.12, 57.375]

# -----------------------
# Repro
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------
# CT windowing + DICOM helpers
# -----------------------
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

def get_hw_from_header(dcm_path: Path) -> Tuple[int, int]:
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    H = int(getattr(ds, "Rows", 0) or 0)
    W = int(getattr(ds, "Columns", 0) or 0)
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid Rows/Columns: {dcm_path}")
    return H, W

def is_ct_series(series_dir: Path) -> bool:
    dcm_files = list(series_dir.glob("*.dcm"))
    if not dcm_files:
        return False
    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True, force=True)
        return str(getattr(ds, "Modality", "")).upper() == "CT"
    except Exception:
        return False

def index_dicom_series(series_dir: Path):
    dcm_paths = sorted(series_dir.glob("*.dcm"))
    items = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue
        z = float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else None
        inst = int(ds.InstanceNumber) if hasattr(ds, "InstanceNumber") else None
        items.append((z, inst, p))

    if not items:
        return [], []

    if items[0][0] is not None:
        items.sort(key=lambda x: x[0])
        z_positions = [x[0] for x in items]
    else:
        items.sort(key=lambda x: x[1] if x[1] is not None else 0)
        z_positions = list(range(len(items)))

    return [x[2] for x in items], z_positions

def z_to_slice_index(z_positions, z_target):
    z_arr = np.array(z_positions, dtype=np.float32)
    return int(np.argmin(np.abs(z_arr - float(z_target))))

def find_series_dirs_with_xml(patient_root: Path):
    xmls = list(patient_root.rglob("*.xml"))
    out = []
    for x in xmls:
        series_dir = x.parent
        if len(list(series_dir.glob("*.dcm"))) > 0:
            out.append((series_dir, x))
    return out

# -----------------------
# XML ROI parsing (polygon mask)
# -----------------------
def parse_lidc_xml_rois(xml_path: Path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    if "}" in root.tag:
        ns_uri = root.tag.split("}")[0].strip("{")
        ns = {"ns": ns_uri}
        def q(t): return f"ns:{t}"
    else:
        ns = {}
        def q(t): return t

    rois_out = []
    nodule_nodes = root.findall(f".//{q('unblindedReadNodule')}", ns)

    for nnode in nodule_nodes:
        roi_nodes = nnode.findall(f".//{q('roi')}", ns)
        for roi in roi_nodes:
            z_node = roi.find(f".//{q('imageZposition')}", ns)
            if z_node is None or z_node.text is None:
                continue
            try:
                roi_z = float(z_node.text)
            except Exception:
                continue

            poly = []
            edge_nodes = roi.findall(f".//{q('edgeMap')}", ns)
            for edge in edge_nodes:
                x_node = edge.find(f".//{q('xCoord')}", ns)
                y_node = edge.find(f".//{q('yCoord')}", ns)
                if x_node is None or y_node is None:
                    continue
                try:
                    poly.append([int(float(x_node.text)), int(float(y_node.text))])
                except Exception:
                    continue

            if len(poly) > 2:
                rois_out.append({"roi_z": roi_z, "poly": np.array(poly, dtype=np.int32)})

    return rois_out

def poly_to_mask2d(poly: np.ndarray, H: int, W: int) -> np.ndarray:
    p = poly.copy()
    p[:, 0] = np.clip(p[:, 0], 0, W - 1)
    p[:, 1] = np.clip(p[:, 1], 0, H - 1)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [p.astype(np.int32)], 1)
    return mask

# -----------------------
# Build ROI-slice samples per patient
# -----------------------
def build_samples_for_patient(patient_root: Path) -> List[dict]:
    case_id = patient_root.name
    candidates = find_series_dirs_with_xml(patient_root)
    samples: List[dict] = []

    for series_dir, xml_path in candidates:
        if not is_ct_series(series_dir):
            continue

        sorted_dcms, z_positions = index_dicom_series(series_dir)
        if not sorted_dcms:
            continue

        try:
            H, W = get_hw_from_header(Path(sorted_dcms[0]))
        except Exception:
            continue

        rois = parse_lidc_xml_rois(xml_path)
        if not rois:
            continue

        for r in rois:
            k = z_to_slice_index(z_positions, r["roi_z"])
            if k < 0 or k >= len(sorted_dcms):
                continue

            dcm_path = Path(sorted_dcms[k])
            hu = safe_read_dicom_hu(dcm_path, verbose=False)
            if hu is None:
                continue

            samples.append({
                "case_id": case_id,
                "dcm_path": str(dcm_path),
                "H": int(H),
                "W": int(W),
                "poly": r["poly"],
            })

    return samples

# -----------------------
# Sliding coords (VAL windows)
# -----------------------
def make_sliding_coords(H: int, W: int, win: int, stride: int):
    ys = list(range(0, max(1, H - win + 1), stride))
    xs = list(range(0, max(1, W - win + 1), stride))
    if not ys: ys = [0]
    if not xs: xs = [0]
    if ys[-1] != max(0, H - win): ys.append(max(0, H - win))
    if xs[-1] != max(0, W - win): xs.append(max(0, W - win))
    return ys, xs

def expand_samples_to_windows(samples: List[dict], win: int, stride: int) -> List[dict]:
    items = []
    for s in samples:
        H, W = int(s["H"]), int(s["W"])
        ys, xs = make_sliding_coords(H, W, win, stride)
        for wy in ys:
            for wx in xs:
                it = dict(s)
                it["wy"] = int(wy)
                it["wx"] = int(wx)
                items.append(it)
    return items

# -----------------------
# Crop/Padding/Aug
# -----------------------
def _pad_if_needed(img: np.ndarray, mask: np.ndarray, top: int, left: int, win: int):
    H, W = img.shape
    bottom = top + win
    right = left + win
    pad_t = max(0, -top)
    pad_l = max(0, -left)
    pad_b = max(0, bottom - H)
    pad_r = max(0, right - W)
    if pad_t or pad_l or pad_b or pad_r:
        img = np.pad(img, ((pad_t, pad_b), (pad_l, pad_r)), mode="constant", constant_values=0.0)
        mask = np.pad(mask, ((pad_t, pad_b), (pad_l, pad_r)), mode="constant", constant_values=0)
        top += pad_t
        left += pad_l
    return img, mask, top, left

def apply_aug_pair(img: np.ndarray, mask: np.ndarray, p_flip=0.5, p_rot90=0.5):
    if random.random() < p_flip:
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        else:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

    if random.random() < p_rot90:
        k = random.randint(0, 3)
        img = np.rot90(img, k).copy()
        mask = np.rot90(mask, k).copy()

    return img, mask

# -----------------------
# ROI crop (training only) -> resize to out_size
# -----------------------
def roi_crop_and_resize(
    img01: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    out_size: int,
    jitter: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = img01.shape
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        cy, cx = H // 2, W // 2
    else:
        i = random.randrange(len(xs))
        cy, cx = int(ys[i]), int(xs[i])
        cy = int(np.clip(cy + random.randint(-jitter, jitter), 0, H - 1))
        cx = int(np.clip(cx + random.randint(-jitter, jitter), 0, W - 1))

    cs = int(crop_size)
    top = cy - cs // 2
    left = cx - cs // 2
    bottom = top + cs
    right = left + cs

    pad_t = max(0, -top)
    pad_l = max(0, -left)
    pad_b = max(0, bottom - H)
    pad_r = max(0, right - W)

    if pad_t or pad_l or pad_b or pad_r:
        img01 = np.pad(img01, ((pad_t, pad_b), (pad_l, pad_r)), mode="constant", constant_values=0.0)
        mask = np.pad(mask, ((pad_t, pad_b), (pad_l, pad_r)), mode="constant", constant_values=0)
        top += pad_t
        left += pad_l

    img_c = img01[top:top + cs, left:left + cs]
    m_c = mask[top:top + cs, left:left + cs]

    img_out = cv2.resize(img_c, (out_size, out_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    m_out = cv2.resize(m_c.astype(np.uint8), (out_size, out_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return img_out, m_out

# -----------------------
# ROI curriculum schedule (start -> end)
# -----------------------
def roi_prob_schedule(epoch: int, max_epochs: int, start: float, end: float) -> float:
    if max_epochs <= 1:
        return float(end)
    t = float(epoch - 1) / float(max_epochs - 1)
    return float(start + (end - start) * t)

# -----------------------
# Pos/neg window sampling on grid (RandCropByPosNegLabel-ish)
# -----------------------
def sample_window_coords_posneg(mask_full: np.ndarray, win: int, stride: int, pos_ratio: float) -> Tuple[int, int, int]:
    """
    Return (wy, wx, used_poswin)
    used_poswin=1 means chosen window has mask_sum>0
    """
    H, W = mask_full.shape
    ys, xs = make_sliding_coords(H, W, win, stride)

    pos_coords = []
    neg_coords = []
    for wy in ys:
        for wx in xs:
            m = mask_full[wy:wy + win, wx:wx + win]
            if int(m.sum()) > 0:
                pos_coords.append((wy, wx))
            else:
                neg_coords.append((wy, wx))

    want_pos = (random.random() < float(pos_ratio))

    if want_pos and len(pos_coords) > 0:
        wy, wx = random.choice(pos_coords)
        return int(wy), int(wx), 1
    if (not want_pos) and len(neg_coords) > 0:
        wy, wx = random.choice(neg_coords)
        return int(wy), int(wx), 0

    # fallback if one side empty
    if len(pos_coords) > 0:
        wy, wx = random.choice(pos_coords)
        return int(wy), int(wx), 1
    if len(neg_coords) > 0:
        wy, wx = random.choice(neg_coords)
        return int(wy), int(wx), 0

    return 0, 0, 0

def crop_and_resize_from_grid(
    img01: np.ndarray,
    mask_full: np.ndarray,
    crop: int,
    stride_train: int,
    out: int,
    want_pos: bool,
    max_tries: int = 50,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Use grid sliding windows on size=crop to pick POS/NEG crop (as requested),
    then resize to out (e.g. 448) for the model pipeline.
    """
    crop = int(crop)
    stride_train = int(stride_train)

    used_poswin = 0
    img_c = None
    m_c = None

    for _ in range(max_tries):
        force_ratio = 1.0 if want_pos else 0.0
        wy, wx, used_poswin = sample_window_coords_posneg(mask_full, win=crop, stride=stride_train, pos_ratio=force_ratio)

        img01_p, mask_p, wy, wx = _pad_if_needed(img01, mask_full, wy, wx, crop)
        img_c = img01_p[wy:wy + crop, wx:wx + crop]
        m_c = mask_p[wy:wy + crop, wx:wx + crop]
        if (want_pos and used_poswin == 1) or ((not want_pos) and used_poswin == 0):
            break

    if img_c is None or m_c is None:
        wy, wx, used_poswin = 0, 0, int(mask_full[:crop, :crop].sum() > 0)
        img01_p, mask_p, wy, wx = _pad_if_needed(img01, mask_full, wy, wx, crop)
        img_c = img01_p[wy:wy + crop, wx:wx + crop]
        m_c = mask_p[wy:wy + crop, wx:wx + crop]

    img_out = cv2.resize(img_c, (out, out), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    m_out = cv2.resize(m_c.astype(np.uint8), (out, out), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return img_out, m_out, int(m_out.sum() > 0)

# -----------------------
# Normalize to MedSAM/SAM tensor (resize to sam_size, 3ch, (x*255-mean)/std)
# -----------------------
def sam_normalize_to_tensor(img01: np.ndarray, mask01: np.ndarray, sam_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if sam_size is not None and int(sam_size) > 0:
        ss = int(sam_size)
        img01 = cv2.resize(img01, (ss, ss), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        mask01 = cv2.resize(mask01.astype(np.uint8), (ss, ss), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    x = (img01 * 255.0).astype(np.float32)
    x = np.stack([x, x, x], axis=0)  # 3 x H x W
    m = np.array(SAM_PIXEL_MEAN, dtype=np.float32)[:, None, None]
    s = np.array(SAM_PIXEL_STD, dtype=np.float32)[:, None, None]
    x = (x - m) / s
    x_t = torch.from_numpy(x).float()

    y = torch.from_numpy(mask01[None, ...].astype(np.float32))  # 1 x H x W
    return x_t, y

# -----------------------
# Dataset (TRAIN)
# -----------------------
class LIDCTrainSliceRepeatSegDataset(Dataset):
    def __init__(
        self,
        slice_samples: List[dict],
        repeat_factor: int,
        win: int,
        sam_size: int,
        train_crop: int,
        stride_train: int,
        max_tries: int,
        hu_window: Tuple[int, int],
        roi_prob_ref: Optional[torch.Tensor],
        roi_crop_size: int,
        roi_jitter: int,
        poswin_ratio: float,
        force_poswin_balance: bool,
        augment: bool,
    ):
        self.samples = slice_samples
        self.repeat_factor = int(max(1, repeat_factor))
        self.win = int(win)
        self.sam_size = int(sam_size)
        self.train_crop = int(train_crop)
        self.stride_train = int(stride_train)
        self.max_tries = int(max_tries)
        self.hu_window = hu_window
        self.roi_prob_ref = roi_prob_ref
        self.roi_crop_size = int(roi_crop_size)
        self.roi_jitter = int(roi_jitter)
        self.poswin_ratio = float(poswin_ratio)
        self.force_poswin_balance = bool(force_poswin_balance)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.samples) * self.repeat_factor

    def current_roi_prob(self) -> float:
        if self.roi_prob_ref is None:
            return 0.0
        return float(self.roi_prob_ref.item())

    def __getitem__(self, idx: int):
        s = self.samples[idx % len(self.samples)]

        hu = safe_read_dicom_hu(Path(s["dcm_path"]), verbose=False)
        if hu is None:
            raise RuntimeError(f"Bad dicom: {s['dcm_path']}")

        img01 = window_and_norm(hu, self.hu_window)
        mask_full = poly_to_mask2d(s["poly"], s["H"], s["W"]).astype(np.uint8)

        roi_prob = self.current_roi_prob()

        # ✅ LOCK intention FIRST
        if self.force_poswin_balance:
            thr = int(max(0, min(1000, round(self.poswin_ratio * 1000))))
            want_pos = (((idx * 9973) + 37) % 1000) < thr
        else:
            want_pos = (random.random() < float(self.poswin_ratio))

        # ROI crop allowed only when want_pos=True
        use_roi = (self.augment and want_pos and roi_prob > 0.0 and (random.random() < roi_prob))

        if use_roi:
            img_w, mask_w = roi_crop_and_resize(
                img01, mask_full,
                crop_size=self.roi_crop_size,
                out_size=self.win,
                jitter=self.roi_jitter,
            )
            used_poswin = 1
        else:
            img_w, mask_w, used_poswin = crop_and_resize_from_grid(
                img01=img01,
                mask_full=mask_full,
                crop=self.train_crop,
                stride_train=self.stride_train,
                out=self.win,
                want_pos=want_pos,
                max_tries=self.max_tries,
            )

        if self.augment:
            img_w, mask_w = apply_aug_pair(img_w, mask_w)

        x, y = sam_normalize_to_tensor(img_w, mask_w, sam_size=self.sam_size)

        return {
            "pixel_values": x,
            "mask": y,
            "case_id": s["case_id"],
            "slice_key": s["dcm_path"],
            "mask_sum": int(mask_w.sum()),
            "used_roi": int(use_roi),
            "used_poswin": int(used_poswin),
        }

# -----------------------
# Dataset (VAL): FULL sliding windows (win) then resize -> sam_size
# -----------------------
class LIDCValSlidingWindowSegDataset(Dataset):
    def __init__(
        self,
        window_items: List[dict],
        win: int,
        sam_size: int,
        hu_window: Tuple[int, int],
    ):
        self.items = window_items
        self.win = int(win)
        self.sam_size = int(sam_size)
        self.hu_window = hu_window

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        s = self.items[idx]
        hu = safe_read_dicom_hu(Path(s["dcm_path"]), verbose=False)
        if hu is None:
            raise RuntimeError(f"Bad dicom: {s['dcm_path']}")

        img01 = window_and_norm(hu, self.hu_window)
        mask_full = poly_to_mask2d(s["poly"], s["H"], s["W"]).astype(np.uint8)

        wy, wx = int(s["wy"]), int(s["wx"])
        img01, mask_full, wy, wx = _pad_if_needed(img01, mask_full, wy, wx, self.win)
        img_w = img01[wy:wy + self.win, wx:wx + self.win]
        mask_w = mask_full[wy:wy + self.win, wx:wx + self.win]

        x, y = sam_normalize_to_tensor(img_w, mask_w, sam_size=self.sam_size)

        return {
            "pixel_values": x,
            "mask": y,
            "case_id": s["case_id"],
            "slice_key": s["dcm_path"],
            "mask_sum": int(mask_w.sum()),
            "used_roi": 0,
            "used_poswin": int(mask_w.sum() > 0),
        }

def collate_seg(batch):
    pv = torch.stack([b["pixel_values"] for b in batch], dim=0)
    mk = torch.stack([b["mask"] for b in batch], dim=0)
    return {
        "pixel_values": pv,
        "mask": mk,
        "case_id": [b["case_id"] for b in batch],
        "slice_key": [b["slice_key"] for b in batch],
        "mask_sum": torch.tensor([b["mask_sum"] for b in batch], dtype=torch.long),
        "used_roi": torch.tensor([b.get("used_roi", 0) for b in batch], dtype=torch.long),
        "used_poswin": torch.tensor([b.get("used_poswin", 0) for b in batch], dtype=torch.long),
    }

# -----------------------
# MedSAM backbone + new segmentation head
# -----------------------
def _load_medsam_sam_model(sam_type: str, ckpt_path: str, device: torch.device):
    ckpt_path = str(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"MedSAM checkpoint not found: {ckpt_path}. Set --medsam_ckpt to the .pth file."
        )
    """
    Robust loader:
    - tries segment_anything.sam_model_registry
    - loads state dict (strict=False) if needed
    """
    try:
        from segment_anything import sam_model_registry
    except Exception as e:
        raise RuntimeError(
            "Cannot import segment_anything. Install:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    if sam_type not in sam_model_registry:
        raise ValueError(f"sam_type={sam_type} not in sam_model_registry keys={list(sam_model_registry.keys())}")

    model = None

    # Try builder with checkpoint kw first. Some environments fail here if ckpt was saved on CUDA.
    try:
        model = sam_model_registry[sam_type](checkpoint=ckpt_path)
    except Exception as e:
        print(f"[MedSAM load] fallback to CPU-safe manual load: {type(e).__name__}: {e}")
        model = sam_model_registry[sam_type]()
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if isinstance(state, dict):
            # Handle potential DataParallel prefixes.
            state = {
                (k.replace("module.", "", 1) if k.startswith("module.") else k): v
                for k, v in state.items()
            }
            # Handle fine-tuned checkpoints saved from wrapper model with keys like "sam.image_encoder..."
            if any(k.startswith("sam.") for k in state.keys()):
                state = {
                    (k.replace("sam.", "", 1) if k.startswith("sam.") else k): v
                    for k, v in state.items()
                }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0:
            print(f"[MedSAM load] missing keys (first 20): {missing[:20]}")
        if len(unexpected) > 0:
            print(f"[MedSAM load] unexpected keys (first 20): {unexpected[:20]}")

    model = model.to(device)
    return model

def freeze_all_params(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_last_n_sam_blocks(image_encoder: nn.Module, n: int = 2) -> int:
    """
    Unfreeze last N transformer blocks inside SAM image_encoder.blocks (ViT).
    Also unfreeze neck if exists.
    Returns number of blocks unfrozen (0 if not found).
    """
    k = 0
    blocks = getattr(image_encoder, "blocks", None)
    if blocks is None:
        return 0
    try:
        n = int(n)
        n = max(0, min(n, len(blocks)))
        if n == 0:
            return 0
        for blk in list(blocks)[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
        k = n
    except Exception:
        k = 0

    neck = getattr(image_encoder, "neck", None)
    if neck is not None:
        for p in neck.parameters():
            p.requires_grad = True
    return k

class SimpleSegHead(nn.Module):
    """
    Upsample SAM embedding (B,256,h,w) back to (B,1,H,W) using 4x stride-2 upsampling (~x16).
    """
    def __init__(self, in_ch: int = 256, mid: int = 128):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid),
            nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(mid, mid // 2, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(8, mid // 2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(mid // 2, mid // 4, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(4, mid // 4),
            nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(mid // 4, mid // 8, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(2, mid // 8),
            nn.GELU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(mid // 8, mid // 16, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(1, mid // 16),
            nn.GELU(),
        )
        self.out = nn.Conv2d(mid // 16, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        x = self.conv0(feat)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.out(x)
        if x.shape[-2:] != out_hw:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x

class MedSAMSegModel(nn.Module):
    def __init__(self, sam_type: str, medsam_ckpt: str, unfreeze_last_n: int, device: torch.device):
        super().__init__()
        self.sam = _load_medsam_sam_model(sam_type=sam_type, ckpt_path=medsam_ckpt, device=device)
        freeze_all_params(self.sam)  # freeze everything by default

        k = unfreeze_last_n_sam_blocks(self.sam.image_encoder, n=unfreeze_last_n)
        print(f"[MedSAM] frozen all, then ✅ unfroze last {k} image-encoder blocks (+ neck if exists).")

        self.head = SimpleSegHead(in_ch=256, mid=128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.sam.image_encoder(x)

        # Robust: if some implementations return (B, N, C)
        if feat.ndim == 3:
            n = feat.shape[1]
            s = int(math.sqrt(n))
            feat = feat.transpose(1, 2).contiguous().view(B, feat.shape[2], s, s)

        logits = self.head(feat, out_hw=(H, W))
        return logits

# -----------------------
# Loss + Metrics
# -----------------------
def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits).float()
    target = target.float()
    inter = (prob * target).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

def _confusion_from_logits(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    prob = torch.sigmoid(logits)
    pred = (prob > thr)
    t = (target > 0.5)
    tp = (pred & t).sum().item()
    fp = (pred & (~t)).sum().item()
    fn = ((~pred) & t).sum().item()
    tn = ((~pred) & (~t)).sum().item()
    return int(tp), int(fp), int(fn), int(tn)

def _metrics_from_confusion(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-6):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }

# -----------------------
# AUTOMATIC THRESHOLD (SEGMENTATION / DICE) via histograms
# -----------------------
def seg_metrics_from_confusion(tp: float, fp: float, fn: float, tn: float, eps: float = 1e-6) -> dict:
    tp = float(tp); fp = float(fp); fn = float(fn); tn = float(tn)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
        "f1": float((2 * precision * recall) / (precision + recall + eps)),
    }

def _cum_from_high(hist: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(hist, dims=[0]), dim=0), dims=[0])

def find_best_threshold_from_hist(pos_hist: torch.Tensor, neg_hist: torch.Tensor, key: str = "dice") -> Tuple[float, dict]:
    key = str(key).lower().strip()
    key_map = {
        "dice": "dice",
        "iou": "iou",
        "jaccard": "iou",
        "precision": "precision",
        "prec": "precision",
        "recall": "recall",
        "sens": "recall",
        "sensitivity": "recall",
        "spec": "specificity",
        "specificity": "specificity",
    }
    if key not in key_map:
        raise ValueError(f"Unknown key={key}. Supported keys={sorted(key_map.keys())}")
    use_key = key_map[key]

    pos_hist_f = pos_hist.to(dtype=torch.float64)
    neg_hist_f = neg_hist.to(dtype=torch.float64)

    pos_total = pos_hist_f.sum()
    neg_total = neg_hist_f.sum()

    tp = _cum_from_high(pos_hist_f)
    fp = _cum_from_high(neg_hist_f)
    fn = pos_total - tp
    tn = neg_total - fp

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-12)
    iou = tp / (tp + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)

    metric_vec = {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }[use_key]

    best_idx = int(torch.argmax(metric_vec).item())
    n_bins = int(pos_hist.numel())
    best_thr = float(best_idx / max(1, (n_bins - 1)))

    tp_i = float(tp[best_idx].item())
    fp_i = float(fp[best_idx].item())
    fn_i = float(fn[best_idx].item())
    tn_i = float(tn[best_idx].item())
    best_mets = seg_metrics_from_confusion(tp_i, fp_i, fn_i, tn_i)
    best_mets.update({"thr": best_thr, "tp": tp_i, "fp": fp_i, "fn": fn_i, "tn": tn_i})
    return best_thr, best_mets

def eval_segmentation_thresholds_from_hist(pos_hist: torch.Tensor, neg_hist: torch.Tensor, fixed_thr: float = 0.5) -> dict:
    n_bins = int(pos_hist.numel())
    fixed_thr = float(fixed_thr)
    fixed_idx = int(round(fixed_thr * (n_bins - 1)))
    fixed_idx = max(0, min(n_bins - 1, fixed_idx))

    pos_hist_f = pos_hist.to(dtype=torch.float64)
    neg_hist_f = neg_hist.to(dtype=torch.float64)
    pos_total = pos_hist_f.sum()
    neg_total = neg_hist_f.sum()

    tp = _cum_from_high(pos_hist_f)
    fp = _cum_from_high(neg_hist_f)
    fn = pos_total - tp
    tn = neg_total - fp

    fixed_mets = seg_metrics_from_confusion(
        float(tp[fixed_idx].item()),
        float(fp[fixed_idx].item()),
        float(fn[fixed_idx].item()),
        float(tn[fixed_idx].item()),
    )
    fixed_mets.update({
        "thr": float(fixed_idx / max(1, (n_bins - 1))),
        "tp": float(tp[fixed_idx].item()),
        "fp": float(fp[fixed_idx].item()),
        "fn": float(fn[fixed_idx].item()),
        "tn": float(tn[fixed_idx].item()),
    })

    thr_dice, best_dice = find_best_threshold_from_hist(pos_hist, neg_hist, key="dice")
    thr_iou, best_iou = find_best_threshold_from_hist(pos_hist, neg_hist, key="iou")

    return {
        "n_bins": n_bins,
        "fixed": fixed_mets,
        "best_dice": best_dice,
        "best_iou": best_iou,
        "thr_dice": float(thr_dice),
        "thr_iou": float(thr_iou),
    }

def pretty_print_seg_thr(pack: dict, prefix: str = "auto_thr(seg)") -> str:
    if pack is None:
        return f"[{prefix}] (none)"

    f = pack["fixed"]
    bd = pack["best_dice"]
    bi = pack["best_iou"]

    def _fmt(x):
        try:
            if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                return "nan"
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    return (
        f"[{prefix}] bins={pack['n_bins']} | "
        f"@thr={_fmt(f['thr'])} dice={_fmt(f['dice'])} iou={_fmt(f['iou'])} prec={_fmt(f['precision'])} rec={_fmt(f['recall'])} spec={_fmt(f['specificity'])} | "
        f"best_dice thr={_fmt(pack['thr_dice'])} dice={_fmt(bd['dice'])} iou={_fmt(bd['iou'])} | "
        f"best_iou thr={_fmt(pack['thr_iou'])} iou={_fmt(bi['iou'])} dice={_fmt(bi['dice'])}"
    )

# -----------------------
# Anti-crash checkpointing (save last + resume)
# -----------------------
def _get_rng_state():
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st

def _set_rng_state(st: dict):
    try:
        random.setstate(st["python"])
        np.random.set_state(st["numpy"])
        torch.set_rng_state(st["torch"])
        if torch.cuda.is_available() and "cuda" in st:
            torch.cuda.set_rng_state_all(st["cuda"])
    except Exception as e:
        print(f"[resume] warning: failed to restore RNG state: {type(e).__name__}: {e}")

def save_ckpt(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_dice: float,
    best_epoch: int,
    bad_epochs: int,
    roi_prob_val: float,
    args: argparse.Namespace,
    fold_idx: int,
    step_in_epoch: int = 0,
):
    ckpt = {
        "epoch": int(epoch),
        "fold": int(fold_idx),
        "step_in_epoch": int(step_in_epoch),
        "best_dice": float(best_dice),
        "best_epoch": int(best_epoch),
        "bad_epochs": int(bad_epochs),
        "roi_prob": float(roi_prob_val),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": _get_rng_state(),
        "args": vars(args),
        "time": float(time.time()),
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # atomic replace on Windows

def load_ckpt(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"[resume] warning: failed to load scheduler state: {type(e).__name__}: {e}")

    # move optimizer tensors to GPU if needed
    if device.type == "cuda":
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    if "rng" in ckpt:
        _set_rng_state(ckpt["rng"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1  # resume from next epoch
    meta = {
        "fold": int(ckpt.get("fold", -1)),
        "step_in_epoch": int(ckpt.get("step_in_epoch", 0)),
        "best_dice": float(ckpt.get("best_dice", -1.0)),
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        "bad_epochs": int(ckpt.get("bad_epochs", 0)),
        "roi_prob": float(ckpt.get("roi_prob", 0.0)),
        "start_epoch": int(start_epoch),
    }
    return meta

# -----------------------
# Train / Val loops
# -----------------------
def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    pos_weight: float = 1.0,
    bce_w: float = 0.5,
    dice_w: float = 0.5,
    save_every_steps: int = 0,
    save_step_fn=None,
):
    """
    Train loss = bce_w * BCEWithLogitsLoss + dice_w * DiceLoss
    pos_weight: >1.0 increases positive-pixel penalty to handle imbalance (BCE part).
    """
    model.train()
    loss_meter = 0.0
    bce_meter = 0.0
    dice_meter = 0.0
    n_seen = 0

    roi_sum = 0
    poswin_sum = 0
    n_samples = 0

    pw = torch.tensor([float(pos_weight)], device=device, dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    t0 = time.time()
    step_idx = 0

    for batch in loader:
        step_idx += 1

        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        bce_loss = bce(logits, y)
        d_loss = dice_loss_with_logits(logits, y)
        loss = float(bce_w) * bce_loss + float(dice_w) * d_loss

        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        loss_meter += float(loss.item()) * bs
        bce_meter += float(bce_loss.item()) * bs
        dice_meter += float(d_loss.item()) * bs
        n_seen += bs

        roi_sum += int(batch["used_roi"].sum().item())
        poswin_sum += int(batch["used_poswin"].sum().item())
        n_samples += bs

        # ✅ optional step checkpoint
        if save_every_steps and save_every_steps > 0 and (step_idx % int(save_every_steps) == 0):
            if callable(save_step_fn):
                try:
                    save_step_fn(step_idx)
                except Exception as e:
                    print(f"[ckpt] warning: step-save failed: {type(e).__name__}: {e}")

    dt = time.time() - t0
    avg_loss = loss_meter / max(1, n_seen)
    avg_bce = bce_meter / max(1, n_seen)
    avg_dice = dice_meter / max(1, n_seen)
    roi_pct = 100.0 * roi_sum / max(1, n_samples)
    poswin_pct = 100.0 * poswin_sum / max(1, n_samples)
    return avg_loss, avg_bce, avg_dice, roi_pct, poswin_pct, dt

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    pos_weight: float = 1.0,
    bce_w: float = 0.5,
    dice_w: float = 0.5,
    thr: float = 0.5,
    auto_thr: bool = True,
    n_grid: int = 101,
):
    model.eval()
    loss_meter = 0.0
    n_seen = 0

    tp_tot = fp_tot = fn_tot = tn_tot = 0

    n_bins = int(max(21, n_grid))
    pos_hist = torch.zeros(n_bins, device=device, dtype=torch.int64) if auto_thr else None
    neg_hist = torch.zeros(n_bins, device=device, dtype=torch.int64) if auto_thr else None

    pw = torch.tensor([float(pos_weight)], device=device, dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        logits = model(x)

        bce_loss = bce(logits, y)
        d_loss = dice_loss_with_logits(logits, y)
        loss = float(bce_w) * bce_loss + float(dice_w) * d_loss

        bs = x.shape[0]
        loss_meter += float(loss.item()) * bs
        n_seen += bs

        tp, fp, fn, tn = _confusion_from_logits(logits, y, thr=float(thr))
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn
        tn_tot += tn

        if auto_thr:
            prob = torch.sigmoid(logits.detach())
            gt = (y > 0.5)

            idx = torch.clamp((prob * (n_bins - 1)).to(dtype=torch.int64), 0, n_bins - 1)
            idx_flat = idx.view(-1)
            gt_flat = gt.view(-1)

            pos_idx = idx_flat[gt_flat]
            neg_idx = idx_flat[~gt_flat]

            pos_hist += torch.bincount(pos_idx, minlength=n_bins)
            neg_hist += torch.bincount(neg_idx, minlength=n_bins)

    avg_loss = loss_meter / max(1, n_seen)
    mets_fixed = _metrics_from_confusion(tp_tot, fp_tot, fn_tot, tn_tot)

    thr_pack = None
    if auto_thr:
        thr_pack = eval_segmentation_thresholds_from_hist(pos_hist, neg_hist, fixed_thr=float(thr))
        thr_pack["log_str"] = pretty_print_seg_thr(thr_pack, prefix="auto_thr(seg)")
        thr_pack["best_thr_dice"] = float(thr_pack["thr_dice"])
        thr_pack["best_thr_iou"] = float(thr_pack["thr_iou"])

    return avg_loss, mets_fixed, thr_pack

# -----------------------
# Simple K-Fold split (no label stratification)
# -----------------------
def make_kfold(case_ids: List[str], k: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    ids = case_ids.copy()
    rng.shuffle(ids)
    folds = [[] for _ in range(k)]
    for i, cid in enumerate(ids):
        folds[i % k].append(cid)
    return folds

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--root_dir", type=str, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # CT / sampling
    ap.add_argument("--hu_lo", type=int, default=-1000)
    ap.add_argument("--hu_hi", type=int, default=400)
    ap.add_argument("--win", type=int, default=448, help="crop/window size BEFORE feeding to SAM (will resize to sam_size)")
    ap.add_argument("--sam_size", type=int, default=1024, help="model input size for MedSAM/SAM (default 1024)")
    ap.add_argument("--stride", type=int, default=224, help="VAL sliding stride on original image (window=win)")
    ap.add_argument("--train_crop", type=int, default=256, help="TRAIN grid crop size BEFORE resize to win")
    ap.add_argument("--stride_train", type=int, default=128)
    ap.add_argument("--repeat_factor", type=int, default=2)
    ap.add_argument("--max_tries", type=int, default=50)

    # ROI curriculum
    ap.add_argument("--roi_prob_start", type=float, default=0.8)
    ap.add_argument("--roi_prob_end", type=float, default=0.2)
    ap.add_argument("--roi_crop", type=int, default=256)
    ap.add_argument("--roi_jitter", type=int, default=16)
    ap.add_argument("--poswin_ratio", type=float, default=0.50)

    # model
    ap.add_argument("--medsam_ckpt", type=str, default=DEFAULT_MEDSAM_CKPT)
    ap.add_argument("--sam_type", type=str, default="vit_b", help="SAM type: vit_b / vit_l / vit_h (depends on your ckpt)")
    ap.add_argument("--unfreeze_last_n", type=int, default=2, help="unfreeze last N transformer blocks in image encoder")

    # train
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--lr_bb", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--scheduler_patience", type=int, default=3)
    ap.add_argument("--early_stop", type=int, default=10)

    ap.add_argument("--pos_weight", type=float, default=1.0, help="BCE pos_weight for positive pixels (imbalance). 1.0 means no reweight.")
    ap.add_argument("--bce_w", type=float, default=0.5, help="weight for BCEWithLogitsLoss")
    ap.add_argument("--dice_w", type=float, default=0.5, help="weight for Dice loss")
    ap.add_argument("--val_thr", type=float, default=0.5, help="threshold for val metrics (dice/iou/prec/rec etc.)")
    ap.add_argument("--auto_thr", type=int, default=1, help="enable automatic threshold sweep (1=yes, 0=no)")
    ap.add_argument("--thr_grid", type=int, default=501, help="threshold grid points for sweep in [0,1]")
    ap.add_argument("--force_poswin_balance", action=argparse.BooleanOptionalAction, default=True, help="make pos/neg window intent deterministic by idx so poswin percent ~= poswin_ratio")
    ap.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True, help="keep DataLoader workers alive across epochs")

    # logging / ckpt
    ap.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)

    # ✅ anti-crash ckpt + resume
    ap.add_argument("--save_last", type=int, default=1, help="save last checkpoint every epoch (1=yes)")
    ap.add_argument("--save_every_steps", type=int, default=0, help="also save every N train steps (0=disable)")
    ap.add_argument("--resume", type=str, default="", help="path to a .pt checkpoint to resume (e.g., ..._last.pt)")

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise SystemExit(f"[ERROR] root_dir not found: {root_dir}")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[save] checkpoints will be saved to: {args.save_dir}")

    # scan patients
    patient_dirs_all = sorted([p for p in root_dir.glob("LIDC-IDRI-*") if p.is_dir()])
    if len(patient_dirs_all) == 0:
        patient_dirs_all = sorted([p for p in root_dir.rglob("LIDC-IDRI-*") if p.is_dir() and p.name.startswith("LIDC-IDRI-")])
    if len(patient_dirs_all) == 0:
        raise SystemExit(f"[ERROR] no LIDC-IDRI-* dirs under: {root_dir}")

    print(f"[data] total patient dirs found: {len(patient_dirs_all)}")
    print("[scan] building ROI-slice samples for each patient .. (this may take a while)")

    patient_to_samples: Dict[str, List[dict]] = {}
    dropped = []
    for p in patient_dirs_all:
        s = build_samples_for_patient(p)
        if len(s) == 0:
            dropped.append(p.name)
        else:
            patient_to_samples[p.name] = s

    case_ids_all = sorted(patient_to_samples.keys())
    print(f"[scan] usable patients with ROI-slices={len(case_ids_all)} | dropped(no ROI)={len(dropped)}")
    if len(case_ids_all) < args.folds:
        raise RuntimeError("Too few patients with ROI-slices after filtering.")

    folds = make_kfold(case_ids_all, k=args.folds, seed=args.seed)
    hu_window = (int(args.hu_lo), int(args.hu_hi))

    roi_prob_ref = torch.tensor([float(args.roi_prob_start)], dtype=torch.float32)

    fold_results = []
    for fi in range(args.folds):
        val_ids = folds[fi]
        train_ids = [cid for fj in range(args.folds) if fj != fi for cid in folds[fj]]

        train_samples = []
        val_samples = []
        for cid in train_ids:
            train_samples += patient_to_samples[cid]
        for cid in val_ids:
            val_samples += patient_to_samples[cid]

        print(f"\n==================== Fold {fi+1}/{args.folds} ====================")
        print(f"[fold split] train patients={len(train_ids)} | val patients={len(val_ids)}")
        print(f"[fold samples] train_slices={len(train_samples)} | val_slices={len(val_samples)}")

        # ---- build model + optimizer per fold ----
        model = MedSAMSegModel(
            sam_type=args.sam_type,
            medsam_ckpt=args.medsam_ckpt,
            unfreeze_last_n=args.unfreeze_last_n,
            device=device,
        ).to(device)

        head_params = [p for p in model.head.parameters() if p.requires_grad]
        bb_params = [p for p in model.sam.image_encoder.parameters() if p.requires_grad]
        print(f"[params] head trainable={sum(p.numel() for p in head_params):,} | bb trainable={sum(p.numel() for p in bb_params):,}")

        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": float(args.lr_head)},
                {"params": bb_params, "lr": float(args.lr_bb)},
            ],
            weight_decay=float(args.wd),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=int(args.scheduler_patience)
        )

        train_ds = LIDCTrainSliceRepeatSegDataset(
            slice_samples=train_samples,
            repeat_factor=args.repeat_factor,
            win=args.win,
            sam_size=args.sam_size,
            train_crop=args.train_crop,
            stride_train=args.stride_train,
            max_tries=args.max_tries,
            hu_window=hu_window,
            roi_prob_ref=roi_prob_ref,
            roi_crop_size=args.roi_crop,
            roi_jitter=args.roi_jitter,
            poswin_ratio=args.poswin_ratio,
            force_poswin_balance=args.force_poswin_balance,
            augment=True,
        )

        val_items = expand_samples_to_windows(val_samples, win=args.win, stride=args.stride)
        val_ds = LIDCValSlidingWindowSegDataset(
            window_items=val_items,
            win=args.win,
            sam_size=args.sam_size,
            hu_window=hu_window,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_seg,
            persistent_workers=(args.persistent_workers and args.num_workers > 0),
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_seg,
            persistent_workers=(args.persistent_workers and args.num_workers > 0),
        )

        # ---- ckpt state ----
        best_dice = -1.0
        best_epoch = -1
        bad_epochs = 0
        start_epoch = 1

        last_ckpt_path = os.path.join(args.save_dir, f"medsam_seg_fold{fi+1}_last.pt")

        # ✅ resume (optional)
        if args.resume:
            if os.path.exists(args.resume):
                meta = load_ckpt(args.resume, model, optimizer, scheduler, device=device)
                if meta["fold"] not in (-1, fi + 1):
                    print(f"[resume] warning: checkpoint fold={meta['fold']} but current fold={fi+1}. Continuing anyway.")
                start_epoch = int(meta["start_epoch"])
                best_dice = float(meta["best_dice"])
                best_epoch = int(meta["best_epoch"])
                bad_epochs = int(meta["bad_epochs"])
                roi_prob_ref[0] = float(meta.get("roi_prob", args.roi_prob_start))
                print(f"[resume] loaded: {args.resume}")
                print(f"[resume] start_epoch={start_epoch} best_dice={best_dice:.4f} best_epoch={best_epoch} bad_epochs={bad_epochs} roi_prob={float(roi_prob_ref.item()):.3f}")
            else:
                print(f"[resume] checkpoint not found: {args.resume} (ignored)")

        for epoch in range(int(start_epoch), int(args.epochs) + 1):
            cur_roi = roi_prob_schedule(epoch, args.epochs, args.roi_prob_start, args.roi_prob_end)
            roi_prob_ref[0] = float(cur_roi)

            def _save_step(step_idx: int):
                save_ckpt(
                    path=last_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_dice=best_dice,
                    best_epoch=best_epoch,
                    bad_epochs=bad_epochs,
                    roi_prob_val=float(roi_prob_ref.item()),
                    args=args,
                    fold_idx=fi + 1,
                    step_in_epoch=step_idx,
                )

            tr_loss, tr_bce, tr_dice, roi_pct, poswin_pct, tr_dt = train_one_epoch(
                model, train_loader, optimizer, device,
                pos_weight=float(args.pos_weight), bce_w=float(args.bce_w), dice_w=float(args.dice_w),
                save_every_steps=int(args.save_every_steps),
                save_step_fn=_save_step,
            )

            va_loss, va_mets, va_bin = validate(
                model, val_loader, device,
                pos_weight=float(args.pos_weight), bce_w=float(args.bce_w), dice_w=float(args.dice_w),
                thr=float(args.val_thr),
                auto_thr=bool(args.auto_thr),
                n_grid=int(args.thr_grid),
            )

            # scheduler monitors val dice (higher is better)
            scheduler.step(va_mets["dice"])

            lr0 = optimizer.param_groups[0]["lr"]
            lr1 = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr0

            print(
                f"[E{epoch:03d}] roi_prob={cur_roi:.3f} | "
                f"train_loss={tr_loss:.4f} (bce={tr_bce:.4f} dice={tr_dice:.4f}) ({tr_dt:.1f}s) | "
                f"roi%={roi_pct:.1f} poswin%={poswin_pct:.1f} | "
                f"val_loss={va_loss:.4f} dice={va_mets['dice']:.4f} iou={va_mets['iou']:.4f} "
                f"prec={va_mets['precision']:.4f} rec={va_mets['recall']:.4f} f1={va_mets['f1']:.4f} "
                f"spec={va_mets['specificity']:.4f} | "
                f"lr_head={lr0:.2e} lr_bb={lr1:.2e}"
            )

            if va_bin is not None:
                print("    " + va_bin["log_str"])

            # ✅ always save "last" at end of epoch (anti-crash)
            if int(args.save_last) == 1:
                try:
                    save_ckpt(
                        path=last_ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        best_dice=best_dice,
                        best_epoch=best_epoch,
                        bad_epochs=bad_epochs,
                        roi_prob_val=float(roi_prob_ref.item()),
                        args=args,
                        fold_idx=fi + 1,
                        step_in_epoch=0,
                    )
                except Exception as e:
                    print(f"[ckpt] warning: epoch-save failed: {type(e).__name__}: {e}")

            improved = va_mets["dice"] > best_dice + 1e-6
            if improved:
                best_dice = va_mets["dice"]
                best_epoch = epoch
                bad_epochs = 0

                ckpt_path = os.path.join(args.save_dir, f"medsam_seg_fold{fi+1}_best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "best_dice": float(best_dice),
                        "model": model.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"  ✅ saved best: {ckpt_path} (dice={best_dice:.4f})")
            else:
                bad_epochs += 1
                if bad_epochs >= int(args.early_stop):
                    print(f"  ⏹️ early stop: no improvement for {bad_epochs} epochs. best@E{best_epoch} dice={best_dice:.4f}")
                    break

        fold_results.append({"fold": fi + 1, "best_epoch": best_epoch, "best_dice": best_dice})

    print("\n==================== DONE ====================")
    for r in fold_results:
        print(f"Fold {r['fold']}: best_epoch={r['best_epoch']} best_dice={r['best_dice']:.4f}")

if __name__ == "__main__":
    main()