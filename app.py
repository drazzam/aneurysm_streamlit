# ---------------------------------------------------------------------
# What this app does:
# 1) Accepts a DICOM series (.zip) or NIfTI (.nii/.nii.gz) CTA.
# 2) Runs GLIA‑Net (pretrained) to produce a 3D aneurysm mask.
# 3) Lets you scroll the study and see mask overlays.
# 4) Affine‑registers an arterial territories atlas (MNI) and annotates each lesion with a territory label.
#
# Folder expectations (relative to this file):
#   PRETRAINED/checkpoint-0245700.pt
#   ATLAS/ArterialAtlasLables.txt
#   ATLAS/ArterialAtlas_level2.nii
#   ATLAS/ArterialAtlas.nii
#   ATLAS/ProbArterialAtlas_average.nii
#   glianet/ ... (or GLIA-Net/ or GLIA-Net-master/) — the GLIA-Net repo code
#
# Requirements (in requirements.txt):
# streamlit, torch, numpy, nibabel, SimpleITK, scikit-image, pyyaml, colorlog,
# tensorboardX, opencv-python-headless, pandas, pydicom
#
# ---------------------------------------------------------------------

import os
import io
import sys
import cv2
import yaml
import uuid
import time
import shutil
import zipfile
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import SimpleITK as sitk
from pathlib import Path
from skimage.measure import label, regionprops

# ----------------------------- App Config -----------------------------
st.set_page_config(page_title="CTA Aneurysm — GLIA‑Net + Atlas Labels", layout="wide")

APP_ROOT = Path.cwd()
ATLAS_DIR = APP_ROOT / "ATLAS"
PRETRAINED_DIR = APP_ROOT / "PRETRAINED"
CHECKPOINT_FILE = PRETRAINED_DIR / "checkpoint-0245700.pt"
OUTPUTS_BASE = APP_ROOT / "outputs"
OUTPUTS_BASE.mkdir(exist_ok=True, parents=True)

# ------------------------- Utility: UI Helpers ------------------------
def ui_badge(text, color="#0f766e"):
    st.markdown(
        f"""<span style="background:{color};color:white;padding:2px 8px;border-radius:6px;font-size:0.8rem">{text}</span>""",
        unsafe_allow_html=True,
    )

def ensure_exists(path: Path, what: str):
    if not path.exists():
        st.error(f"❌ Missing **{what}** at: `{path}`")
        st.stop()

# ------------------------- Locate GLIA‑Net Code -----------------------
def find_glia_root() -> Path | None:
    candidates = [
        APP_ROOT / "glianet",
        APP_ROOT / "GLIA-Net",
        APP_ROOT / "GLIA-Net-master",
    ]
    # also scan subdirs with GLIA in name
    for p in APP_ROOT.iterdir():
        if p.is_dir() and "GLIA" in p.name.upper():
            candidates.append(p)
    for c in candidates:
        if (c / "inference.py").exists() and (c / "utils" / "project_utils.py").exists():
            return c.resolve()
    return None

GLIA_ROOT = find_glia_root()
if GLIA_ROOT is None:
    st.error("❌ Could not locate the GLIA‑Net repo folder. Place it as `glianet/`, `GLIA-Net/`, or `GLIA-Net-master/` next to this app.")
    st.stop()

sys.path.insert(0, str(GLIA_ROOT))

# Now import GLIA‑Net internals
try:
    from utils.project_utils import load_config, get_logger, get_devices
    from core import Inferencer
    from data_loader import AneurysmSegTestManager
except Exception as e:
    st.exception(e)
    st.error("❌ Failed to import GLIA‑Net internals from the repo. Make sure the repo is complete.")
    st.stop()

# --------------------------- GLIA‑Net Runner --------------------------
def _load_glia_config(cfg_path: Path):
    # Use GLIA’s loader if present; otherwise use yaml.safe_load
    if cfg_path.exists():
        try:
            cfg = load_config(str(cfg_path))
        except Exception:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
    else:
        st.error(f"❌ Missing GLIA config YAML at {cfg_path}")
        st.stop()
    # Normalize to dict-like
    try:
        _ = cfg.get("ckpt_folder", None)  # DotMap & dict both support get
    except AttributeError:
        # wrap it so .get exists
        cfg = dict(cfg)
    return cfg

def _ensure_ckpt_in_config(config: dict):
    # Ensure checkpoint entries exist and point to PRETRAINED/checkpoint-0245700.pt
    ckpt_folder = str(PRETRAINED_DIR)
    ckpt_file = CHECKPOINT_FILE.name
    config["ckpt_folder"] = ckpt_folder
    config["ckpt_file"] = ckpt_file
    return config

def _resolve_output_path(output_dir: Path) -> Path | None:
    # Grab the newest .nii or .nii.gz from output_dir
    cands = list(output_dir.glob("*.nii")) + list(output_dir.glob("*.nii.gz"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def run_glianet_inference(input_path: Path,
                          input_type: str,
                          device_str: str,
                          output_dir: Path) -> Path:
    """
    Run GLIA‑Net inference on a folder (DICOM series) or single NIfTI.
    Returns path to the predicted mask NIfTI.
    """
    ensure_exists(CHECKPOINT_FILE, "pretrained checkpoint")
    cfg_path = GLIA_ROOT / "configs" / "inference_GLIA-Net.yaml"
    config = _load_glia_config(cfg_path)
    config = _ensure_ckpt_in_config(config)

    logger = get_logger("GLIA-Infer", logging_folder=None, verbose=False)
    devices = get_devices(device_str, logger)

    # Data manager (handles resampling/tiling per config)
    test_mgr = AneurysmSegTestManager(config, logger, devices)

    output_dir.mkdir(parents=True, exist_ok=True)

    infer = Inferencer(
        config=config,
        exp_path=str(APP_ROOT.resolve()),            # so ckpt_folder is relative to the app root
        devices=devices,
        inference_file_or_folder=str(input_path.resolve()),
        output_folder=str(output_dir.resolve()),
        input_type=input_type,                       # "dcm" or "nii"
        save_binary=True,                            # write binary mask
        save_prob=False,
        save_global=False,
        test_loader_manager=test_mgr,
        logger=logger
    )

    infer.inference()

    pred_path = _resolve_output_path(output_dir)
    if pred_path is None or not pred_path.exists():
        raise RuntimeError("GLIA‑Net did not produce a prediction file.")
    return pred_path

# ------------------------------ IO Utils ------------------------------
def extract_zip_to_temp(upload: bytes) -> Path:
    tmp_root = Path(tempfile.mkdtemp(prefix="cta_case_"))
    with zipfile.ZipFile(io.BytesIO(upload)) as zf:
        zf.extractall(tmp_root)
    # Return the folder that (likely) contains the DICOM series
    return tmp_root

def read_patient_image(input_kind: str, path: Path) -> sitk.Image:
    """
    input_kind: "dcm_dir" (folder with one DICOM series) or "nii" (single file)
    Returns a SimpleITK 3D image in patient space.
    """
    if input_kind == "dcm_dir":
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(path))
        if not series_ids:
            raise RuntimeError("No DICOM series found in the uploaded folder.")
        if len(series_ids) > 1:
            st.info(f"ℹ️ Found {len(series_ids)} series; using the first one.")
        file_names = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
        reader.SetFileNames(file_names)
        img = reader.Execute()
        # Ensure direction is set; SITK handles it.
        return img
    elif input_kind == "nii":
        return sitk.ReadImage(str(path))
    else:
        raise ValueError("input_kind must be 'dcm_dir' or 'nii'.")

def sitk_to_numpy_and_spacing(img: sitk.Image) -> tuple[np.ndarray, tuple[float,float,float]]:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [Z,Y,X]
    sx, sy, sz = img.GetSpacing()  # NOTE: SITK spacing order is (X,Y,Z)
    spacing_zyx = (sz, sy, sx)
    return arr, spacing_zyx

def window_ct(img_u16_or_f32: np.ndarray, wl=40.0, ww=400.0) -> np.ndarray:
    lo, hi = wl - ww/2.0, wl + ww/2.0
    v = np.clip(img_u16_or_f32, lo, hi)
    v = (v - lo) / max(hi - lo, 1e-6)
    return (v * 255.0).astype(np.uint8)

# ----------------------------- Overlay Utils -------------------------
def components_3d(mask_np: np.ndarray, spacing_zyx: tuple[float,float,float]):
    lab = label(mask_np > 0, connectivity=1)  # 6-connectivity
    props = regionprops(lab)
    lesions = []
    dz, dy, dx = spacing_zyx
    for p in props:
        zc, yc, xc = p.centroid
        vol_mm3 = float(p.area) * dz * dy * dx
        equiv_d = ( (6.0 * vol_mm3 / np.pi) ** (1.0 / 3.0) )
        lesions.append({
            "id": int(p.label),
            "centroid_zyx": (float(zc), float(yc), float(xc)),
            "bbox_zyx": tuple(int(v) for v in p.bbox),  # (z0,y0,x0,z1,y1,x1)
            "volume_mm3": float(vol_mm3),
            "equiv_diam_mm": float(equiv_d)
        })
    return lab, lesions

def draw_overlay(gray_u8: np.ndarray, mask_slice: np.ndarray, lesions_here=None, show_boxes=True) -> np.ndarray:
    rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    m = (mask_slice > 0).astype(np.uint8) * 255
    if m.any():
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cnts, -1, (0, 0, 255), 1)  # red contours
    if show_boxes and lesions_here:
        for L in lesions_here:
            y0, x0 = int(L["bbox_zyx"][1]), int(L["bbox_zyx"][2])
            y1, x1 = int(L["bbox_zyx"][4]-1), int(L["bbox_zyx"][5]-1)
            cv2.rectangle(rgb, (x0,y0), (x1,y1), (0,255,0), 1)
            label_txt = f"ID {L['id']}"
            if "territory_name" in L and L["territory_name"]:
                label_txt += f" • {L['territory_name']}"
            cv2.putText(rgb, label_txt, (x0, max(12, y0-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)
    return rgb

# ---------------------------- Atlas Functions -------------------------
def parse_labels_txt(path: Path) -> dict[int, str]:
    lut = {}
    if not path.exists():
        return {1: "ACA", 2: "MCA", 3: "PCA", 4: "Vertebro-Basilar"}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # Try "ID name..." tolerant parsing
            parts = [p for p in s.replace(",", " ").split() if p]
            try:
                k = int(parts[0])
                name = " ".join(parts[1:]) if len(parts) > 1 else str(k)
                lut[k] = name
            except Exception:
                continue
    if not lut:
        lut = {1: "ACA", 2: "MCA", 3: "PCA", 4: "Vertebro-Basilar"}
    return lut

@st.cache_resource(show_spinner=False)
def load_atlas():
    # We use ProbArterialAtlas_average.nii as the moving (intensity) image for MI registration,
    # and ArterialAtlas_level2.nii as the discrete labelmap (nearest-neighbor resample).
    prob_path = ATLAS_DIR / "ProbArterialAtlas_average.nii"
    lvl2_path = ATLAS_DIR / "ArterialAtlas_level2.nii"
    fine_path = ATLAS_DIR / "ArterialAtlas.nii"  # optional (not used by default)
    labels_path = ATLAS_DIR / "ArterialAtlasLables.txt"

    ensure_exists(prob_path, "ATLAS/ProbArterialAtlas_average.nii")
    ensure_exists(lvl2_path, "ATLAS/ArterialAtlas_level2.nii")
    ensure_exists(labels_path, "ATLAS/ArterialAtlasLables.txt")

    moving_prob = sitk.ReadImage(str(prob_path))                # float
    atlas_lvl2 = sitk.ReadImage(str(lvl2_path), sitk.sitkUInt16)  # labels
    atlas_fine = sitk.ReadImage(str(fine_path), sitk.sitkUInt16) if fine_path.exists() else None
    lut = parse_labels_txt(labels_path)
    return moving_prob, atlas_lvl2, atlas_fine, lut

def affine_register_mni_to_patient(patient_img: sitk.Image, moving_mni_img: sitk.Image) -> sitk.Transform:
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-3, numberOfIterations=200)
    reg.SetOptimizerScalesFromPhysicalShift()
    # Multi-resolution for speed/stability
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    init = sitk.CenteredTransformInitializer(patient_img, moving_mni_img, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg.SetInitialTransform(init, inPlace=False)
    # moving = MNI prob, fixed = patient
    tx = reg.Execute(patient_img, moving_mni_img)
    return tx

def resample_to_patient(moving_img: sitk.Image, fixed_img: sitk.Image, transform: sitk.Transform, is_label=False) -> sitk.Image:
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    default_val = 0 if is_label else 0.0
    out = sitk.Resample(moving_img, fixed_img, transform, interp, default_val,
                        sitk.sitkUInt16 if is_label else sitk.sitkFloat32)
    return out

def annotate_with_atlas(patient_img: sitk.Image, mask_np: np.ndarray):
    moving_prob, atlas_lvl2, _, lut = load_atlas()
    tx = affine_register_mni_to_patient(patient_img, moving_prob)
    atlas_in_patient = resample_to_patient(atlas_lvl2, patient_img, tx, is_label=True)
    atlas_np = sitk.GetArrayFromImage(atlas_in_patient).astype(np.int32)
    spacing_zyx = (patient_img.GetSpacing()[2], patient_img.GetSpacing()[1], patient_img.GetSpacing()[0])
    _, lesions = components_3d(mask_np, spacing_zyx)
    for L in lesions:
        z, y, x = [int(round(v)) for v in L["centroid_zyx"]]
        z = np.clip(z, 0, atlas_np.shape[0]-1)
        y = np.clip(y, 0, atlas_np.shape[1]-1)
        x = np.clip(x, 0, atlas_np.shape[2]-1)
        lbl = int(atlas_np[z, y, x])
        L["territory_id"] = lbl
        L["territory_name"] = lut.get(lbl, str(lbl))
    return lesions, atlas_np

# ---------------------------- Session State ---------------------------
if "case_dir" not in st.session_state:
    st.session_state.case_dir = None
if "input_kind" not in st.session_state:
    st.session_state.input_kind = None  # "dcm_dir" or "nii"
if "patient_img" not in st.session_state:
    st.session_state.patient_img = None
if "vol_np" not in st.session_state:
    st.session_state.vol_np = None
if "spacing_zyx" not in st.session_state:
    st.session_state.spacing_zyx = None
if "pred_path" not in st.session_state:
    st.session_state.pred_path = None
if "mask_np" not in st.session_state:
    st.session_state.mask_np = None
if "lesions" not in st.session_state:
    st.session_state.lesions = []
if "atlas_np" not in st.session_state:
    st.session_state.atlas_np = None

# ------------------------------- Sidebar ------------------------------
st.sidebar.header("Settings")
ui_badge("GLIA‑Net repo", "#334155")
st.sidebar.code(str(GLIA_ROOT), language="bash")

# Device selection
try:
    import torch
    has_cuda = torch.cuda.is_available()
except Exception:
    has_cuda = False

device_choice = st.sidebar.selectbox(
    "Device",
    options=["Auto (GPU if available)", "CPU"] + ([f"GPU:{i}" for i in range(4)] if has_cuda else []),
    index=0
)

def resolve_device_string(choice: str) -> str:
    if choice.startswith("GPU:"):
        return choice.split(":")[1]  # "0", "1", ...
    if choice.startswith("Auto"):
        return "0" if has_cuda else "cpu"
    return "cpu"

# Input type
in_kind = st.sidebar.radio("Input type", ["DICOM (.zip of a series)", "NIfTI (.nii/.nii.gz)"], index=0)

# Uploaders
dicom_zip = None
nii_file = None
if in_kind.startswith("DICOM"):
    dicom_zip = st.sidebar.file_uploader("Upload DICOM series (.zip)", type=["zip"])
else:
    nii_file = st.sidebar.file_uploader("Upload NIfTI (.nii or .nii.gz)", type=["nii", "nii.gz"])

# Actions
colA, colB = st.sidebar.columns(2)
run_infer = colA.button("▶️ Run GLIA‑Net")
reset_case = colB.button("🧹 Reset")

# --------------------------- Reset Handling ---------------------------
if reset_case:
    # clean any previous temp dir
    if st.session_state.case_dir and Path(st.session_state.case_dir).exists():
        try:
            shutil.rmtree(st.session_state.case_dir, ignore_errors=True)
        except Exception:
            pass
    for k in ["case_dir","input_kind","patient_img","vol_np","spacing_zyx","pred_path","mask_np","lesions","atlas_np"]:
        st.session_state[k] = None if k != "lesions" else []
    st.experimental_rerun()

# ---------------------------- Load the Study --------------------------
with st.spinner("Preparing input..."):
    if dicom_zip is not None and in_kind.startswith("DICOM"):
        # Extract to a temp case dir
        if st.session_state.case_dir is None:
            st.session_state.case_dir = str(Path(tempfile.mkdtemp(prefix="cta_case_")).resolve())
        case_dir = Path(st.session_state.case_dir)
        series_dir = case_dir / "dicom_series"
        if series_dir.exists():
            shutil.rmtree(series_dir, ignore_errors=True)
        series_dir.mkdir(parents=True, exist_ok=True)
        # Unzip
        with zipfile.ZipFile(io.BytesIO(dicom_zip.getvalue())) as zf:
            zf.extractall(series_dir)
        # Load image via SITK series reader
        img = read_patient_image("dcm_dir", series_dir)
        st.session_state.patient_img = img
        st.session_state.vol_np, st.session_state.spacing_zyx = sitk_to_numpy_and_spacing(img)
        st.session_state.input_kind = "dcm_dir"
        st.session_state.pred_path = None
        st.session_state.mask_np = None
        st.session_state.lesions = []
        st.session_state.atlas_np = None

    elif nii_file is not None and in_kind.startswith("NIfTI"):
        if st.session_state.case_dir is None:
            st.session_state.case_dir = str(Path(tempfile.mkdtemp(prefix="cta_case_")).resolve())
        case_dir = Path(st.session_state.case_dir)
        nii_path = case_dir / "input.nii.gz"
        with open(nii_path, "wb") as f:
            f.write(nii_file.getvalue())
        img = read_patient_image("nii", nii_path)
        st.session_state.patient_img = img
        st.session_state.vol_np, st.session_state.spacing_zyx = sitk_to_numpy_and_spacing(img)
        st.session_state.input_kind = "nii"
        st.session_state.pred_path = None
        st.session_state.mask_np = None
        st.session_state.lesions = []
        st.session_state.atlas_np = None

# ------------------------------- Header -------------------------------
st.title("CTA Aneurysm — GLIA‑Net + Arterial Atlas")
left, right = st.columns([1.1, 1])

# ----------------------------- Main Viewer ----------------------------
with left:
    st.subheader("Study")
    if st.session_state.vol_np is None:
        st.info("Upload a **DICOM .zip** or a **NIfTI** to begin.")
    else:
        vol = st.session_state.vol_np
        D, H, W = vol.shape
        spacing = st.session_state.spacing_zyx
        st.caption(f"Volume: **{D}×{H}×{W}** vox • Spacing (mm): **{spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f}**")

        # WL/WW controls
        c1, c2, c3 = st.columns([1,1,1])
        wl = c1.slider("Window Level (WL)", -200.0, 200.0, 40.0, 1.0)
        ww = c2.slider("Window Width (WW)", 100.0, 3000.0, 400.0, 10.0)
        show_boxes = c3.checkbox("Show lesion boxes & IDs", value=True)

        # Slice slider
        z = st.slider("Axial slice", 0, D-1, D//2, 1)

        # Determine mask on this slice (if present)
        mask_slice = None
        if st.session_state.mask_np is not None and z < st.session_state.mask_np.shape[0]:
            mask_slice = st.session_state.mask_np[z]
        else:
            mask_slice = np.zeros((H, W), dtype=np.uint8)

        # Filter lesions on this slice
        lesions_here = []
        for L in (st.session_state.lesions or []):
            z0,y0,x0,z1,y1,x1 = L["bbox_zyx"]
            if z0 <= z < z1:
                lesions_here.append(L)

        img_u8 = window_ct(vol[z], wl, ww)
        rgb = draw_overlay(img_u8, mask_slice, lesions_here=lesions_here, show_boxes=show_boxes)
        st.image(rgb, caption=f"Axial slice {z}", use_column_width=True)

with right:
    st.subheader("Pipeline")
    # Show model + atlas status
    ok_ckpt = CHECKPOINT_FILE.exists()
    ok_atlas = all((ATLAS_DIR / name).exists() for name in ["ArterialAtlasLables.txt","ArterialAtlas_level2.nii","ProbArterialAtlas_average.nii"])
    cols = st.columns(2)
    cols[0].markdown("**GLIA‑Net weights**")
    cols[0].write(("✅ Found" if ok_ckpt else "❌ Missing"))
    cols[1].markdown("**Atlas files**")
    cols[1].write(("✅ Found" if ok_atlas else "❌ Missing"))

    add_labels = st.checkbox("Add atlas labels (ACA/MCA/PCA/VB)", value=True, disabled=not ok_atlas)

    # Run inference
    if run_infer and st.session_state.vol_np is not None:
        try:
            device_str = resolve_device_string(device_choice)
            with st.spinner(f"Running GLIA‑Net on {device_str}…"):
                case_dir = Path(st.session_state.case_dir or tempfile.mkdtemp(prefix="cta_case_"))
                out_dir = case_dir / "preds"
                inp_kind = st.session_state.input_kind
                if inp_kind == "dcm_dir":
                    input_path = Path(case_dir) / "dicom_series"
                    input_type = "dcm"
                else:
                    input_path = Path(case_dir) / "input.nii.gz"
                    input_type = "nii"
                pred_path = run_glianet_inference(input_path, input_type, device_str, out_dir)
                st.session_state.pred_path = str(pred_path)

            # Load predicted mask and align to patient grid if needed
            with st.spinner("Loading prediction…"):
                pred_img = sitk.ReadImage(st.session_state.pred_path)
                patient_img = st.session_state.patient_img
                if pred_img.GetSize() != patient_img.GetSize() or \
                   pred_img.GetSpacing() != patient_img.GetSpacing() or \
                   pred_img.GetOrigin()  != patient_img.GetOrigin()  or \
                   pred_img.GetDirection()!= patient_img.GetDirection():
                    # Resample with identity to patient grid
                    pred_img = resample_to_patient(pred_img, patient_img, sitk.Transform(), is_label=True)
                mask_np = sitk.GetArrayFromImage(pred_img).astype(np.uint8)
                st.session_state.mask_np = (mask_np > 0).astype(np.uint8)

            # Lesion analysis (and optional atlas)
            with st.spinner("Computing lesion summaries…"):
                if add_labels and ok_atlas:
                    lesions, atlas_np = annotate_with_atlas(st.session_state.patient_img, st.session_state.mask_np)
                    st.session_state.atlas_np = atlas_np
                else:
                    _, lesions = components_3d(st.session_state.mask_np, st.session_state.spacing_zyx)

                st.session_state.lesions = lesions

        except Exception as e:
            st.exception(e)
            st.error("Inference failed. See the exception above.")
            st.stop()

    # Results
    if st.session_state.mask_np is not None:
        st.markdown("**Detections**")
        if not st.session_state.lesions:
            st.info("No lesions found in the prediction mask.")
        else:
            rows = []
            for L in st.session_state.lesions:
                rows.append({
                    "ID": L["id"],
                    "Territory": L.get("territory_name", ""),
                    "Centroid (z,y,x)": tuple(round(v,1) for v in L["centroid_zyx"]),
                    "Volume (mm³)": round(L["volume_mm3"], 1),
                    "Equiv. diameter (mm)": round(L["equiv_diam_mm"], 1),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            # Downloads
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download lesions CSV", data=csv_bytes, file_name="lesions.csv", mime="text/csv")

        # Mask download
        if st.session_state.pred_path and Path(st.session_state.pred_path).exists():
            with open(st.session_state.pred_path, "rb") as f:
                st.download_button("⬇️ Download mask (NIfTI)", data=f.read(), file_name=Path(st.session_state.pred_path).name, mime="application/gzip")

# ----------------------------- Footer Help ----------------------------
with st.expander("ℹ️ Help & Notes"):
    st.markdown(
        """
- **Inputs**: Upload either a **DICOM .zip** (one series) or a **NIfTI** file of your CTA.
- **Inference**: Click **Run GLIA‑Net**. The app loads `PRETRAINED/checkpoint-0245700.pt` and saves a mask NIfTI under a temp case folder.
- **Atlas labels**: When enabled, the app affine‑registers the atlas (MNI) to the patient CTA using mutual information on `ProbArterialAtlas_average.nii`, then samples `ArterialAtlas_level2.nii` at each lesion centroid to label the territory (ACA/MCA/PCA/VB).
- **Display**: Use the **Axial slice** slider to scroll. Red = mask contour; green = lesion box with ID (and territory if enabled).
- **CPU/GPU**: On GPU hosts, choose **GPU:0** (or Auto). On CPU, it still works—just slower.
- **Troubleshooting**:
  - If you see *“Missing GLIA‑Net repo folder”*, make sure the GLIA repo is checked out as `glianet/` or `GLIA-Net/` next to this file.
  - If checkpoint isn’t found, verify the file at `PRETRAINED/checkpoint-0245700.pt`.
  - If DICOM upload fails, ensure the .zip contains only the target series (or the first found series is OK).
"""
    )
