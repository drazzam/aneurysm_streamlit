#!/usr/bin/env python3
"""
CTA Aneurysm Detection - GLIA-Net + Arterial Atlas
Complete Gradio app for Hugging Face Spaces deployment

This app detects intracranial aneurysms in CTA images using GLIA-Net
and provides arterial territory labeling using an anatomical atlas.
"""

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
import gradio as gr
import SimpleITK as sitk
from pathlib import Path
from skimage.measure import label, regionprops
import gc
import torch
import traceback

# ----------------------------- App Configuration -----------------------------
APP_ROOT = Path.cwd()
ATLAS_DIR = APP_ROOT / "ATLAS"
PRETRAINED_DIR = APP_ROOT / "PRETRAINED"
CHECKPOINT_FILE = PRETRAINED_DIR / "checkpoint-0245700.pt"
OUTPUTS_BASE = APP_ROOT / "outputs"
OUTPUTS_BASE.mkdir(exist_ok=True, parents=True)

# ----------------------------- Locate GLIA‑Net Code -------------------------
def find_glia_root() -> Path | None:
    """Locate the GLIA-Net source code directory."""
    candidates = [
        APP_ROOT / "glianet",
        APP_ROOT / "GLIA-Net", 
        APP_ROOT / "GLIA-Net-master",
    ]
    
    # Also scan for any directory with GLIA in the name
    for p in APP_ROOT.iterdir():
        if p.is_dir() and "GLIA" in p.name.upper():
            candidates.append(p)
    
    for c in candidates:
        if (c / "inference.py").exists() and (c / "utils" / "project_utils.py").exists():
            return c.resolve()
    return None

GLIA_ROOT = find_glia_root()
if GLIA_ROOT is None:
    raise RuntimeError("❌ Could not locate GLIA-Net repo folder. Expected 'glianet/' directory.")

# Add to Python path
sys.path.insert(0, str(GLIA_ROOT))

# Import GLIA-Net modules
try:
    from utils.project_utils import load_config, get_logger, get_devices
    from core import Inferencer
    from data_loader import AneurysmSegTestManager
    import torch
except Exception as e:
    raise RuntimeError(f"❌ Failed to import GLIA-Net modules: {e}")

# ----------------------------- Utility Functions ----------------------------
def ensure_exists(path: Path, what: str):
    """Ensure a required file/directory exists."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {what} at: {path}")

def find_dicom_directory(root_path: Path) -> Path:
    """Recursively search for a directory containing DICOM files."""
    reader = sitk.ImageSeriesReader()
    
    # Check if root_path itself contains DICOM files
    try:
        files_in_root = [f for f in root_path.iterdir() if f.is_file()]
        if files_in_root:
            series_ids = reader.GetGDCMSeriesIDs(str(root_path))
            if series_ids:
                return root_path
    except Exception:
        pass
    
    # Search subdirectories recursively
    for item in root_path.rglob("*"):
        if item.is_dir():
            try:
                files_in_dir = [f for f in item.iterdir() if f.is_file()]
                if files_in_dir:
                    series_ids = reader.GetGDCMSeriesIDs(str(item))
                    if series_ids:
                        return item
            except Exception:
                continue
    
    raise RuntimeError(f"No DICOM series found in the uploaded zip file")

def read_patient_image(input_kind: str, path: Path) -> sitk.Image:
    """Read DICOM series or NIfTI file and return SimpleITK image."""
    if input_kind == "dcm_dir":
        dicom_dir = find_dicom_directory(path)
        reader = sitk.ImageSeriesReader()
        reader.SetGlobalWarningDisplay(False)
        reader.SetMetaDataDictionaryArrayUpdate(True)
        reader.SetLoadPrivateTags(False)
        
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        if not series_ids:
            raise RuntimeError(f"No DICOM series found in {dicom_dir}")
        
        if len(series_ids) > 1:
            print(f"Found {len(series_ids)} series, using the first one")
        
        file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        file_names = [f for f in file_names if Path(f).is_file()]
        
        if not file_names:
            raise RuntimeError("No valid DICOM files found in the series")
        
        reader.SetFileNames(file_names)
        try:
            img = reader.Execute()
            print(f"✅ Successfully loaded DICOM series with {len(file_names)} slices")
            return img
        except Exception as e:
            raise RuntimeError(f"Failed to read DICOM series: {str(e)}")
            
    elif input_kind == "nii":
        try:
            img = sitk.ReadImage(str(path))
            print(f"✅ Successfully loaded NIfTI file: {path.name}")
            return img
        except Exception as e:
            raise RuntimeError(f"Failed to read NIfTI file: {str(e)}")
    else:
        raise ValueError("input_kind must be 'dcm_dir' or 'nii'")

def sitk_to_numpy_and_spacing(img: sitk.Image):
    """Convert SimpleITK image to numpy array and extract spacing."""
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [Z,Y,X]
    sx, sy, sz = img.GetSpacing()  # SimpleITK spacing order is (X,Y,Z)
    spacing_zyx = (sz, sy, sx)  # Convert to (Z,Y,X) order for numpy array
    return arr, spacing_zyx

def window_ct(img_array: np.ndarray, wl=40.0, ww=400.0) -> np.ndarray:
    """Apply windowing to CT image for display."""
    lo, hi = wl - ww/2.0, wl + ww/2.0
    v = np.clip(img_array, lo, hi)
    v = (v - lo) / max(hi - lo, 1e-6)
    return (v * 255.0).astype(np.uint8)

def components_3d(mask_np: np.ndarray, spacing_zyx):
    """Extract 3D connected components and compute properties."""
    lab = label(mask_np > 0, connectivity=1)
    props = regionprops(lab)
    lesions = []
    dz, dy, dx = spacing_zyx
    
    for p in props:
        zc, yc, xc = p.centroid
        vol_mm3 = float(p.area) * dz * dy * dx
        equiv_d = ((6.0 * vol_mm3 / np.pi) ** (1.0 / 3.0))
        lesions.append({
            "id": int(p.label),
            "centroid_zyx": (float(zc), float(yc), float(xc)),
            "bbox_zyx": tuple(int(v) for v in p.bbox),
            "volume_mm3": float(vol_mm3),
            "equiv_diam_mm": float(equiv_d)
        })
    return lab, lesions

def draw_overlay(gray_u8: np.ndarray, mask_slice: np.ndarray, lesions_here=None) -> np.ndarray:
    """Draw mask contours and lesion bounding boxes on grayscale image."""
    rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    
    # Draw mask contours in red
    m = (mask_slice > 0).astype(np.uint8) * 255
    if m.any():
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (0, 0, 255), 2)  # Red contours
    
    # Draw lesion bounding boxes and labels in green
    if lesions_here:
        for L in lesions_here:
            y0, x0 = int(L["bbox_zyx"][1]), int(L["bbox_zyx"][2])
            y1, x1 = int(L["bbox_zyx"][4]-1), int(L["bbox_zyx"][5]-1)
            cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 255, 0), 2)
            
            label_txt = f"ID {L['id']}"
            if "territory_name" in L and L["territory_name"]:
                label_txt += f" • {L['territory_name']}"
            
            cv2.putText(rgb, label_txt, (x0, max(15, y0-5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    return rgb

# ----------------------------- GLIA-Net Inference ----------------------------
def run_glianet_inference(input_path: Path, input_type: str, output_dir: Path) -> Path:
    """Run GLIA-Net inference and return path to prediction file."""
    ensure_exists(CHECKPOINT_FILE, "GLIA-Net checkpoint")
    
    # Load and modify config
    cfg_path = GLIA_ROOT / "configs" / "inference_GLIA-Net.yaml"
    config = load_config(cfg_path)
    
    # Update config for our setup
    config['ckpt_folder'] = str(PRETRAINED_DIR)
    config['ckpt_file'] = CHECKPOINT_FILE.name
    
    # Memory optimizations for Hugging Face Spaces
    config['data']['num_proc_workers'] = 0
    config['data']['num_io_workers'] = 0
    config['data']['overlap_step'] = [72, 72, 72]  # Larger overlap to reduce patches
    config['train']['batch_size'] = 1
    
    logger = get_logger("GLIA-Infer", logging_folder=None, verbose=False)
    devices = get_devices("cpu", logger)  # Force CPU for stability
    
    # Create test manager and output directory
    test_mgr = AneurysmSegTestManager(config, logger, devices)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and run inferencer
    infer = Inferencer(
        config=config,
        exp_path=str(APP_ROOT.resolve()),
        devices=devices,
        inference_file_or_folder=str(input_path.resolve()),
        output_folder=str(output_dir.resolve()),
        input_type=input_type,
        save_binary=True,
        save_prob=False,
        save_global=False,
        test_loader_manager=test_mgr,
        logger=logger
    )
    
    infer.inference()
    
    # Clean up memory
    del infer, test_mgr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Find and return prediction file
    pred_files = list(output_dir.glob("*.nii")) + list(output_dir.glob("*.nii.gz"))
    if not pred_files:
        raise RuntimeError("GLIA-Net inference did not produce any prediction files")
    
    return max(pred_files, key=lambda p: p.stat().st_mtime)

# ----------------------------- Atlas Integration ----------------------------
def parse_labels_txt(path: Path) -> dict:
    """Parse atlas labels file."""
    lut = {1: "ACA", 2: "MCA", 3: "PCA", 4: "Vertebro-Basilar"}  # Default fallback
    
    if not path.exists():
        return lut
    
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                
                # Parse line: "ID name..."
                parts = [p for p in s.replace(",", " ").split() if p]
                try:
                    k = int(parts[0])
                    name = " ".join(parts[1:]) if len(parts) > 1 else str(k)
                    lut[k] = name
                except Exception:
                    continue
    except Exception:
        pass
    
    return lut

def load_atlas():
    """Load atlas files if available."""
    prob_path = ATLAS_DIR / "ProbArterialAtlas_average.nii"
    lvl2_path = ATLAS_DIR / "ArterialAtlas_level2.nii"
    labels_path = ATLAS_DIR / "ArterialAtlasLables.txt"
    
    # Check if all required atlas files exist
    if not all(p.exists() for p in [prob_path, lvl2_path, labels_path]):
        return None, None, None
    
    try:
        moving_prob = sitk.ReadImage(str(prob_path))
        atlas_lvl2 = sitk.ReadImage(str(lvl2_path), sitk.sitkUInt16)
        lut = parse_labels_txt(labels_path)
        return moving_prob, atlas_lvl2, lut
    except Exception:
        return None, None, None

def annotate_with_atlas(patient_img: sitk.Image, mask_np: np.ndarray):
    """Add atlas-based territory labels to lesions."""
    # Load atlas
    atlas_data = load_atlas()
    spacing_zyx = (patient_img.GetSpacing()[2], patient_img.GetSpacing()[1], patient_img.GetSpacing()[0])
    _, lesions = components_3d(mask_np, spacing_zyx)
    
    if atlas_data[0] is None:
        # No atlas available
        return lesions, None
    
    moving_prob, atlas_lvl2, lut = atlas_data
    
    # Perform registration (simplified for robustness)
    try:
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(32)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-3, 200)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        init = sitk.CenteredTransformInitializer(
            patient_img, moving_prob, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        reg.SetInitialTransform(init, inPlace=False)
        
        tx = reg.Execute(patient_img, moving_prob)
        atlas_in_patient = sitk.Resample(atlas_lvl2, patient_img, tx, 
                                        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
        atlas_np = sitk.GetArrayFromImage(atlas_in_patient).astype(np.int32)
        
        # Add territory labels to lesions
        for L in lesions:
            z, y, x = [int(round(v)) for v in L["centroid_zyx"]]
            z = np.clip(z, 0, atlas_np.shape[0]-1)
            y = np.clip(y, 0, atlas_np.shape[1]-1) 
            x = np.clip(x, 0, atlas_np.shape[2]-1)
            lbl = int(atlas_np[z, y, x])
            L["territory_id"] = lbl
            L["territory_name"] = lut.get(lbl, str(lbl))
            
    except Exception as e:
        print(f"Atlas registration failed: {e}")
        atlas_np = None
    
    return lesions, atlas_np

# ----------------------------- Main Processing Function ------------------
def process_medical_image(file_input, input_type, add_labels, wl, ww):
    """Main processing function called by Gradio interface."""
    if file_input is None:
        return None, None, "❌ Please upload a file first.", None, gr.update()
    
    try:
        # Create temporary directory for this case
        case_dir = Path(tempfile.mkdtemp(prefix="cta_case_"))
        
        # Handle different input types
        if input_type == "DICOM (.zip)":
            # Extract DICOM zip file
            series_dir = case_dir / "dicom_series"
            series_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(file_input.name) as zf:
                zf.extractall(series_dir)
            
            img = read_patient_image("dcm_dir", series_dir)
            input_kind = "dcm"
            input_path = find_dicom_directory(series_dir)
            
        else:  # NIfTI
            nii_path = case_dir / "input.nii.gz"
            shutil.copy2(file_input.name, nii_path)
            img = read_patient_image("nii", nii_path)
            input_kind = "nii"
            input_path = nii_path
        
        # Extract volume data and properties
        vol_np, spacing_zyx = sitk_to_numpy_and_spacing(img)
        D, H, W = vol_np.shape
        
        # Run GLIA-Net inference
        output_dir = case_dir / "preds"
        yield None, None, f"🔄 Running GLIA-Net inference...\n📊 Volume: {D}×{H}×{W} voxels\n⏱️ This may take 5-10 minutes", None, gr.update()
        
        pred_path = run_glianet_inference(input_path, input_kind, output_dir)
        
        # Load prediction and align with patient image
        pred_img = sitk.ReadImage(str(pred_path))
        if (pred_img.GetSize() != img.GetSize() or 
            pred_img.GetSpacing() != img.GetSpacing() or
            pred_img.GetOrigin() != img.GetOrigin() or
            pred_img.GetDirection() != img.GetDirection()):
            pred_img = sitk.Resample(pred_img, img, sitk.Transform(), 
                                   sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
        
        mask_np = sitk.GetArrayFromImage(pred_img).astype(np.uint8)
        mask_np = (mask_np > 0).astype(np.uint8)
        
        # Analyze lesions and add atlas labels if requested
        yield None, None, f"🔄 Analyzing detected lesions...", None, gr.update()
        
        if add_labels:
            lesions, atlas_np = annotate_with_atlas(img, mask_np)
        else:
            spacing_zyx = (img.GetSpacing()[2], img.GetSpacing()[1], img.GetSpacing()[0])
            _, lesions = components_3d(mask_np, spacing_zyx)
            atlas_np = None
        
        # Create results table
        if lesions:
            results_data = []
            for L in lesions:
                results_data.append({
                    "ID": L["id"],
                    "Territory": L.get("territory_name", ""),
                    "Centroid (z,y,x)": f"({L['centroid_zyx'][0]:.1f}, {L['centroid_zyx'][1]:.1f}, {L['centroid_zyx'][2]:.1f})",
                    "Volume (mm³)": round(L["volume_mm3"], 1),
                    "Equivalent Diameter (mm)": round(L["equiv_diam_mm"], 1),
                })
            results_df = pd.DataFrame(results_data)
        else:
            results_df = pd.DataFrame({"Status": ["No aneurysms detected"]})
        
        # Store case data for slice viewer
        case_data = {
            'vol_np': vol_np,
            'mask_np': mask_np,
            'lesions': lesions,
            'spacing_zyx': spacing_zyx,
            'pred_path': pred_path,
            'case_dir': case_dir
        }
        
        # Create initial slice image
        mid_slice = D // 2
        slice_img = create_slice_image(case_data, mid_slice, wl, ww)
        
        # Success message
        status_msg = f"""✅ Analysis Complete!
        
📊 **Image Properties:**
• Volume: {D} × {H} × {W} voxels  
• Spacing: {spacing_zyx[0]:.2f} × {spacing_zyx[1]:.2f} × {spacing_zyx[2]:.2f} mm

🎯 **Detection Results:**
• Found {len(lesions)} lesions
• Atlas labels: {'Enabled' if add_labels else 'Disabled'}

🔍 **Instructions:**
• Use the slice slider to navigate through the volume
• Red contours show detected aneurysms
• Green boxes show lesion IDs and territories
• Adjust window level/width for better contrast
"""
        
        yield case_data, slice_img, status_msg, results_df, gr.update(maximum=D-1, value=mid_slice, visible=True)
        
    except Exception as e:
        error_msg = f"❌ **Error during processing:**\n\n{str(e)}\n\n**Troubleshooting:**\n• Ensure file is a valid DICOM zip or NIfTI\n• Check file size is under 500MB\n• Try with a different file format"
        yield None, None, error_msg, None, gr.update(visible=False)

def create_slice_image(case_data, slice_idx, wl, ww):
    """Create slice image with mask overlays."""
    if case_data is None:
        return None
    
    vol_np = case_data['vol_np']
    mask_np = case_data['mask_np']
    lesions = case_data['lesions']
    
    # Ensure valid slice index
    if slice_idx >= vol_np.shape[0]:
        slice_idx = vol_np.shape[0] - 1
    elif slice_idx < 0:
        slice_idx = 0
    
    # Get slice and corresponding mask
    img_slice = vol_np[slice_idx]
    mask_slice = mask_np[slice_idx] if slice_idx < mask_np.shape[0] else np.zeros_like(img_slice)
    
    # Find lesions that intersect this slice
    lesions_here = []
    for L in lesions:
        z0, y0, x0, z1, y1, x1 = L["bbox_zyx"]
        if z0 <= slice_idx < z1:
            lesions_here.append(L)
    
    # Apply windowing and create overlay
    img_u8 = window_ct(img_slice, wl, ww)
    rgb = draw_overlay(img_u8, mask_slice, lesions_here)
    
    return rgb

def update_slice_view(case_data, slice_idx, wl, ww):
    """Update slice view when slider or window settings change."""
    return create_slice_image(case_data, slice_idx, wl, ww)

def download_prediction_file(case_data):
    """Prepare prediction mask for download."""
    if case_data is None or 'pred_path' not in case_data:
        return None
    return str(case_data['pred_path'])

def download_results_csv(results_df):
    """Prepare results CSV for download."""
    if results_df is None or results_df.empty:
        return None
    
    csv_path = Path(tempfile.mktemp(suffix=".csv"))
    results_df.to_csv(csv_path, index=False)
    return str(csv_path)

# ----------------------------- Gradio Interface ---------------------------
def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px;
        margin: auto;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .status-box {
        background: #f0f8ff;
        border: 1px solid #add8e6;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(
        title="🧠 CTA Aneurysm Detection - GLIA-Net",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as app:
        
        # Header
        gr.Markdown(
            """
            # 🧠 CTA Aneurysm Detection - GLIA-Net + Arterial Atlas
            
            **AI-powered detection of intracranial aneurysms in CTA images**
            
            Upload a DICOM series (.zip) or NIfTI file to detect aneurysms using deep learning.
            """
        )
        
        # State to store case data
        case_data_state = gr.State(None)
        
        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):
                gr.Markdown("### 📁 **Input & Settings**")
                
                input_type = gr.Radio(
                    choices=["DICOM (.zip)", "NIfTI (.nii/.nii.gz)"],
                    value="DICOM (.zip)",
                    label="Input Type",
                    info="Select the format of your medical image file"
                )
                
                file_input = gr.File(
                    label="Upload Medical Image",
                    file_types=[".zip", ".nii", ".gz"],
                    file_count="single"
                )
                
                gr.Markdown("### ⚙️ **Processing Options**")
                
                add_labels = gr.Checkbox(
                    label="🏷️ Add arterial territory labels",
                    value=True,
                    info="Label detected aneurysms with arterial territories (ACA/MCA/PCA/VB)"
                )
                
                gr.Markdown("### 🖼️ **Display Settings**")
                
                with gr.Row():
                    wl = gr.Slider(
                        minimum=-200, maximum=200, value=40, step=1,
                        label="Window Level",
                        info="Adjust image brightness"
                    )
                    ww = gr.Slider(
                        minimum=100, maximum=3000, value=400, step=10,
                        label="Window Width", 
                        info="Adjust image contrast"
                    )
                
                process_btn = gr.Button(
                    "🚀 Run Analysis",
                    variant="primary",
                    size="lg"
                )
                
                # Status display
                status_display = gr.Markdown(
                    "👆 Upload a file and click '🚀 Run Analysis' to begin",
                    elem_classes=["status-box"]
                )
            
            # Right column - Visualization
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ **Interactive Slice Viewer**")
                
                slice_image = gr.Image(
                    label="Axial Slice with Aneurysm Detection",
                    type="numpy",
                    height=400,
                    show_label=True
                )
                
                slice_slider = gr.Slider(
                    minimum=0, maximum=100, value=50, step=1,
                    label="Slice Index",
                    info="Navigate through axial slices",
                    visible=False
                )
                
                gr.Markdown(
                    """
                    **Legend:**  
                    🔴 **Red contours** = Detected aneurysm regions  
                    🟢 **Green boxes** = Lesion IDs + Territory labels
                    """
                )
        
        # Results section
        gr.Markdown("### 📊 **Analysis Results**")
        
        results_table = gr.Dataframe(
            label="Detected Lesions Summary",
            wrap=True,
            interactive=False
        )
        
        # Download section
        gr.Markdown("### ⬇️ **Download Results**")
        
        with gr.Row():
            download_csv_btn = gr.Button("📄 Download CSV Report")
            download_mask_btn = gr.Button("🧠 Download Prediction Mask")
            
        with gr.Row():
            csv_download = gr.File(label="CSV Report", visible=False)
            mask_download = gr.File(label="Prediction Mask", visible=False)
        
        # Information section
        with gr.Accordion("ℹ️ **Information & Help**", open=False):
            gr.Markdown(
                """
                ### 📋 **Usage Instructions**
                1. **Upload** your CTA scan (DICOM zip or NIfTI file)
                2. **Select** processing options (territory labeling recommended)
                3. **Click** "🚀 Run Analysis" to start detection
                4. **Explore** results with the interactive slice viewer
                5. **Download** predictions and detailed reports
                
                ### ⚙️ **Technical Details**
                - **Model**: GLIA-Net (Global Localization based Intracranial Aneurysm Network)
                - **Processing Time**: ~5-10 minutes on CPU
                - **Max File Size**: 500MB per upload
                - **Supported Formats**: DICOM series (.zip) and NIfTI (.nii/.nii.gz)
                
                ### 🏥 **Arterial Territories**
                - **ACA**: Anterior Cerebral Artery
                - **MCA**: Middle Cerebral Artery  
                - **PCA**: Posterior Cerebral Artery
                - **VB**: Vertebro-Basilar system
                
                ### ⚠️ **Important Notice**
                This tool is for **research purposes only** and should not be used for clinical diagnosis.
                Always consult qualified medical professionals for medical decisions.
                
                ### 📚 **Citation**
                If you use this tool in your research, please cite:
                ```
                GLIA-Net: Global-to-local image analysis network for intracranial aneurysm detection
                Patterns, 2021
                ```
                """
            )
        
        # Event handlers
        process_btn.click(
            fn=process_medical_image,
            inputs=[file_input, input_type, add_labels, wl, ww],
            outputs=[case_data_state, slice_image, status_display, results_table, slice_slider]
        )
        
        # Update slice view when slider changes
        slice_slider.change(
            fn=update_slice_view,
            inputs=[case_data_state, slice_slider, wl, ww],
            outputs=[slice_image]
        )
        
        # Update slice view when window settings change
        wl.change(
            fn=update_slice_view,
            inputs=[case_data_state, slice_slider, wl, ww],
            outputs=[slice_image]
        )
        
        ww.change(
            fn=update_slice_view,
            inputs=[case_data_state, slice_slider, wl, ww],
            outputs=[slice_image]
        )
        
        # Download handlers
        download_csv_btn.click(
            fn=download_results_csv,
            inputs=[results_table],
            outputs=[csv_download]
        )
        
        download_mask_btn.click(
            fn=download_prediction_file,
            inputs=[case_data_state],
            outputs=[mask_download]
        )
    
    return app

# ----------------------------- Launch Application -------------------------
if __name__ == "__main__":
    # Verify required files exist
    try:
        ensure_exists(GLIA_ROOT, "GLIA-Net source code")
        ensure_exists(CHECKPOINT_FILE, "GLIA-Net checkpoint")
        print("✅ All required files found!")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please ensure all required files are present before launching.")
        exit(1)
    
    # Create and launch the Gradio app
    app = create_gradio_interface()
    
    # Launch configuration for Hugging Face Spaces
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        enable_queue=True,
        max_threads=1  # Limit concurrent users to avoid memory issues
    )
