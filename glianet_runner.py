import os, sys
from pathlib import Path

# Make sure Python can import the GLIA-Net code (adjust if you nest it)
GLIA_ROOT = Path(__file__).resolve().parent / "glianet"   # or whatever you named the folder
sys.path.insert(0, str(GLIA_ROOT))

from utils.project_utils import load_config, get_logger, get_devices
from core import Inferencer
from data_loader import AneurysmSegTestManager

def run_glianet(input_path: str,
                input_type: str = "dcm",        # "dcm" for a DICOM series folder, "nii" for .nii.gz
                output_dir: str = "outputs",    # where to write the prediction NIfTI
                device: str = "cpu",            # "cpu" or like "0"
                config_rel: str = "configs/inference_GLIA-Net.yaml",
                exp_path: str = ".",
                save_binary: bool = True,
                save_prob: bool = False,
                save_global: bool = False) -> str:
    """
    Returns the path to the saved prediction NIfTI for the input.
    If input is a DICOM series, the file name will be <series-folder-name>.nii.gz.
    If input is a single NIfTI, it will mirror the input name with _pred.nii.gz.
    """

    # Load config and logger
    cfg_path = str(GLIA_ROOT / config_rel)
    config = load_config(cfg_path)
    logger = get_logger("GLIA-Infer", logging_folder=None, verbose=False)

    # Devices
    devices = get_devices(device, logger)

    # Data manager (provides sliding windows, spacing, etc.)
    test_mgr = AneurysmSegTestManager(config, logger, devices)

    # Make sure output dir exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # exp_path base (ckpt_folder is relative to this)
    exp_path = str(Path(exp_path).resolve())

    # Build the inferencer
    infer = Inferencer(config=config,
                       exp_path=exp_path,
                       devices=devices,
                       inference_file_or_folder=str(Path(input_path).resolve()),
                       output_folder=str(Path(output_dir).resolve()),
                       input_type=input_type,
                       save_binary=save_binary,
                       save_prob=save_prob,
                       save_global=save_global,
                       test_loader_manager=test_mgr,
                       logger=logger)

    # Run inference (writes NIfTI(s) to output_dir)
    infer.inference()

    # Figure out the output filepath it just wrote
    input_path = Path(input_path)
    if input_type == "nii":
        out_name = input_path.name.replace(".nii.gz", "_pred.nii.gz")
    else:
        # DICOM → outputs/<series-folder-name>.nii.gz
        series_name = input_path.name if input_path.is_dir() else input_path.parent.name
        out_name = f"{series_name}.nii.gz"
    return str(Path(output_dir) / out_name)
