import onnx
import os
from core.logger import log_info, log_warning, log_success

def optimize_for_npu(onnx_path, simplify=True):
    TARGET_OPSET = 11
    TARGET_IR = 7
    
    model = onnx.load(onnx_path)
    
    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify
            log_info("Simplifying ONNX model...")
            model, check = onnx_simplify(model)
            if not check:
                log_warning("Simplification check failed but proceeding.")
        except ImportError:
            log_warning("onnx-simplifier not found. Export will be less optimized.")
            log_info("You can install it with: pip install onnx-simplifier")
    
    if model.ir_version > TARGET_IR:
        log_info(f"Lowering IR version from v{model.ir_version} to v{TARGET_IR}")
        model.ir_version = TARGET_IR

    for opset_import in model.opset_import:
        if opset_import.domain == '' or opset_import.domain == 'ai.onnx':
            if opset_import.version != TARGET_OPSET:
                log_info(f"Forcing Opset version {opset_import.version} -> {TARGET_OPSET}")
            opset_import.version = TARGET_OPSET
            
    onnx.save(model, onnx_path)
    
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        log_info(f"Cleaning up external weights file: {data_file}")
        os.remove(data_file)
    
    log_success("Post-processing finished.")
    return onnx_path