import onnx
import os

def optimize_for_npu(onnx_path, simplify=True):
    TARGET_OPSET = 11
    TARGET_IR = 7
    
    model = onnx.load(onnx_path)
    
    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify
            print("[INFO] Simplifying ONNX model...")
            model, check = onnx_simplify(model)
            if not check:
                print("[WARNING] Simplification check failed but proceeding.")
        except ImportError:
            print("[WARNING] onnx-simplifier not found. Skipping simplification.")
    
    if model.ir_version > TARGET_IR:
        print(f"[INFO] Lowering IR version from v{model.ir_version} to v{TARGET_IR}")
        model.ir_version = TARGET_IR

    for opset_import in model.opset_import:
        if opset_import.domain == '' or opset_import.domain == 'ai.onnx':
            opset_import.version = TARGET_OPSET
            
    onnx.save(model, onnx_path)
    
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    
    print("[SUCCESS] Post-processing finished.")
    return onnx_path