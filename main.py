import argparse
import os
import sys
import torch

from core.darknet_parser import DarknetParser
from exporters.engine import export_pytorch_to_onnx, try_export_ultralytics
from exporters.post_process import optimize_for_npu
from validators.compare import validate_conversion

def show_tutorial():
    tutorial_path = os.path.join(os.path.dirname(__file__), 'tutorial.txt')
    if os.path.exists(tutorial_path):
        with open(tutorial_path, 'r') as f:
            print(f.read())
    else:
        print("[ERROR] Tutorial file not found.")

def get_args():
    parser = argparse.ArgumentParser(
        description="Universal ONNX Converter: Optimize models for NPU deployment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    group = parser.add_argument_group('Required Arguments')
    group.add_argument('--mode', type=str, choices=['darknet', 'pytorch'], 
                        help='Input model type:\n'
                             '  darknet: Use .cfg and .weights\n'
                             '  pytorch: Use .pt file')
    
    group = parser.add_argument_group('Configuration')
    group.add_argument('--weights', type=str, help='Path to weight file (.weights or .pt)')
    group.add_argument('--cfg', type=str, help='Path to .cfg file (Darknet mode only)')
    group.add_argument('--output', type=str, default='model.onnx', help='Output path (default: model.onnx)')
    group.add_argument('--output_mode', type=str, default='onnx', choices=['onnx', 'pt'],
                        help='Output format:\n'
                             '  onnx: NPU optimized Opset 11\n'
                             '  pt  : Standard PyTorch file')
    group.add_argument('--shape', type=int, nargs=4, default=[1, 3, 416, 416], 
                        help='Input shape: B C H W (default: 1 3 416 416)')
    
    group = parser.add_argument_group('Advanced Flags')
    group.add_argument('--validate', action='store_true', help='Compare PyTorch and ONNX outputs')
    group.add_argument('--no-simplify', action='store_true', help='Skip ONNX optimization step')
    group.add_argument('--tutorial', action='store_true', help='Show usage examples and exit')

    return parser.parse_args()

def load_model(args):
    print(f"[INFO] Loading {args.mode.upper()} model...")
    model = None
    
    if args.mode == 'darknet':
        if not args.cfg:
            print("[ERROR] Darknet mode requires a --cfg file.")
            sys.exit(1)
        try:
            model = DarknetParser(args.cfg)
            if args.weights:
                model.load_weights(args.weights)
            else:
                print("[INFO] No weights provided. Using random initialization.")
            print(f"[SUCCESS] Darknet model is ready.")
        except Exception as e:
            print(f"[ERROR] Could not load Darknet model: {e}")
            sys.exit(1)

    elif args.mode == 'pytorch':
        if not args.weights:
            print("[ERROR] PyTorch mode requires a --weights file.")
            sys.exit(1)
            
        print(f"[INFO] Reading file: {args.weights}")
        try:
            try:
                loaded = torch.load(args.weights, map_location='cpu', weights_only=False)
            except TypeError:
                loaded = torch.load(args.weights, map_location='cpu')

            if isinstance(loaded, dict):
                if 'model' in loaded:
                    model = loaded['model']
                    if hasattr(model, 'float'): model.float()
                elif 'state_dict' in loaded:
                    print("[ERROR] Found only state_dict. Model architecture is missing.")
                    sys.exit(1)
                else:
                    model = loaded
            else:
                model = loaded
            
            if hasattr(model, 'fuse'):
                try: model.fuse()
                except: pass
                
            print("[SUCCESS] PyTorch model is ready.")
        except Exception as e:
            print(f"[ERROR] Could not load PyTorch model: {e}")
            sys.exit(1)

    if hasattr(model, 'eval'):
        model.eval()
    return model

def main():
    args = get_args()
    
    if args.tutorial:
        show_tutorial()
        return

    if not args.mode:
        print("[ERROR] Please specify --mode or use --tutorial.")
        return

    model = load_model(args)

    if args.output_mode == 'pt':
        print(f"[INFO] Saving model to PyTorch format...")
        save_path = args.output if args.output.endswith('.pt') else args.output + '.pt'
        torch.save(model, save_path)
        print(f"[SUCCESS] Model saved to {save_path}")

    elif args.output_mode == 'onnx':
        print(f"[INFO] Starting ONNX export process...")
        save_path = args.output if args.output.endswith('.onnx') else args.output + '.onnx'

        try:
            is_ultralytics = try_export_ultralytics(model, args.shape, save_path)
            
            if not is_ultralytics:
                print("[INFO] Standard model detected. Using generic export.")
                export_pytorch_to_onnx(model, args.shape, save_path)
            
            print("[INFO] Running NPU optimization patches...")
            final_path = optimize_for_npu(save_path, simplify=not args.no_simplify)
            
            if args.validate:
                print("[INFO] Validating conversion...")
                is_valid = validate_conversion(model, final_path, args.shape)
                if is_valid:
                    print(f"[SUCCESS] Validation passed. Model is ready for NPU.")
                else:
                    print(f"[WARNING] Validation completed with mismatches. Check the details above.")
            else:
                print(f"[SUCCESS] Export finished: {final_path}")
                
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()