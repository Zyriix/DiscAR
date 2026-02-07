#!/usr/bin/env python3

import sys
from typing import List, Tuple

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    try:
        __import__(module_name)
        return True, "✅"
    except ImportError as e:
        pkg = package_name or module_name
        return False, f"❌ (install: pip install {pkg})"

def test_torch_cuda() -> Tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"✅ (CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s))"
        else:
            return False, "⚠️  CUDA not available (CPU-only)"
    except:
        return False, "❌"

def test_optional_import(module_name: str) -> Tuple[bool, str]:
    try:
        __import__(module_name)
        return True, "✅ Installed"
    except ImportError:
        return True, "⚠️  Not installed (optional)"

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def main():
    print("\n🚀 DiscAR Installation Test")
    print("="*60)
    
    print_section("System Information")
    print(f"Python version: {sys.version}")
    
    print_section("Core Dependencies (Required)")
    
    tests = [
        ("torch", "torch (PyTorch)"),
        ("torchvision", "torchvision"),
        ("lightning", "lightning"),
        ("wandb", "wandb"),
        ("einops", "einops"),
        ("omegaconf", "omegaconf"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("tensorflow", "tensorflow"),
        ("PIL", "pillow"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
        ("requests", "requests"),
        ("timm", "timm"),
        ("click", "click"),
        ("pandas", "pandas"),
        ("cv2", "opencv-python"),
    ]
    
    failed_core = []
    for module, display_name in tests:
        success, status = test_import(module, display_name.split()[0])
        print(f"{display_name:.<40} {status}")
        if not success:
            failed_core.append(display_name)
    
    print(f"\n{'PyTorch CUDA Support':.<40}", end=" ")
    cuda_ok, cuda_msg = test_torch_cuda()
    print(cuda_msg)
    
    print_section("Optional Dependencies (Performance)")
    
    optional_tests = [
        ("flash_attn", "flash-attn"),
        ("triton", "triton"),
        ("xformers", "xformers"),
        ("kornia", "kornia"),
        ("thop", "thop"),
    ]
    
    for module, display_name in optional_tests:
        success, status = test_optional_import(module)
        print(f"{display_name:.<40} {status}")
    
    print_section("DiscAR Modules")
    
    discar_tests = [
        ("models", "models.py"),
        ("dataset", "dataset.py"),
        ("lpips", "lpips.py"),
        ("gan_loss", "gan_loss.py"),
        ("evaluator", "evaluator.py"),
    ]
    
    failed_discar = []
    for module, display_name in discar_tests:
        success, status = test_import(module)
        print(f"{display_name:.<40} {status}")
        if not success:
            failed_discar.append(display_name)
    
    print_section("Functional Tests")
    
    try:
        from models import Encoder, Decoder, ARModel
        print(f"{'Model classes import':.<40} ✅")
        model_import_ok = True
    except Exception as e:
        print(f"{'Model classes import':.<40} ❌ ({str(e)[:30]}...)")
        model_import_ok = False
    
    try:
        import torch
        dummy = torch.randn(1, 3, 32, 32)
        if torch.cuda.is_available():
            dummy = dummy.cuda()
        print(f"{'Tensor operations':.<40} ✅")
        tensor_ok = True
    except Exception as e:
        print(f"{'Tensor operations':.<40} ❌ ({str(e)[:30]}...)")
        tensor_ok = False
    
    print_section("Test Summary")
    
    all_ok = (
        len(failed_core) == 0 and 
        len(failed_discar) == 0 and 
        model_import_ok and 
        tensor_ok
    )
    
    if all_ok:
        print("\n🎉 All tests passed! Environment is configured correctly.")
        if not cuda_ok:
            print("⚠️  Warning: CUDA not available, training will be slow on CPU")
        print("\nYou can start training:")
        print("  python train.py --config=configs/CIFAR10_VQ_ae.yaml")
    else:
        print("\n⚠️  Some tests failed, please check:")
        
        if failed_core:
            print("\nMissing core dependencies:")
            for pkg in failed_core:
                print(f"  - {pkg}")
            print("\nInstall command:")
            print(f"  pip install {' '.join([p.split()[0] for p in failed_core])}")
        
        if failed_discar:
            print("\nFailed to import DiscAR modules:")
            for mod in failed_discar:
                print(f"  - {mod}")
            print("\nMake sure you run this script in the DiscAR project directory")
        
        if not model_import_ok:
            print("\nModel import failed, check models.py for errors")
        
        if not tensor_ok:
            print("\nTensor operations failed, check PyTorch installation")
    
    print("\n" + "="*60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
