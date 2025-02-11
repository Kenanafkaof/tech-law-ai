# check_cuda.py
import torch
import sys
import subprocess
import pkg_resources

def check_cuda():
    print("\nSystem CUDA Check:")
    print("-" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"\nCUDA devices found: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {props.name}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Current device: {'Yes' if i == torch.cuda.current_device() else 'No'}")
            
            # Get current GPU memory usage
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.1f} MB")
            print(f"  Memory cached: {torch.cuda.memory_reserved(i) / 1024**2:.1f} MB")
            
            # Test CUDA functionality
            print("\nRunning CUDA test...")
            try:
                x = torch.rand(1000, 1000).cuda()
                y = torch.rand(1000, 1000).cuda()
                z = torch.mm(x, y)
                del x, y, z
                torch.cuda.empty_cache()
                print("  ✓ CUDA test passed successfully")
            except Exception as e:
                print(f"  ✗ CUDA test failed: {str(e)}")
    else:
        print("\nNo CUDA devices found!")
        print("\nTroubleshooting steps:")
        print("1. Check if CUDA is installed:")
        try:
            nvcc = subprocess.check_output(['nvcc', '--version']).decode()
            print(f"\nNVCC found:\n{nvcc}")
        except:
            print("\nNVCC not found. Please install CUDA toolkit.")
        
        print("\n2. Current PyTorch installation:")
        print(f"Location: {torch.__file__}")
        
        print("\n3. Recommended fix:")
        print("Run these commands:")
        print("pip uninstall torch torchvision torchaudio")
        print("pip cache purge")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\nEnvironment Information:")
    print("-" * 50)
    print(f"Python version: {sys.version}")
    print(f"PyTorch location: {torch.__file__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

if __name__ == "__main__":
    check_cuda()