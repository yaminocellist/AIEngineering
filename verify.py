import os
import sys
import torch

# 1. Explicitly add the torch lib folder to the DLL search path
torch_lib_path = r"C:\Users\ROG\miniforge3\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib_path):
    os.add_dll_directory(torch_lib_path)
    print(f"Added DLL directory: {torch_lib_path}")
else:
    print(f"ERROR: Could not find torch lib at {torch_lib_path}")

# 2. Try the import
try:
    import fused_op
    print("========================================")
    print("SUCCESS: Handshake Successful!")
    print("========================================")
except ImportError as e:
    print("========================================")
    print(f"IMPORT ERROR: {e}")
    # This will help us see if other DLLs are missing
    if hasattr(os, 'add_dll_directory'):
        print("\nNote: Python 3.8+ requires os.add_dll_directory for C extensions.")
    print("========================================")