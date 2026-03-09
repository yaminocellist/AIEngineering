import os
import torch
import time

# --- DLL fix for Windows (PyTorch CUDA libs) ---
TORCH_LIB_PATH = r"C:\Users\ROG\miniforge3\Lib\site-packages\torch\lib"
if os.path.exists(TORCH_LIB_PATH):
    os.add_dll_directory(TORCH_LIB_PATH)

# Import compiled CUDA extension
import fused_op


def benchmark_fused_op():
    size = 100_000_000  # 100M elements (~400MB)
    print(f"Allocating {size/1e6:.0f}M elements (~400MB)...")

    # --- Force contiguous float32 CUDA tensor ---
    input_tensor = torch.randn(size, device='cuda', dtype=torch.float32).contiguous()
    print(f"Contiguous? {input_tensor.is_contiguous()}")

    # -----------------------------
    # Warmup
    # -----------------------------
    print("Warming up GPU...")
    for _ in range(10):
        _ = torch.nn.functional.gelu(input_tensor) * torch.sigmoid(input_tensor)
        _ = fused_op.forward(input_tensor)

    torch.cuda.synchronize()

    # -----------------------------
    # Native PyTorch Benchmark
    # -----------------------------
    print("Running Native PyTorch Benchmark...")
    iters = 50

    torch.cuda.synchronize()
    start_native = time.time()

    for _ in range(iters):
        _ = torch.nn.functional.gelu(input_tensor) * torch.sigmoid(input_tensor)

    torch.cuda.synchronize()
    native_time = (time.time() - start_native) / iters

    # -----------------------------
    # Fused CUDA Benchmark
    # -----------------------------
    print("Running Fused CUDA Kernel Benchmark...")

    torch.cuda.synchronize()
    start_fused = time.time()

    for _ in range(iters):
        _ = fused_op.forward(input_tensor)

    torch.cuda.synchronize()
    fused_time = (time.time() - start_fused) / iters

    # -----------------------------
    # Validation
    # -----------------------------
    with torch.no_grad():

        expected = torch.nn.functional.gelu(input_tensor) * torch.sigmoid(input_tensor)
        actual = fused_op.forward(input_tensor)

        diff = torch.abs(expected - actual)
        max_val, max_idx = torch.max(diff, dim=0)

        print("\n--- Numerical Comparison (first 10 elements) ---")
        print(f"{'Index':>6} | {'Expected':>12} | {'Actual':>12} | {'Diff':>12}")
        print("-" * 50)

        for i in range(10):
            print(
                f"{i:>6} | "
                f"{expected[i]:12.8f} | "
                f"{actual[i]:12.8f} | "
                f"{diff[i]:12.8e}"
            )

        print("-" * 50)
        print(f"MAX ERROR AT INDEX {max_idx.item()}: {max_val.item():.2e}")

        is_correct = torch.allclose(expected, actual, atol=1e-6)

    # -----------------------------
    # Results
    # -----------------------------
    print("\n" + "=" * 40)
    print(f"DEVICE: {torch.cuda.get_device_name(0)}")
    print(f"VALIDATION: {'PASSED' if is_correct else 'FAILED'}")
    print(f"Native Time: {native_time * 1000:.4f} ms")
    print(f"Fused Time:  {fused_time * 1000:.4f} ms")
    print(f"SPEEDUP:     {native_time / fused_time:.2f}x")
    print("=" * 40)


if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("CUDA not found. Check GPU drivers.")
    else:
        benchmark_fused_op()
