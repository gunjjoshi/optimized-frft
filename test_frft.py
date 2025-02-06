import torch
import torch.nn as nn
import time
import gc  # Garbage collector for memory management

# Import the old and optimized FRFT implementations
from old_frft import FRFT as OldFRFT  # Old implementation
from optimized_frft import FRFT as OptFRFT  # Optimized implementation

# Define a test function
def test_model(frft_class, input_shape=(1, 3, 64, 64), num_runs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate random input tensor
    x = torch.randn(input_shape).to(device)

    # Instantiate model
    model = frft_class(in_channels=input_shape[1]).to(device)
    model.eval()  # Set to evaluation mode (no dropout, batch norm is fixed)

    # Warm-up run (for stable benchmarking)
    with torch.no_grad():
        _ = model(x)

    # Measure execution time
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            _ = model(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start_time)

    avg_time = sum(times) / num_runs
    print(f"✅ {frft_class.__name__}: Avg Execution Time: {avg_time:.6f} sec")

    # Check GPU Memory Usage
    torch.cuda.empty_cache()
    gc.collect()

    return avg_time

# Run tests
if __name__ == "__main__":
    print("Running tests for FRFT implementations...")

    time_old = test_model(OldFRFT)
    time_opt = test_model(OptFRFT)

    print("\nPerformance Comparison:")
    print(f"Old FRFT Execution Time: {time_old:.6f} sec")
    print(f"Optimized FRFT Execution Time: {time_opt:.6f} sec")
    print(f"Speed-up Factor: {time_old / time_opt:.2f}x")

    # Validate output consistency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test = torch.randn((1, 3, 64, 64)).to(device)

    old_model = OldFRFT(in_channels=3).to(device).eval()
    opt_model = OptFRFT(in_channels=3).to(device).eval()

    with torch.no_grad():
        old_output = old_model(x_test)
        opt_output = opt_model(x_test)

    diff = torch.abs(old_output - opt_output).mean().item()
    print(f"\n✅ Output Difference (Mean Absolute Error): {diff:.8f}")
