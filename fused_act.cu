#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/api/include/torch/python.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// -----------------------------
// GELU × Sigmoid device function
// -----------------------------
__device__ __forceinline__ float gelu_sigmoid(float x)
{
    float gelu = 0.5f * x * (1.0f + erff(x * 0.70710678118f));

    float exp_neg_abs = expf(-fabsf(x));
    float sigmoid = (x >= 0.0f) ?
        1.0f / (1.0f + exp_neg_abs) :
        exp_neg_abs / (1.0f + exp_neg_abs);

    return gelu * sigmoid;
}

// -----------------------------
// CUDA kernel
// -----------------------------
__global__ void fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        output[idx] = gelu_sigmoid(input[idx]);
}

// -----------------------------
// C++ bridge
// -----------------------------
at::Tensor forward(at::Tensor input)
{
    if (!input.is_cuda())
        throw std::runtime_error("Input must be CUDA");

    if (input.scalar_type() != at::kFloat)
        throw std::runtime_error("Input must be float32");

    auto output = at::empty_like(input);

    int64_t size = input.numel();

    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    cudaDeviceSynchronize();

    return output;
}

// -----------------------------
// Python binding
// -----------------------------
PYBIND11_MODULE(fused_op, m)
{
    m.def("forward", &forward, "Fused GELU × Sigmoid (CUDA)");
}
