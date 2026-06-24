#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")

void check_rms_norm_inputs(const torch::Tensor& input, const torch::Tensor& gamma) {
    CHECK_CUDA(input);
    CHECK_CUDA(gamma);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(gamma);
    CHECK_FLOAT(input);
    CHECK_FLOAT(gamma);
    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(input.size(-1) == gamma.size(0), "gamma size must match input last dimension");
}

// 原始版本：纯 shared memory 的树形规约。
__global__ void rms_norm_kernel(
    float* out,
    const float* input,
    const float* gamma,
    float epsilon,
    int hidden_dim) {
    int row_idx = blockIdx.x;
    const float* row_input = input + row_idx * hidden_dim;
    float* row_output = out + row_idx * hidden_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = row_input[i];
        sum_sq += val * val;
    }

    extern __shared__ float s_mem[];
    s_mem[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_mem[threadIdx.x] += s_mem[threadIdx.x + s];
        }
        __syncthreads();
    }

    __shared__ float rms_factor;
    if (threadIdx.x == 0) {
        rms_factor = rsqrtf(s_mem[0] / hidden_dim + epsilon);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        row_output[i] = row_input[i] * rms_factor * gamma[i];
    }
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 新版本：warp 内用 shuffle 规约，warp 间只用少量 shared memory。
__global__ void rms_norm_kernel_warp_shuffle(
    float* out,
    const float* input,
    const float* gamma,
    float epsilon,
    int hidden_dim) {
    int row_idx = blockIdx.x;
    const float* row_input = input + row_idx * hidden_dim;
    float* row_output = out + row_idx * hidden_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = row_input[i];
        sum_sq += val * val;
    }

    int lane = threadIdx.x % kWarpSize;
    int warp_id = threadIdx.x / kWarpSize;
    int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

    sum_sq = warp_reduce_sum(sum_sq);

    extern __shared__ float warp_sums[];
    if (lane == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }

    __shared__ float rms_factor;
    if (threadIdx.x == 0) {
        rms_factor = rsqrtf(block_sum / hidden_dim + epsilon);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        row_output[i] = row_input[i] * rms_factor * gamma[i];
    }
}

torch::Tensor prepare_output_and_launch_checks(
    torch::Tensor input,
    torch::Tensor gamma,
    int64_t* rows,
    int64_t* hidden_dim) {
    check_rms_norm_inputs(input, gamma);

    auto contiguous_input = input.contiguous();
    auto contiguous_gamma = gamma.contiguous();
    *hidden_dim = contiguous_input.size(-1);
    *rows = contiguous_input.numel() / *hidden_dim;
    TORCH_CHECK(*hidden_dim <= std::numeric_limits<int>::max(), "hidden_dim is too large");
    TORCH_CHECK(*rows <= std::numeric_limits<unsigned int>::max(), "number of rows is too large");
    return torch::empty_like(contiguous_input);
}

}  // namespace

// 给 PyTorch 调用的 C++ 包装函数：shared memory 版本。
torch::Tensor rms_norm_shared_cuda(torch::Tensor input, torch::Tensor gamma, float epsilon) {
    int64_t rows = 0;
    int64_t hidden_dim = 0;
    auto contiguous_input = input.contiguous();
    auto contiguous_gamma = gamma.contiguous();
    auto output = prepare_output_and_launch_checks(contiguous_input, contiguous_gamma, &rows, &hidden_dim);

    int shared_mem_bytes = kThreads * sizeof(float);
    rms_norm_kernel<<<static_cast<unsigned int>(rows), kThreads, shared_mem_bytes>>>(
        output.data_ptr<float>(),
        contiguous_input.data_ptr<float>(),
        contiguous_gamma.data_ptr<float>(),
        epsilon,
        static_cast<int>(hidden_dim));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// 给 PyTorch 调用的 C++ 包装函数：warp shuffle 版本。
torch::Tensor rms_norm_warp_shuffle_cuda(torch::Tensor input, torch::Tensor gamma, float epsilon) {
    int64_t rows = 0;
    int64_t hidden_dim = 0;
    auto contiguous_input = input.contiguous();
    auto contiguous_gamma = gamma.contiguous();
    auto output = prepare_output_and_launch_checks(contiguous_input, contiguous_gamma, &rows, &hidden_dim);

    int shared_mem_bytes = ((kThreads + kWarpSize - 1) / kWarpSize) * sizeof(float);
    rms_norm_kernel_warp_shuffle<<<static_cast<unsigned int>(rows), kThreads, shared_mem_bytes>>>(
        output.data_ptr<float>(),
        contiguous_input.data_ptr<float>(),
        contiguous_gamma.data_ptr<float>(),
        epsilon,
        static_cast<int>(hidden_dim));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_shared", &rms_norm_shared_cuda, "RMSNorm CUDA shared-memory implementation");
    m.def("rms_norm_warp_shuffle", &rms_norm_warp_shuffle_cuda, "RMSNorm CUDA warp-shuffle implementation");
}
