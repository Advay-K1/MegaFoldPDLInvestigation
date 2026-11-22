// CUDA C++ kernels that take part in PDL exactly like Triton's gdc_wait / gdc_launch_dependents:
// the same two PTX instructions, written as inline asm, plus a cudaLaunchKernelEx launch with the
// programmatic-stream-serialization attribute. Used for the Triton <-> CUDA C++ boundaries.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__device__ __forceinline__ void gdc_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void gdc_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" :::); }
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

// y = x*a + b after `work` dependent FMAs per element (grid-stride)
__global__ void axpb_kernel(const float *x, float *y, long n, float a, float b, int wait, int trig, int work) {
    if (wait) gdc_wait();
    if (trig) gdc_launch_dependents();
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        float v = x[i];
        for (int k = 0; k < work; ++k) v = v * 0.999f + 0.001f;
        y[i] = v * a + b;
    }
}

// probe: records %globaltimer at entry and right after griddepcontrol.wait. A large gap means this grid was
// running while its predecessor was still running, i.e. the predecessor triggered launch_dependents early.
__global__ void probe_kernel(unsigned long long *out) {
    unsigned long long t0 = gtimer();
    gdc_wait();
    unsigned long long t1 = gtimer();
    if (threadIdx.x == 0) { out[0] = t0; out[1] = t1; }
}

static cudaLaunchConfig_t cfg_for(dim3 g, dim3 b, bool pdl, cudaLaunchAttribute *attr) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = g; cfg.blockDim = b; cfg.stream = at::cuda::getCurrentCUDAStream();
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = pdl ? 1 : 0;
    return cfg;
}

void axpb(torch::Tensor x, torch::Tensor y, double a, double b, bool pdl_attr, bool wait, bool trig,
          int64_t work, int64_t grid, int64_t block) {
    cudaLaunchAttribute attr[1];
    auto cfg = cfg_for(dim3(grid), dim3(block), pdl_attr, attr);
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, axpb_kernel, (const float *)x.data_ptr<float>(), y.data_ptr<float>(),
                                      (long)x.numel(), (float)a, (float)b, (int)wait, (int)trig, (int)work));
}

void probe(torch::Tensor out) {
    cudaLaunchAttribute attr[1];
    auto cfg = cfg_for(dim3(1), dim3(32), true, attr);
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, probe_kernel, (unsigned long long *)out.data_ptr<int64_t>()));
}

// Is ProgrammaticStreamSerialization accepted as a *stream* attribute (would make every launch on the stream PDL)?
int64_t try_stream_attr() {
    cudaLaunchAttributeValue v = {};
    v.programmaticStreamSerializationAllowed = 1;
    cudaError_t e = cudaStreamSetAttribute(at::cuda::getCurrentCUDAStream(),
                                           (cudaStreamAttrID)cudaLaunchAttributeProgrammaticStreamSerialization, &v);
    cudaGetLastError();
    return (int64_t)e;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("axpb", &axpb); m.def("probe", &probe); m.def("try_stream_attr", &try_stream_attr);
}
