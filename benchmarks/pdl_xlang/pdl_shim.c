// LD_PRELOAD shim: turns ordinary runtime launches (cudaLaunchKernel, i.e. every `<<<>>>` launch inside
// libtorch_cuda's precompiled ATen kernels) into cudaLaunchKernelExC launches carrying
// cudaLaunchAttributeProgrammaticStreamSerialization -- i.e. it makes unmodified ATen kernels PDL
// *secondaries* without recompiling PyTorch. It cannot add griddepcontrol.wait to their code.
// Off by default; toggled at runtime with pdl_shim_set(1/0) (via ctypes) so only chosen regions are affected.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <cuda_runtime_api.h>

static volatile int g_enabled = 0;
static volatile long g_converted = 0;
void pdl_shim_set(int on) { g_enabled = on; }
long pdl_shim_count(void) { return g_converted; }
int pdl_shim_loaded(void) { return 1; }

typedef cudaError_t (*launch_fn)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
typedef cudaError_t (*launch_ex_fn)(const cudaLaunchConfig_t *, const void *, void **);

cudaError_t cudaLaunchKernel(const void *func, dim3 grid, dim3 block, void **args, size_t smem,
                             cudaStream_t stream) {
    static launch_fn real = 0;
    static launch_ex_fn real_ex = 0;
    if (!real) {
        real = (launch_fn)dlsym(RTLD_NEXT, "cudaLaunchKernel");
        real_ex = (launch_ex_fn)dlsym(RTLD_NEXT, "cudaLaunchKernelExC");
    }
    if (!g_enabled || !real_ex) return real(func, grid, block, args, smem, stream);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {0};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = stream;
    cfg.attrs = attr; cfg.numAttrs = 1;
    __sync_fetch_and_add(&g_converted, 1);
    return real_ex(&cfg, func, args);
}
