
#include <nvrtc.h>
#include <cuda.h>
#include <vector>
#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;


// Save CUDA code to file
void saveKernelCUDA(const char* filename, const char* cudaCode) {
  FILE* f = fopen(filename, "w");
  fprintf(f, "%s", cudaCode);
  fclose(f);
}

// Compile CUDA to PTX
const char* compileCUDAtoPTX(const char* cudaCode, const char* filename) {
  nvrtcProgram prog;
  nvrtcCreateProgram(&prog, cudaCode, filename, 0, NULL, NULL); 
  nvrtcCompileProgram(prog, 0, NULL);

  size_t ptxSize;
  nvrtcGetPTXSize(prog, &ptxSize);
  char* ptx = new char[ptxSize];
  nvrtcGetPTX(prog, ptx);
  // std::cout <<"get ptx: " << ptx << std::endl;

  return ptx;
}

// Compile PTX to CUBIN
std::vector<char> compilePTXtoCUBIN(const string &ptxcode) {

  char jitErrorBuffer[4096] = {0};
  (cuInit(0));
  CUdevice device; 
  (cuDeviceGet(&device, 0));
  CUcontext context;
  (cuCtxCreate(&context, 0, device));
  CUlinkState linkState;

  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void* jitOptionsVals[] = {jitErrorBuffer, reinterpret_cast<void*>(sizeof(jitErrorBuffer))};

  (cuLinkCreate(2, jitOptions, jitOptionsVals, &linkState));

  cuLinkAddData(linkState, CUjitInputType::CU_JIT_INPUT_PTX, const_cast<void *>(static_cast<const void *>(ptxcode.c_str())), 
                ptxcode.length(), "_Z8myKernelPfS_S_", 0, nullptr, nullptr);
  
  void* cubin;
  size_t cubinSize;
  cuLinkComplete(linkState, &cubin, &cubinSize);

  std::vector<char> result(static_cast<char*>(cubin), 
                           static_cast<char*>(cubin) + cubinSize);
                           
  cuLinkDestroy(linkState);
  cuCtxDestroy(context);
  
  return result;
}

void initialData(float *ip, int size) {
    // generate different seed for random number
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(i);
    }

    return;
}


// Load CUBIN and launch kernel

void launchCUBIN(std::vector<char>& cubin) {

  // 设置device 
  cudaSetDevice(0);
  
  // 从vector获取CUBIN数据
  void* cubinData = (void*)cubin.data();
  size_t cubinSize = cubin.size();

  // 创建CUDA context
  cudaFree(0);

  // 加载CUBIN模块
  CUmodule module;
  cuModuleLoadData(&module, cubinData);

  // 获取kernel函数指针
  CUfunction kernel;
  // this is mangling name, not kernel name in cuda kernel
  // cuModuleGetFunction(&kernel, module, "myKernel");
  cuModuleGetFunction(&kernel, module, "_Z8myKernelPfS_S_"); 
  // 设置启动参数
  // 定义grid大小,block大小等

  // 设置启动参数
  int blockSize = 32;
  int gridSize = (10 + blockSize - 1) / blockSize;
  size_t nBytes = 10 * sizeof(float);

  float* h_A = (float *)malloc(nBytes);
  float* h_B = (float *)malloc(nBytes);
  float* h_C = (float *)malloc(nBytes);

  initialData(h_A, 10); 
  initialData(h_B, 10); 

  float *d_A, *d_B, *d_C;
  cudaMalloc((float**)&d_A, nBytes);
  cudaMalloc((float**)&d_B, nBytes);
  cudaMalloc((float**)&d_C, nBytes);

  // transfer data from host to device
  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

  void *args[] = {&d_A, &d_B, &d_C};

  // 启动kernel
  cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 
                 0, nullptr, (void **)args, nullptr);
  cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++) {
     cout<< h_C[i] << endl;
  }

  // free device global memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  // 释放模块
  cuModuleUnload(module);

}

int main() {
  // CUDA kernel code
  const char* cudaCode = R"(
    __global__ void myKernel(float* a, float* b, float* c) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < 10) {
          c[idx] = a[idx] + b[idx]; 
        }
    }
  )";
  
  saveKernelCUDA("kernel.cu", cudaCode);

  const char* ptx = compileCUDAtoPTX(cudaCode, "kernel.cu");
  const string ptxcode(ptx);

  std::vector<char> cubin = compilePTXtoCUBIN(ptxcode);

  launchCUBIN(cubin);

  return 0;
}
