// File: ica_processor.cu
#include "signal_processing.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// Error checking macros
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
       std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " code:" << err << " msg:" << cudaGetErrorString(err) << std::endl; \
       exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t stat = call; \
    if(stat != CUBLAS_STATUS_SUCCESS) { \
       std::cerr << "CUBLAS error in " << __FILE__ << ":" << __LINE__ << std::endl; \
       exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t stat = call; \
    if(stat != CUSOLVER_STATUS_SUCCESS) { \
       std::cerr << "CUSOLVER error in " << __FILE__ << ":" << __LINE__ << std::endl; \
       exit(EXIT_FAILURE); \
    } \
}

// Kernel to compute mean per channel
__global__ void computeMeanKernel(const float* data, float* means, int n_samples, int n_channels) {
    extern __shared__ float sdata[];
    int j = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n_samples; i += blockDim.x) {
        sum += data[i + j * n_samples];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        means[j] = sdata[0] / n_samples;
    }
}

// Kernel to subtract channel means
__global__ void subtractMeanKernel(float* data, const float* means, int n_samples, int n_channels) {
    int j = blockIdx.x;
    int i = threadIdx.x;
    if (j < n_channels && i < n_samples) {
        data[i + j * n_samples] -= means[j];
    }
}

// Kernel to compute nonlinearity and its derivative for fastICA
__global__ void computeGKernel(const float* u, float* gu, float* gprime, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float tanh_u = tanhf(u[idx]);
        gu[idx] = tanh_u;
        gprime[idx] = 1.0f - tanh_u * tanh_u;
    }
}

// Kernel for reduction sum using shared memory
__global__ void reduceSumKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

void computeCovariance(cublasHandle_t handle, float* d_X, float* d_cov, int n_samples, int n_channels) {
    float alpha = 1.0f / n_samples;
    float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             n_channels, n_channels, n_samples,
                             &alpha,
                             d_X, n_samples,
                             d_X, n_samples,
                             &beta,
                             d_cov, n_channels));
}

void computeWhiteningMatrix(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float* d_cov, float* whitening, int n_channels) {
    int lwork = 0;
    int info_gpu = 0;
    float *d_work = nullptr;
    int *devInfo = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));

    float *d_W = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_W, sizeof(float) * n_channels * n_channels));
    CHECK_CUDA(cudaMemcpy(d_W, d_cov, sizeof(float) * n_channels * n_channels, cudaMemcpyDeviceToDevice));

    float* d_eigenvalues = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_eigenvalues, sizeof(float) * n_channels));

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    CHECK_CUSOLVER(cusolverDnSsyevd_bufferSize(cusolverH, jobz, uplo, n_channels, d_W, n_channels, d_eigenvalues, &lwork));
    CHECK_CUDA(cudaMalloc((void**)&d_work, sizeof(float) * lwork));
    CHECK_CUSOLVER(cusolverDnSsyevd(cusolverH, jobz, uplo, n_channels, d_W, n_channels, d_eigenvalues, d_work, lwork, devInfo));
    CHECK_CUDA(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_gpu != 0) {
        std::cerr << "Error: cusolverDnSsyevd failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<float> h_eigenvalues(n_channels);
    CHECK_CUDA(cudaMemcpy(h_eigenvalues.data(), d_eigenvalues, sizeof(float) * n_channels, cudaMemcpyDeviceToHost));
    cudaFree(d_eigenvalues);

    std::vector<float> h_V(n_channels * n_channels);
    CHECK_CUDA(cudaMemcpy(h_V.data(), d_W, sizeof(float) * n_channels * n_channels, cudaMemcpyDeviceToHost));

    std::vector<float> h_diag(n_channels * n_channels, 0.0f);
    for (int i = 0; i < n_channels; i++) {
        float inv_sqrt = 1.0f / sqrtf(h_eigenvalues[i] + 1e-5f);
        h_diag[i + i * n_channels] = inv_sqrt;
    }
    std::vector<float> h_temp(n_channels * n_channels, 0.0f);
    for (int i = 0; i < n_channels; i++) {
        for (int j = 0; j < n_channels; j++) {
            h_temp[i + j * n_channels] = h_V[i + j * n_channels] * h_diag[j + j * n_channels];
        }
    }
    std::vector<float> h_whitening(n_channels * n_channels, 0.0f);
    for (int i = 0; i < n_channels; i++) {
        for (int j = 0; j < n_channels; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n_channels; k++) {
                sum += h_temp[i + k * n_channels] * h_V[j + k * n_channels];
            }
            h_whitening[i + j * n_channels] = sum;
        }
    }
    CHECK_CUDA(cudaMemcpy(whitening, h_whitening.data(), sizeof(float) * n_channels * n_channels, cudaMemcpyHostToDevice));

    cudaFree(d_W);
    cudaFree(d_work);
    cudaFree(devInfo);
}

void fastICA(cublasHandle_t cublasH, float* d_Y, int n_samples, int n_channels, float* d_W, int maxIter = 5, float tol = 1e-4f) {
    float* d_w;
    float* d_w_old;
    float* d_u;
    float* d_gu;
    float* d_gprime;
    float* d_temp;
    CHECK_CUDA(cudaMalloc((void**)&d_w, sizeof(float) * n_channels));
    CHECK_CUDA(cudaMalloc((void**)&d_w_old, sizeof(float) * n_channels));
    CHECK_CUDA(cudaMalloc((void**)&d_u, sizeof(float) * n_samples));
    CHECK_CUDA(cudaMalloc((void**)&d_gu, sizeof(float) * n_samples));
    CHECK_CUDA(cudaMalloc((void**)&d_gprime, sizeof(float) * n_samples));
    CHECK_CUDA(cudaMalloc((void**)&d_temp, sizeof(float) * n_channels));

    for (int comp = 0; comp < n_channels; comp++) {
        std::vector<float> h_w(n_channels);
        for (int i = 0; i < n_channels; i++) {
            h_w[i] = (float)rand()/RAND_MAX;
        }
        float norm = 0.0f;
        for (int i = 0; i < n_channels; i++) {
            norm += h_w[i]*h_w[i];
        }
        norm = sqrtf(norm);
        for (int i = 0; i < n_channels; i++) {
            h_w[i] /= norm;
        }
        CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), sizeof(float)*n_channels, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_w_old, 0, sizeof(float)*n_channels));

        for (int iter = 0; iter < maxIter; iter++) {
            CHECK_CUDA(cudaMemcpy(d_w_old, d_w, sizeof(float)*n_channels, cudaMemcpyDeviceToDevice));
            float alpha = 1.0f, beta = 0.0f;
            CHECK_CUBLAS(cublasSgemv(cublasH, CUBLAS_OP_N, n_samples, n_channels,
                                     &alpha, d_Y, n_samples, d_w, 1,
                                     &beta, d_u, 1));
            int blockSize = 256;
            int gridSize = (n_samples + blockSize - 1) / blockSize;
            computeGKernel<<<gridSize, blockSize>>>(d_u, d_gu, d_gprime, n_samples);
            cudaDeviceSynchronize();
            float h_sum = 0.0f;
            float* d_sum;
            CHECK_CUDA(cudaMalloc((void**)&d_sum, sizeof(float)));
            CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(float)));
            int redBlockSize = 256;
            int redGridSize = 32;
            reduceSumKernel<<<redGridSize, redBlockSize, redBlockSize * sizeof(float)>>>(d_gprime, d_sum, n_samples);
            cudaDeviceSynchronize();
            CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
            float mean_gprime = h_sum / n_samples;
            cudaFree(d_sum);
            beta = 0.0f;
            CHECK_CUBLAS(cublasSgemv(cublasH, CUBLAS_OP_T, n_samples, n_channels,
                                     &alpha, d_Y, n_samples, d_gu, 1,
                                     &beta, d_temp, 1));
            float scale = 1.0f / n_samples;
            CHECK_CUBLAS(cublasSscal(cublasH, n_channels, &scale, d_temp, 1));
            float neg_mean = -mean_gprime;
            CHECK_CUBLAS(cublasSaxpy(cublasH, n_channels, &neg_mean, d_w, 1, d_temp, 1));
            for (int j = 0; j < comp; j++) {
                float dot = 0.0f;
                CHECK_CUBLAS(cublasSdot(cublasH, n_channels, d_temp, 1, d_W + j * n_channels, 1, &dot));
                float neg_dot = -dot;
                CHECK_CUBLAS(cublasSaxpy(cublasH, n_channels, &neg_dot, d_W + j * n_channels, 1, d_temp, 1));
            }
            float norm_w = 0.0f;
            CHECK_CUBLAS(cublasSnrm2(cublasH, n_channels, d_temp, 1, &norm_w));
            float inv_norm = 1.0f / norm_w;
            CHECK_CUBLAS(cublasSscal(cublasH, n_channels, &inv_norm, d_temp, 1));
            float dot_w = 0.0f;
            CHECK_CUBLAS(cublasSdot(cublasH, n_channels, d_temp, 1, d_w_old, 1, &dot_w));
            if (fabsf(fabsf(dot_w) - 1.0f) < tol) {
                CHECK_CUDA(cudaMemcpy(d_w, d_temp, sizeof(float)*n_channels, cudaMemcpyDeviceToDevice));
                break;
            }
            CHECK_CUDA(cudaMemcpy(d_w, d_temp, sizeof(float)*n_channels, cudaMemcpyDeviceToDevice));
        }
        CHECK_CUDA(cudaMemcpy(d_W + comp * n_channels, d_w, sizeof(float)*n_channels, cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_w);
    cudaFree(d_w_old);
    cudaFree(d_u);
    cudaFree(d_gu);
    cudaFree(d_gprime);
    cudaFree(d_temp);
}

std::vector<std::vector<float>> performICA(const std::vector<std::vector<float>>& signals, int n_samples) {
    //std::cout << "ICA start";

    int n_channels = signals.size();
    std::vector<float> X(n_samples * n_channels, 0.0f);
    for (int ch = 0; ch < n_channels; ch++) {
        for (int i = 0; i < n_samples; i++) {
            X[i + ch * n_samples] = signals[ch][i];
        }
    }
    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    float* d_X;
    CHECK_CUDA(cudaMalloc((void**)&d_X, sizeof(float) * n_samples * n_channels));
    CHECK_CUDA(cudaMemcpy(d_X, X.data(), sizeof(float) * n_samples * n_channels, cudaMemcpyHostToDevice));
    //std::cout << "ICA 1";

    float* d_means;
    CHECK_CUDA(cudaMalloc((void**)&d_means, sizeof(float) * n_channels));
    int blockSize = 256;
    int gridSize = n_channels;
    computeMeanKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_X, d_means, n_samples, n_channels);
    cudaDeviceSynchronize();
    subtractMeanKernel<<<gridSize, n_samples>>>(d_X, d_means, n_samples, n_channels);
    cudaDeviceSynchronize();
    cudaFree(d_means);
    //std::cout << "ICA 2";

    float* d_cov;
    CHECK_CUDA(cudaMalloc((void**)&d_cov, sizeof(float) * n_channels * n_channels));
    computeCovariance(cublasH, d_X, d_cov, n_samples, n_channels);
    //std::cout << "ICA 3";

    float* d_whitening;
    CHECK_CUDA(cudaMalloc((void**)&d_whitening, sizeof(float) * n_channels * n_channels));
    computeWhiteningMatrix(cusolverH, cublasH, d_cov, d_whitening, n_channels);
    cudaFree(d_cov);
    //std::cout << "ICA 4";

    float* d_Y;
    CHECK_CUDA(cudaMalloc((void**)&d_Y, sizeof(float) * n_samples * n_channels));
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                             n_samples, n_channels, n_channels,
                             &alpha,
                             d_X, n_samples,
                             d_whitening, n_channels,
                             &beta,
                             d_Y, n_samples));
    cudaFree(d_X);
    cudaFree(d_whitening);
    //std::cout << "ICA 5";



    // Seems to die >50 components
    float* d_W;
    CHECK_CUDA(cudaMalloc((void**)&d_W, sizeof(float) * n_channels * n_channels));
    fastICA(cublasH, d_Y, n_samples, n_channels, d_W);

    //std::cout << "ICA 7";

    float* d_S;
    CHECK_CUDA(cudaMalloc((void**)&d_S, sizeof(float) * n_samples * n_channels));
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                             n_samples, n_channels, n_channels,
                             &alpha,
                             d_Y, n_samples,
                             d_W, n_channels,
                             &beta,
                             d_S, n_samples));
    cudaFree(d_Y);
    cudaFree(d_W);
    //std::cout << "ICA 6";
    std::vector<float> S(n_samples * n_channels);
    CHECK_CUDA(cudaMemcpy(S.data(), d_S, sizeof(float) * n_samples * n_channels, cudaMemcpyDeviceToHost));
    cudaFree(d_S);

    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);

    std::vector<std::vector<float>> components(n_channels, std::vector<float>(n_samples));
    for (int ch = 0; ch < n_channels; ch++) {
        for (int i = 0; i < n_samples; i++) {
            components[ch][i] = S[i + ch * n_samples];
        }
    }
    //std::cout << "ICA complete";

    return components;
}
