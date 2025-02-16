#include "monte_carlo_simulation.h"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

// CUDA kernel that generates one Monte Carlo simulation path per thread.
__global__ void monteCarloKernel(double startPrice, int nSteps, double dt, double mu, double sigma,
                                 double jumpProb, double jumpMagnitude, double* d_paths, int nPaths) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nPaths) return;
    int offset = p * nSteps;
    d_paths[offset] = startPrice;

    // Initialize cuRAND state for each thread.
    curandState state;
    curand_init(1234, p, 0, &state);

    double price = startPrice;
    for (int t = 1; t < nSteps; t++) {
        double dW = curand_normal_double(&state) * sqrt(dt);
        double drift = (mu - 0.5 * sigma * sigma) * dt;
        double diffusion = sigma * dW;
        double jump = 0.0;
        double u = curand_uniform_double(&state);
        if (u < jumpProb) {
            jump = jumpMagnitude * curand_normal_double(&state);
        }
        price = price * exp(drift + diffusion + jump);
        // Clamp the price to be within 20% of the initial price.
        if (price > startPrice * 1.2) {
            price = startPrice * 1.2;
        } else if (price < startPrice * 0.8) {
            price = startPrice * 0.8;
        }
        d_paths[offset + t] = price;
    }
}

std::vector<std::vector<double>> simulateMonteCarloPaths(double startPrice, int nPaths, int nSteps,
                                                         double dt, double mu, double sigma,
                                                         double jumpProb, double jumpMagnitude) {
    size_t totalSize = nPaths * nSteps * sizeof(double);
    double* d_paths = nullptr;
    cudaMalloc(&d_paths, totalSize);

    // Determine grid dimensions.
    int threadsPerBlock = 256;
    int blocks = (nPaths + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel.
    monteCarloKernel<<<blocks, threadsPerBlock>>>(startPrice, nSteps, dt, mu, sigma, jumpProb, jumpMagnitude, d_paths, nPaths);
    cudaDeviceSynchronize();

    // Copy results from device to host.
    std::vector<double> h_paths(nPaths * nSteps);
    cudaMemcpy(h_paths.data(), d_paths, totalSize, cudaMemcpyDeviceToHost);
    cudaFree(d_paths);

    // Organize the flat host array into a vector of paths.
    std::vector<std::vector<double>> paths(nPaths, std::vector<double>(nSteps));
    for (int p = 0; p < nPaths; p++) {
        for (int t = 0; t < nSteps; t++) {
            paths[p][t] = h_paths[p * nSteps + t];
        }
    }
    return paths;
}
