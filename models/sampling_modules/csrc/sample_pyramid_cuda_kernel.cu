/*
 * Fused CUDA kernel for HashedProbabilityPyramid sampling.
 *
 * Each thread processes ONE sample through ALL pyramid layers, eliminating
 * ~50 kernel launches and intermediate tensor allocations per sample() call.
 *
 * The sampler finds indices and a companion autograd wrapper computes
 * log-prob gradients with fused CUDA kernels.
 */
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_LAYERS 8

/* ── helpers ────────────────────────────────────────────────────────── */

// Tiled mapping: (x + y*accum + z*accum^2) % ncomp
__device__ __forceinline__ int tiled_map(int ax, int ay, int az,
                                         int ncomp, int accum) {
    long long idx = (long long)ax + (long long)ay * accum
                  + (long long)az * accum * accum;
    return (int)(((idx % ncomp) + ncomp) % ncomp);
}

// Hashed mapping: hash(x, y, z) % ncomp
__device__ __forceinline__ int hashed_map(int ax, int ay, int az, int ncomp) {
    const int pi_2 = 19349663;
    const int pi_3 = 83492791;
    int h = ax ^ (ay * pi_2) ^ (az * pi_3);
    return ((h % ncomp) + ncomp) % ncomp;
}

// Linear search for inverse-CDF sampling in a small sorted CDF array.
// Returns bin_low such that cdf[bin_low] <= u < cdf[bin_low+1].
__device__ __forceinline__ int inverse_cdf_search(
    const float* __restrict__ cdf_row,   // [res+1] cumulative sums
    float u, int res)
{
    int bin = 0;
    for (int i = 1; i < res; i++) {
        if (u >= cdf_row[i]) bin = i;
    }
    return min(bin, res - 1);
}

/* ── main kernel ────────────────────────────────────────────────────── */

__global__ void sample_pyramid_fused(
    // --- random inputs ----------------------
    const float* __restrict__ ux_in,      // [N]
    const float* __restrict__ uy_in,      // [N]
    const float* __restrict__ uz_in,      // [N]
    // --- CDF tables (device-side ptr array) --
    const float* const* __restrict__ cx_ptrs,   // [nlayers] -> each [ncomp*(res+1)]
    const float* const* __restrict__ cy_ptrs,   // [nlayers] -> each [ncomp*res*(res+1)]
    const float* const* __restrict__ cz_ptrs,   // [nlayers] -> each [ncomp*res^2*(res+1)]
    // --- per-layer config (device) ----------
    const int* __restrict__ d_res,         // [nlayers]
    const int* __restrict__ d_ncomp,       // [nlayers]
    const int* __restrict__ d_accum,       // [nlayers]
    const int* __restrict__ d_map_type,    // [nlayers]  0=zero, 1=tiled
    const float* __restrict__ d_factor,    // [nlayers]  fres / prod(res[:l+1])
    int nlayers,
    int fres,
    bool snap_to_center,
    // --- outputs ----------------------------
    float* __restrict__ xyz_out,          // [N, 3]
    int64_t* __restrict__ bin_1d_out,     // [N]
    int64_t* __restrict__ pxyz_idx_out,   // [nlayers * N]
    int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float u_x = ux_in[tid];
    float u_y = uy_in[tid];
    float u_z = uz_in[tid];

    // Running accumulators
    int acc_x = 0, acc_y = 0, acc_z = 0;
    // Previous layer's bin outputs (for accum update)
    int prev_bx = 0, prev_by = 0, prev_bz = 0;
    float samp_x = 0.0f, samp_y = 0.0f, samp_z = 0.0f;

    for (int ll = 0; ll < nlayers; ll++) {
        const int res   = d_res[ll];
        const int ncomp = d_ncomp[ll];
        const int accum = d_accum[ll];
        const int res_p1 = res + 1;
        const float factor = d_factor[ll];

        const float* cx = cx_ptrs[ll];
        const float* cy = cy_ptrs[ll];
        const float* cz = cz_ptrs[ll];

        // ---- update accumulated bins: accum = accum * curr_res + prev_bin ----
        // (matches Python: accum_bins = accum_bins * self.reslist[curr_lvl] + bin_low)
        acc_x = acc_x * res + prev_bx;
        acc_y = acc_y * res + prev_by;
        acc_z = acc_z * res + prev_bz;

        // ---- component mapping ----
        int comp;
        if (d_map_type[ll] == 0)
            comp = 0;
        else if (d_map_type[ll] == 1)
            comp = tiled_map(acc_x, acc_y, acc_z, ncomp, accum);
        else
            comp = hashed_map(acc_x, acc_y, acc_z, ncomp);

        // ---- sample x from marginal CDF ----
        const float* cx_row = cx + comp * res_p1;
        int bx = inverse_cdf_search(cx_row, u_x, res);
        float cx_lo  = cx_row[bx];
        float cx_hi  = cx_row[bx + 1];
        float frac_x = (u_x - cx_lo) / fmaxf(cx_hi - cx_lo, 1e-10f) + (float)bx;

        // ---- sample y from conditional CDF(y|x) ----
        int cy_row_idx = comp * res + bx;
        const float* cy_row = cy + cy_row_idx * res_p1;
        int by = inverse_cdf_search(cy_row, u_y, res);
        float cy_lo  = cy_row[by];
        float cy_hi  = cy_row[by + 1];
        float frac_y = (u_y - cy_lo) / fmaxf(cy_hi - cy_lo, 1e-10f) + (float)by;

        // ---- sample z from conditional CDF(z|x,y) ----
        int cz_row_idx = comp * res * res + bx * res + by;
        const float* cz_row = cz + cz_row_idx * res_p1;
        int bz = inverse_cdf_search(cz_row, u_z, res);
        float cz_lo  = cz_row[bz];
        float cz_hi  = cz_row[bz + 1];
        float frac_z = (u_z - cz_lo) / fmaxf(cz_hi - cz_lo, 1e-10f) + (float)bz;

        // ---- pxyz flat index for log-prob (computed in PyTorch) ----
        int64_t flat_idx = (int64_t)comp * res * res * res
                         + (int64_t)bx * res * res
                         + (int64_t)by * res
                         + (int64_t)bz;
        pxyz_idx_out[ll * N + tid] = flat_idx;

        // ---- accumulate xyz_samples ----
        if (ll < nlayers - 1) {
            samp_x += (float)bx * factor;
            samp_y += (float)by * factor;
            samp_z += (float)bz * factor;
        } else {
            samp_x += frac_x * factor;
            samp_y += frac_y * factor;
            samp_z += frac_z * factor;
        }

        // ---- store current bins for next layer's accum update ----
        prev_bx = bx;
        prev_by = by;
        prev_bz = bz;

        // ---- next-layer u values: fractional position within the bin ----
        u_x = fminf(fmaxf(frac_x - (float)bx, 1e-8f), 1.0f - 1e-8f);
        u_y = fminf(fmaxf(frac_y - (float)by, 1e-8f), 1.0f - 1e-8f);
        u_z = fminf(fmaxf(frac_z - (float)bz, 1e-8f), 1.0f - 1e-8f);
    }

    // ---- snap to voxel center ----
    if (snap_to_center) {
        samp_x = floorf(samp_x) + 0.5f;
        samp_y = floorf(samp_y) + 0.5f;
        samp_z = floorf(samp_z) + 0.5f;
    }

    // ---- final bin_1d ----
    int bx_final = (int)floorf(samp_x);
    int by_final = (int)floorf(samp_y);
    int bz_final = (int)floorf(samp_z);
    bin_1d_out[tid] = (int64_t)bx_final * fres * fres
                    + (int64_t)by_final * fres
                    + (int64_t)bz_final;

    // ---- normalize to [0, 1] ----
    float inv_fres = 1.0f / (float)fres;
    samp_x = fminf(fmaxf(samp_x * inv_fres, 1e-8f), 1.0f - 1e-8f);
    samp_y = fminf(fmaxf(samp_y * inv_fres, 1e-8f), 1.0f - 1e-8f);
    samp_z = fminf(fmaxf(samp_z * inv_fres, 1e-8f), 1.0f - 1e-8f);

    xyz_out[tid * 3 + 0] = samp_x;
    xyz_out[tid * 3 + 1] = samp_y;
    xyz_out[tid * 3 + 2] = samp_z;
}


/* ── C++ wrapper ────────────────────────────────────────────────────── */

std::vector<torch::Tensor> sample_pyramid_cuda(
    torch::Tensor ux,                            // [N]
    torch::Tensor uy,                            // [N]
    torch::Tensor uz,                            // [N]
    std::vector<torch::Tensor> cx_list,          // [nlayers]
    std::vector<torch::Tensor> cy_list,          // [nlayers]
    std::vector<torch::Tensor> cz_list,          // [nlayers]
    std::vector<int64_t> res_list,
    std::vector<int64_t> ncomp_list,
    std::vector<int64_t> accum_list,
    std::vector<int64_t> mapping_types,          // 0=zero, 1=tiled
    std::vector<double> factors,
    int64_t fres,
    bool snap_to_center)
{
    const int N = ux.size(0);
    const int nlayers = (int)res_list.size();
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(ux.device());
    auto opts_l = torch::TensorOptions().dtype(torch::kInt64).device(ux.device());

    // Allocate outputs
    auto xyz_out     = torch::empty({N, 3}, opts_f);
    auto bin_1d_out  = torch::empty({N},    opts_l);
    auto pxyz_idx    = torch::empty({nlayers, N}, opts_l);

    // Build device arrays for CDF pointers and config
    std::vector<const float*> h_cx(nlayers), h_cy(nlayers), h_cz(nlayers);
    std::vector<int> h_res(nlayers), h_ncomp(nlayers), h_accum(nlayers), h_map(nlayers);
    std::vector<float> h_factor(nlayers);

    for (int i = 0; i < nlayers; i++) {
        h_cx[i]     = cx_list[i].data_ptr<float>();
        h_cy[i]     = cy_list[i].data_ptr<float>();
        h_cz[i]     = cz_list[i].data_ptr<float>();
        h_res[i]    = (int)res_list[i];
        h_ncomp[i]  = (int)ncomp_list[i];
        h_accum[i]  = (int)accum_list[i];
        h_map[i]    = (int)mapping_types[i];
        h_factor[i] = (float)factors[i];
    }

    // Copy small config arrays to device
    const float **d_cx, **d_cy, **d_cz;
    int *d_res, *d_ncomp, *d_accum, *d_map;
    float *d_factor;

    cudaMalloc(&d_cx,     nlayers * sizeof(float*));
    cudaMalloc(&d_cy,     nlayers * sizeof(float*));
    cudaMalloc(&d_cz,     nlayers * sizeof(float*));
    cudaMalloc(&d_res,    nlayers * sizeof(int));
    cudaMalloc(&d_ncomp,  nlayers * sizeof(int));
    cudaMalloc(&d_accum,  nlayers * sizeof(int));
    cudaMalloc(&d_map,    nlayers * sizeof(int));
    cudaMalloc(&d_factor, nlayers * sizeof(float));

    cudaMemcpy(d_cx,     h_cx.data(),     nlayers * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cy,     h_cy.data(),     nlayers * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cz,     h_cz.data(),     nlayers * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res,    h_res.data(),    nlayers * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_ncomp,  h_ncomp.data(),  nlayers * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_accum,  h_accum.data(),  nlayers * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_map,    h_map.data(),    nlayers * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_factor, h_factor.data(), nlayers * sizeof(float),  cudaMemcpyHostToDevice);

    // Launch kernel
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    sample_pyramid_fused<<<blocks, threads>>>(
        ux.data_ptr<float>(),
        uy.data_ptr<float>(),
        uz.data_ptr<float>(),
        d_cx, d_cy, d_cz,
        d_res, d_ncomp, d_accum, d_map, d_factor,
        nlayers, (int)fres, snap_to_center,
        xyz_out.data_ptr<float>(),
        bin_1d_out.data_ptr<int64_t>(),
        pxyz_idx.data_ptr<int64_t>(),
        N);

    // Free temp device arrays
    cudaFree(d_cx);    cudaFree(d_cy);    cudaFree(d_cz);
    cudaFree(d_res);   cudaFree(d_ncomp); cudaFree(d_accum);
    cudaFree(d_map);   cudaFree(d_factor);

    return {xyz_out, bin_1d_out, pxyz_idx};
}

__global__ void logprob_forward_kernel(
    const float* const* __restrict__ pxyz_ptrs,
    const int64_t* __restrict__ idx,
    int nlayers,
    int N,
    float* __restrict__ log_prob)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float accum = 0.0f;
    for (int ll = 0; ll < nlayers; ll++) {
        const int64_t flat_idx = idx[(int64_t)ll * N + tid];
        const float p = fmaxf(pxyz_ptrs[ll][flat_idx], 1e-20f);
        accum += logf(p);
    }
    log_prob[tid] = accum;
}

__global__ void logprob_backward_kernel(
    const float* const* __restrict__ pxyz_ptrs,
    const int64_t* __restrict__ idx,
    const float* __restrict__ grad_log_prob,
    int nlayers,
    int N,
    float** __restrict__ grad_ptrs)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    const float upstream = grad_log_prob[tid];
    for (int ll = 0; ll < nlayers; ll++) {
        const int64_t flat_idx = idx[(int64_t)ll * N + tid];
        const float p = fmaxf(pxyz_ptrs[ll][flat_idx], 1e-20f);
        atomicAdd(grad_ptrs[ll] + flat_idx, upstream / p);
    }
}

torch::Tensor logprob_forward_cuda(
    std::vector<torch::Tensor> pxyz_layers,
    torch::Tensor pxyz_idx)
{
    const int nlayers = (int)pxyz_layers.size();
    TORCH_CHECK(nlayers == pxyz_idx.size(0), "pxyz layer count must match index rows");
    TORCH_CHECK(pxyz_idx.dtype() == torch::kInt64, "pxyz_idx must be int64");
    TORCH_CHECK(pxyz_idx.is_cuda(), "pxyz_idx must be a CUDA tensor");

    const int N = pxyz_idx.size(1);
    auto idx_contig = pxyz_idx.contiguous();
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(pxyz_idx.device());
    auto log_prob = torch::empty({N, 1}, opts_f);

    std::vector<const float*> h_pxyz(nlayers);
    for (int i = 0; i < nlayers; i++) {
        TORCH_CHECK(pxyz_layers[i].is_cuda(), "pxyz layers must be CUDA tensors");
        TORCH_CHECK(pxyz_layers[i].dtype() == torch::kFloat32, "pxyz layers must be float32");
        TORCH_CHECK(pxyz_layers[i].is_contiguous(), "pxyz layers must be contiguous");
        h_pxyz[i] = pxyz_layers[i].data_ptr<float>();
    }

    const float** d_pxyz;
    cudaMalloc(&d_pxyz, nlayers * sizeof(float*));
    cudaMemcpy(d_pxyz, h_pxyz.data(), nlayers * sizeof(float*), cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    logprob_forward_kernel<<<blocks, threads>>>(
        d_pxyz,
        idx_contig.data_ptr<int64_t>(),
        nlayers,
        N,
        log_prob.data_ptr<float>());

    cudaFree(d_pxyz);
    return log_prob;
}

std::vector<torch::Tensor> logprob_backward_cuda(
    std::vector<torch::Tensor> pxyz_layers,
    torch::Tensor pxyz_idx,
    torch::Tensor grad_log_prob)
{
    const int nlayers = (int)pxyz_layers.size();
    TORCH_CHECK(nlayers == pxyz_idx.size(0), "pxyz layer count must match index rows");
    TORCH_CHECK(pxyz_idx.dtype() == torch::kInt64, "pxyz_idx must be int64");
    TORCH_CHECK(pxyz_idx.is_cuda(), "pxyz_idx must be a CUDA tensor");
    TORCH_CHECK(grad_log_prob.is_cuda(), "grad_log_prob must be a CUDA tensor");
    TORCH_CHECK(grad_log_prob.dtype() == torch::kFloat32, "grad_log_prob must be float32");

    const int N = pxyz_idx.size(1);
    auto idx_contig = pxyz_idx.contiguous();
    auto grad_contig = grad_log_prob.contiguous();
    std::vector<torch::Tensor> grads;
    grads.reserve(nlayers);

    std::vector<const float*> h_pxyz(nlayers);
    std::vector<float*> h_grads(nlayers);
    for (int i = 0; i < nlayers; i++) {
        TORCH_CHECK(pxyz_layers[i].is_cuda(), "pxyz layers must be CUDA tensors");
        TORCH_CHECK(pxyz_layers[i].dtype() == torch::kFloat32, "pxyz layers must be float32");
        TORCH_CHECK(pxyz_layers[i].is_contiguous(), "pxyz layers must be contiguous");
        grads.push_back(torch::zeros_like(pxyz_layers[i]));
        h_pxyz[i] = pxyz_layers[i].data_ptr<float>();
        h_grads[i] = grads.back().data_ptr<float>();
    }

    const float** d_pxyz;
    float** d_grads;
    cudaMalloc(&d_pxyz, nlayers * sizeof(float*));
    cudaMalloc(&d_grads, nlayers * sizeof(float*));
    cudaMemcpy(d_pxyz, h_pxyz.data(), nlayers * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grads, h_grads.data(), nlayers * sizeof(float*), cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    logprob_backward_kernel<<<blocks, threads>>>(
        d_pxyz,
        idx_contig.data_ptr<int64_t>(),
        grad_contig.data_ptr<float>(),
        nlayers,
        N,
        d_grads);

    cudaFree(d_pxyz);
    cudaFree(d_grads);
    return grads;
}

__global__ void logprob_xyz_forward_kernel(
    const float* const* __restrict__ pxyz_ptrs,
    const float* __restrict__ xyz_norm,
    const int* __restrict__ d_res,
    const int* __restrict__ d_ncomp,
    const int* __restrict__ d_accum,
    const int* __restrict__ d_map_type,
    int nlayers,
    int fres,
    int N,
    float* __restrict__ log_prob,
    int64_t* __restrict__ pxyz_idx_out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int ix = (int)floorf(xyz_norm[tid * 3 + 0] * (float)fres);
    int iy = (int)floorf(xyz_norm[tid * 3 + 1] * (float)fres);
    int iz = (int)floorf(xyz_norm[tid * 3 + 2] * (float)fres);
    ix = min(max(ix, 0), fres - 1);
    iy = min(max(iy, 0), fres - 1);
    iz = min(max(iz, 0), fres - 1);

    int res0 = d_res[0];
    int denom0 = max(fres / res0, 1);
    int loc_x = ix / denom0;
    int loc_y = iy / denom0;
    int loc_z = iz / denom0;
    int res_accum = 1;
    float accum_logp = 0.0f;

    for (int ll = 0; ll < nlayers; ll++) {
        const int res = d_res[ll];
        const int ncomp = d_ncomp[ll];
        const int accum = d_accum[ll];

        int comp;
        if (d_map_type[ll] == 0)
            comp = 0;
        else if (d_map_type[ll] == 1)
            comp = tiled_map(loc_x, loc_y, loc_z, ncomp, accum);
        else
            comp = hashed_map(loc_x, loc_y, loc_z, ncomp);

        const int step = max(fres / (res_accum * res), 1);
        const int vx = (ix / step) % res;
        const int vy = (iy / step) % res;
        const int vz = (iz / step) % res;
        const int64_t flat_idx = (int64_t)comp * res * res * res
                               + (int64_t)vx * res * res
                               + (int64_t)vy * res
                               + (int64_t)vz;
        pxyz_idx_out[(int64_t)ll * N + tid] = flat_idx;
        const float p = fmaxf(pxyz_ptrs[ll][flat_idx], 1e-20f);
        accum_logp += logf(p);

        res_accum *= res;
        const int next_denom = max(fres / res_accum, 1);
        loc_x = ix / next_denom;
        loc_y = iy / next_denom;
        loc_z = iz / next_denom;
    }

    log_prob[tid] = accum_logp;
}

std::vector<torch::Tensor> logprob_xyz_forward_cuda(
    std::vector<torch::Tensor> pxyz_layers,
    torch::Tensor xyz_samples_norm,
    std::vector<int64_t> res_list,
    std::vector<int64_t> accum_list,
    std::vector<int64_t> mapping_types,
    int64_t fres)
{
    const int nlayers = (int)pxyz_layers.size();
    TORCH_CHECK((int)res_list.size() == nlayers, "res_list length must match pxyz layers");
    TORCH_CHECK((int)accum_list.size() == nlayers, "accum_list length must match pxyz layers");
    TORCH_CHECK((int)mapping_types.size() == nlayers, "mapping_types length must match pxyz layers");
    TORCH_CHECK(xyz_samples_norm.is_cuda(), "xyz_samples_norm must be a CUDA tensor");
    TORCH_CHECK(xyz_samples_norm.dtype() == torch::kFloat32, "xyz_samples_norm must be float32");
    TORCH_CHECK(xyz_samples_norm.size(1) == 3, "xyz_samples_norm must have shape [N, 3]");

    auto xyz_contig = xyz_samples_norm.contiguous();
    const int N = xyz_contig.size(0);
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(xyz_samples_norm.device());
    auto opts_l = torch::TensorOptions().dtype(torch::kInt64).device(xyz_samples_norm.device());
    auto log_prob = torch::empty({N, 1}, opts_f);
    auto pxyz_idx = torch::empty({nlayers, N}, opts_l);

    std::vector<const float*> h_pxyz(nlayers);
    std::vector<int> h_res(nlayers), h_ncomp(nlayers), h_accum(nlayers), h_map(nlayers);
    for (int i = 0; i < nlayers; i++) {
        TORCH_CHECK(pxyz_layers[i].is_cuda(), "pxyz layers must be CUDA tensors");
        TORCH_CHECK(pxyz_layers[i].dtype() == torch::kFloat32, "pxyz layers must be float32");
        TORCH_CHECK(pxyz_layers[i].is_contiguous(), "pxyz layers must be contiguous");
        h_pxyz[i] = pxyz_layers[i].data_ptr<float>();
        h_res[i] = (int)res_list[i];
        h_accum[i] = (int)accum_list[i];
        h_map[i] = (int)mapping_types[i];
        h_ncomp[i] = (int)(pxyz_layers[i].numel() / (res_list[i] * res_list[i] * res_list[i]));
    }

    const float** d_pxyz;
    int *d_res, *d_ncomp, *d_accum, *d_map;
    cudaMalloc(&d_pxyz, nlayers * sizeof(float*));
    cudaMalloc(&d_res, nlayers * sizeof(int));
    cudaMalloc(&d_ncomp, nlayers * sizeof(int));
    cudaMalloc(&d_accum, nlayers * sizeof(int));
    cudaMalloc(&d_map, nlayers * sizeof(int));

    cudaMemcpy(d_pxyz, h_pxyz.data(), nlayers * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res.data(), nlayers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ncomp, h_ncomp.data(), nlayers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_accum, h_accum.data(), nlayers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_map, h_map.data(), nlayers * sizeof(int), cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    logprob_xyz_forward_kernel<<<blocks, threads>>>(
        d_pxyz,
        xyz_contig.data_ptr<float>(),
        d_res,
        d_ncomp,
        d_accum,
        d_map,
        nlayers,
        (int)fres,
        N,
        log_prob.data_ptr<float>(),
        pxyz_idx.data_ptr<int64_t>());

    cudaFree(d_pxyz);
    cudaFree(d_res);
    cudaFree(d_ncomp);
    cudaFree(d_accum);
    cudaFree(d_map);
    return {log_prob, pxyz_idx};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_pyramid_cuda", &sample_pyramid_cuda,
          "Fused sampling from hashed probability pyramid (CUDA)");
    m.def("logprob_forward_cuda", &logprob_forward_cuda,
          "Fused log-probability gather for hashed probability pyramid (CUDA)");
    m.def("logprob_backward_cuda", &logprob_backward_cuda,
          "Fused log-probability scatter-add backward for hashed probability pyramid (CUDA)");
    m.def("logprob_xyz_forward_cuda", &logprob_xyz_forward_cuda,
          "Fused log-probability query from normalized xyz samples (CUDA)");
}
