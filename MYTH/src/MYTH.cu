// Overarching file collecting some utility functions + all implementations
// gpu layers/functions are only implemented for THCudaTensor (i.e. single-precision)

#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include "src/utils.hpp"

#define MAX_ACCUMULATED_CHANNELS (32)

__inline__ __device__ float MYTH_get_point_depth(float *camera, float *win) {
    return camera[8]*win[0] + camera[9]*win[1] + camera[10]*win[2]+ camera[11];
}

__inline__ __device__ bool MYTH_project_point(float *camera, float *win, int *out, int input_width, int input_height) {
    float cx = camera[0]*win[0] + camera[1]*win[1] + camera[2]*win[2] + camera[3];
    float cy = camera[4]*win[0] + camera[5]*win[1] + camera[6]*win[2] + camera[7];
    float cz = MYTH_get_point_depth(camera, win);
    out[0] = int(cx / cz + 0.5f);
    out[1] = int(cy / cz + 0.5f);
    return (out[0] >= 0) && (out[1] >= 0) && (out[0]<input_width) && (out[1]<input_height);
}

__inline__ __device__ bool MYTH_project_pointf(float *camera, float *win, float *out, int input_width, int input_height) {
    float cx = camera[0]*win[0] + camera[1]*win[1] + camera[2]*win[2] + camera[3];
    float cy = camera[4]*win[0] + camera[5]*win[1] + camera[6]*win[2] + camera[7];
    float cz = MYTH_get_point_depth(camera, win);
    out[0] = cx / cz;
    out[1] = cy / cz;
    return (out[0] >= 0) && (out[1] >= 0) && (out[0]<input_width) && (out[1]<input_height);
}

__inline__ __device__ void MYTH_unproject_point(float *camloc, float *invKR, int u, int v, float z, float *out) {
    out[0] = camloc[0] + (invKR[0] * (u + 0.5f) + invKR[1] * (v + 0.5f) + invKR[2]) * z;
    out[1] = camloc[1] + (invKR[3] * (u + 0.5f) + invKR[4] * (v + 0.5f) + invKR[5]) * z;
    out[2] = camloc[2] + (invKR[6] * (u + 0.5f) + invKR[7] * (v + 0.5f) + invKR[8]) * z;
}

__device__ static float MYTH_atomicMinf(float* addr, float val)
{
    float old;
    old = (val >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(val))) :
         __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(val)));

    return old;
}

__inline__ __device__ void MYTH_applyWorldT_3d(float *worldt_p, float sx, float sy, float sz, float *out) {
    out[0] = worldt_p[0]*sx + worldt_p[1]*sy + worldt_p[2]*sz + worldt_p[3];
    out[1] = worldt_p[4]*sx + worldt_p[5]*sy + worldt_p[6]*sz + worldt_p[7];
    out[2] = worldt_p[8]*sx + worldt_p[9]*sy + worldt_p[10]*sz+ worldt_p[11];
}

__inline__ __device__ void MYTH_applyWorldT(float *worldt_p, int x, int y, int z, int cube_dimension, float *out) {
    float sx = (x + 0.5f)/cube_dimension, sy = (y + 0.5f)/cube_dimension, sz = (z + 0.5f)/cube_dimension;
    MYTH_applyWorldT_3d(worldt_p, sx, sy, sz, out);
}

// #include "InvertCams.cu"
// #include "DepthReprojectionNeighbours.cu"
// #include "DepthReprojectionNeighboursAlt.cu"
// #include "DepthColorAngleReprojectionNeighbours.cu"

// we get as input a set of depth maps (B x N x 1 x H x W)
// and their cameras (B x N x 3 x 4), and generate the synthetic depth maps in the first views' image plane
// yes, the first image is simply left unchanged.
// we also pass the angle between both camera's viewing directions for each reprojected point, through the cos

// input dimension: (B x N x 1 x H x W)
// output dimension: (B x N x 1 x H x W)

__global__ void DepthColorAngleReprojectionNeighbours_forward_depth_kernel(
                    float *input,
                    float *output,
                    float *cameras,
                    float *invKRs,
                    float *camlocs,
                    int batch_size,
                    int nrcams,
                    int input_height,
                    int input_width)
{
    int colstep = 1;
    int rowstep = colstep * input_width;
    int camstep = rowstep * input_height;
    int btcstep = camstep * nrcams;

    int clocs_camstep = 3;
    int clocs_btcstep = clocs_camstep * nrcams;
    int invKRs_camstep = 9;
    int invKRs_btcstep = invKRs_camstep * nrcams;

    int proj[2];
    float w[3];

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < batch_size; b += blockDim.x * gridDim.x) {
        float *camera0 = cameras + b * nrcams * 12;
    for (int n = 0; n < nrcams; n++) {
    for (int in_row = blockIdx.y * blockDim.y + threadIdx.y; in_row < input_height; in_row += blockDim.y * gridDim.y) {
    for (int in_col = blockIdx.z * blockDim.z + threadIdx.z; in_col < input_width;  in_col += blockDim.z * gridDim.z) {
        float depth_n = input[b * btcstep + n * camstep + in_row * rowstep + in_col * colstep];
        if(n == 0) {
            // simply copy the first camera's depth map
            output[b * btcstep + n * camstep + in_row * rowstep + in_col * colstep] = depth_n;
        }
        else if(depth_n > 0) {
            // cast this point into space
            float *camloc = camlocs + b * clocs_btcstep + n * clocs_camstep;
            float *invKR = invKRs + b * invKRs_btcstep + n * invKRs_camstep;
            MYTH_unproject_point(camloc, invKR, in_col, in_row, depth_n, w);
            // project it onto the first camera again
            if(MYTH_project_point(camera0, w, proj, input_width, input_height)) {
                MYTH_atomicMinf(
                    output + b * btcstep + n * camstep + proj[1] * rowstep + proj[0] * colstep,
                    MYTH_get_point_depth(camera0, w)
                );
            }
        }
    }
    }
    }
    }
}

__global__ void DepthColorAngleReprojectionNeighbours_forward_colorangle_kernel(
                    float *input_depth,
                    float *output_depth,
                    float *input_color,
                    float *output_color,
                    float *output_angle,
                    float *cameras,
                    float *invKRs,
                    float *camlocs,
                    int batch_size,
                    int nrcams,
                    int nrchans,
                    int input_height,
                    int input_width)
{
    int colstep = 1;
    int rowstep = colstep * input_width;

    int camstep_d = rowstep * input_height;
    int btcstep_d = camstep_d * nrcams;

    int chnstep_c = rowstep * input_height;
    int camstep_c = chnstep_c * nrchans;
    int btcstep_c = camstep_c * nrcams;

    int clocs_camstep = 3;
    int clocs_btcstep = clocs_camstep * nrcams;
    int invKRs_camstep = 9;
    int invKRs_btcstep = invKRs_camstep * nrcams;

    int proj[2];
    float w[3];

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < batch_size; b += blockDim.x * gridDim.x) {
        float *camera0 = cameras + b * nrcams * 12;
        float *camloc0 = camlocs + b * clocs_btcstep;
        for (int n = 0; n < nrcams; n++) {
            for (int in_row = blockIdx.y * blockDim.y + threadIdx.y; in_row < input_height; in_row += blockDim.y * gridDim.y) {
                for (int in_col = blockIdx.z * blockDim.z + threadIdx.z; in_col < input_width;  in_col += blockDim.z * gridDim.z) {
                    float depth_n = input_depth[b * btcstep_d + n * camstep_d + in_row * rowstep + in_col * colstep];
                    if(n == 0) {
                        // simply copy the first camera's color
                        for(int c = 0; c < nrchans; c++) {
                            float color_n = input_color[b * btcstep_c + n * camstep_c + c * chnstep_c + in_row * rowstep + in_col * colstep];
                            output_color[b * btcstep_c + n * camstep_c + c * chnstep_c + in_row * rowstep + in_col * colstep] = color_n;
                        }
                        output_angle[b * btcstep_d + n * camstep_d + in_row * rowstep + in_col * colstep] = 1.0f;
                    } else if(depth_n > 0) {
                        // cast this point into space
                        float *camloc = camloc0 + n * clocs_camstep;
                        float *invKR = invKRs + b * invKRs_btcstep + n * invKRs_camstep;
                        MYTH_unproject_point(camloc, invKR, in_col, in_row, depth_n, w);
                        // project it onto the first camera again
                        if(MYTH_project_point(camera0, w, proj, input_width, input_height)) {
                            float zbuffer = output_depth[b * btcstep_d + n * camstep_d + proj[1] * rowstep + proj[0] * colstep];
                            float this_z = MYTH_get_point_depth(camera0, w);
                            if(this_z <= zbuffer) {
                                for(int c = 0; c < nrchans; c++) {
                                    float color_n = input_color[b * btcstep_c + n * camstep_c + c * chnstep_c + in_row * rowstep + in_col * colstep];
                                    output_color[b * btcstep_c + n * camstep_c + c * chnstep_c + proj[1] * rowstep + proj[0] * colstep] = color_n;
                                }
                                // also pass the cosine of the angle between the viewing lines as a feature
                                float angle = 0.0f;
                                float norm = 0.0f, norm0 = 0.0f;
                                for(int i = 0; i < 3; i++) {
                                    angle += (camloc[i] - w[i]) * (camloc0[i] - w[i]);
                                    norm += (camloc[i] - w[i]) * (camloc[i] - w[i]);
                                    norm0 += (camloc0[i] - w[i]) * (camloc0[i] - w[i]);
                                }
                                output_angle[b * btcstep_d + n * camstep_d + proj[1] * rowstep + proj[0] * colstep] = angle / sqrt(norm * norm0);
                            }
                        }
                    }
                }
            }
        }
    }
}

//the input dimension is (B x N x 1 x H x W)
//the output dimension is (B x N x 1 x H x W)
void DepthColorAngleReprojectionNeighbours_updateOutput_gpu(
    at::Tensor input_depth,
    at::Tensor output_depth,
    at::Tensor input_color,
    at::Tensor output_color,
    at::Tensor output_angle,
    at::Tensor cameras,
    at::Tensor invKRs,
    at::Tensor camlocs)
{
    int blkdim = 16;
    int batch_size = input_depth.size(0);
    int nrviews = input_depth.size(1);
    int color_channels = input_color.size(2);
    int input_height = input_depth.size(3);
    int input_width = input_depth.size(4);

    // we will use one thread for all depth hypotheses, to save some calculations regarding the directions and matrix inversions
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1,ceil(output_depth.size(3)*1.0f/blkdim), ceil(output_depth.size(4)*1.0f/blkdim));
    float *input_depth_p = input_depth.data<float>(); // THCudaTensor_data(state, input_depth);
    float *output_depth_p = output_depth.data<float>(); // THCudaTensor_data(state, output_depth);
    float *input_color_p = input_color.data<float>(); // THCudaTensor_data(state, input_color);
    float *output_color_p = output_color.data<float>(); // THCudaTensor_data(state, output_color);
    float *output_angle_p = output_angle.data<float>(); // THCudaTensor_data(state, output_angle);
    float *cameras_p = cameras.data<float>(); // THCudaTensor_data(state, cameras);
    float *invKRs_p = invKRs.data<float>(); // THCudaTensor_data(state, invKRs);
    float *camlocs_p = camlocs.data<float>(); // THCudaTensor_data(state, camlocs);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    DepthColorAngleReprojectionNeighbours_forward_depth_kernel<<<grid, block, 0, stream>>>(input_depth_p, output_depth_p, cameras_p, invKRs_p, camlocs_p, batch_size, nrviews, input_height, input_width);
    cudaStreamSynchronize(stream);
    DepthColorAngleReprojectionNeighbours_forward_colorangle_kernel<<<grid, block, 0, stream>>>(input_depth_p, output_depth_p, input_color_p, output_color_p, output_angle_p, cameras_p, invKRs_p, camlocs_p, batch_size, nrviews, color_channels, input_height, input_width);

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    throw std::runtime_error(Formatter() << "CUDA kernel failed : " << std::to_string(err));
}


__global__ void DepthColorAngleReprojectionNeighbours_backward_color_kernel(
                    float *input_depth,
                    float *output_depth,
                    float *dloss_input_color,
                    float *dloss_output_color,
                    float *cameras,
                    float *invKRs,
                    float *camlocs,
                    int batch_size,
                    int nrcams,
                    int nrchans,
                    int input_height,
                    int input_width)
{
    int colstep = 1;
    int rowstep = colstep * input_width;

    int camstep_d = rowstep * input_height;
    int btcstep_d = camstep_d * nrcams;

    int chnstep_c = rowstep * input_height;
    int camstep_c = chnstep_c * nrchans;
    int btcstep_c = camstep_c * nrcams;

    int clocs_camstep = 3;
    int clocs_btcstep = clocs_camstep * nrcams;
    int invKRs_camstep = 9;
    int invKRs_btcstep = invKRs_camstep * nrcams;

    int proj[2];
    float w[3];

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < batch_size; b += blockDim.x * gridDim.x) {
        float *camera0 = cameras + b * nrcams * 12;
        float *camloc0 = camlocs + b * clocs_btcstep;
    for (int n = 0; n < nrcams; n++) {
    for (int in_row = blockIdx.y * blockDim.y + threadIdx.y; in_row < input_height; in_row += blockDim.y * gridDim.y) {
    for (int in_col = blockIdx.z * blockDim.z + threadIdx.z; in_col < input_width;  in_col += blockDim.z * gridDim.z) {
        float depth_n = input_depth[b * btcstep_d + n * camstep_d + in_row * rowstep + in_col * colstep];
        if(n == 0) {
            // simply copy the first camera's color
            for(int c = 0; c < nrchans; c++) {
                float dloss_output_n = dloss_output_color[b * btcstep_c + n * camstep_c + c * chnstep_c + in_row * rowstep + in_col * colstep];
                atomicAdd(
                    dloss_input_color + b * btcstep_c + n * camstep_c + c * chnstep_c + in_row * rowstep + in_col * colstep,
                    dloss_output_n
                );
            }
        }
        else if(depth_n > 0) {
            // cast this point into space
            float *camloc = camloc0 + n * clocs_camstep;
            float *invKR = invKRs + b * invKRs_btcstep + n * invKRs_camstep;
            MYTH_unproject_point(camloc, invKR, in_col, in_row, depth_n, w);
            // project it onto the first camera again
            if(MYTH_project_point(camera0, w, proj, input_width, input_height)) {
                float zbuffer = output_depth[b * btcstep_d + n * camstep_d + proj[1] * rowstep + proj[0] * colstep];
                float this_z = MYTH_get_point_depth(camera0, w);
                if(this_z <= zbuffer) {
                    for(int c = 0; c < nrchans; c++) {
                        float dloss_output_n = dloss_output_color[b * btcstep_c + n * camstep_c + c * chnstep_c + proj[1] * rowstep + proj[0] * colstep];
                        atomicAdd(
                            dloss_input_color + b * btcstep_c + n * camstep_c + c * chnstep_c + in_row * rowstep + in_col * colstep,
                            dloss_output_n
                        );
                    }
                }
            }
        }
    }
    }
    }
    }
}

//the input dimension is (B x N x 1 x H x W)
//the output dimension is (B x N x 1 x H x W)
void DepthColorAngleReprojectionNeighbours_updateGradInput_gpu(
    at::Tensor input_depth,
    at::Tensor output_depth,
    at::Tensor dloss_input_color,
    at::Tensor dloss_output_color,
    at::Tensor cameras,
    at::Tensor invKRs,
    at::Tensor camlocs)
{
    int blkdim = 16;
    int batch_size = input_depth.size(0);
    int nrviews = input_depth.size(1);
    int color_channels = dloss_output_color.size(2);
    int input_height = input_depth.size(3);
    int input_width = input_depth.size(4);

    // we will use one thread for all depth hypotheses, to save some calculations regarding the directions and matrix inversions
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1,ceil(output_depth.size(3)*1.0f/blkdim), ceil(output_depth.size(4)*1.0f/blkdim));
    float *input_depth_p = input_depth.data<float>();
    float *output_depth_p = output_depth.data<float>();
    float *dloss_input_color_p = dloss_input_color.data<float>();
    float *dloss_output_color_p = dloss_output_color.data<float>();
    float *cameras_p = cameras.data<float>();
    float *invKRs_p = invKRs.data<float>();
    float *camlocs_p = camlocs.data<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    DepthColorAngleReprojectionNeighbours_backward_color_kernel<<<grid, block, 0, stream>>>(input_depth_p, output_depth_p, dloss_input_color_p, dloss_output_color_p, cameras_p, invKRs_p, camlocs_p, batch_size, nrviews, color_channels, input_height, input_width);

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    throw std::runtime_error(Formatter() << "CUDA kernel failed : " << std::to_string(err));
}


// Small bit of CUDA code for inverting a bunch of 3x3 cameras, and getting their camera locations

__global__ void InvertCams_kernel(
    float *cameras,
    float *invKRs,
    float *camlocs,
    int B
) {
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
        float *cam = cameras + b*12;
        float KR_det = cam[0] * cam[5] * cam[10];
                KR_det += cam[1] * cam[6] * cam[8];
                KR_det += cam[2] * cam[4] * cam[9];
                KR_det -= cam[2] * cam[5] * cam[8];
                KR_det -= cam[1] * cam[4] * cam[10];
                KR_det -= cam[0] * cam[6] * cam[9];
        float KR_det_inv = 1.0f/KR_det;
        float *invKR = invKRs + b * 9;
        invKR[0] = (cam[5] * cam[10] - cam[6] * cam[9]) * KR_det_inv;
        invKR[1] = (cam[2] * cam[9] - cam[1] * cam[10]) * KR_det_inv;
        invKR[2] = (cam[1] * cam[6] - cam[2] * cam[5]) * KR_det_inv;
        invKR[3] = (cam[6] * cam[8] - cam[4] * cam[10]) * KR_det_inv;
        invKR[4] = (cam[0] * cam[10] - cam[2] * cam[8]) * KR_det_inv;
        invKR[5] = (cam[2] * cam[4] - cam[0] * cam[6]) * KR_det_inv;
        invKR[6] = (cam[4] * cam[9] - cam[5] * cam[8]) * KR_det_inv;
        invKR[7] = (cam[1] * cam[8] - cam[0] * cam[9]) * KR_det_inv;
        invKR[8] = (cam[0] * cam[5] - cam[1] * cam[4]) * KR_det_inv;
        float *camloc = camlocs + b * 3;
        camloc[0] = -(invKR[0] * cam[3] + invKR[1] * cam[7] + invKR[2] * cam[11]);
        camloc[1] = -(invKR[3] * cam[3] + invKR[4] * cam[7] + invKR[5] * cam[11]);
        camloc[2] = -(invKR[6] * cam[3] + invKR[7] * cam[7] + invKR[8] * cam[11]);
    }
}

// yes this is horribly slow/inefficient/whatever
// I just don't want to copying to CPU for this
void InvertCams_gpu(
    at::Tensor cameras,
    at::Tensor invKRs,
    at::Tensor camlocs)
{
    int B = cameras.size(0);

    const dim3 block = dim3(1, 1, 1);
    const dim3 grid = dim3(1, 1, 1);

    float *cameras_p     = cameras.data<float>();
    float *invKRs_p      = invKRs.data<float>();
    float *camlocs_p     = camlocs.data<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    InvertCams_kernel<<<grid, block, 0, stream>>>(
        cameras_p,
        invKRs_p,
        camlocs_p,
        B
    );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    throw std::runtime_error(Formatter() << "CUDA kernel failed : " << std::to_string(err));
}