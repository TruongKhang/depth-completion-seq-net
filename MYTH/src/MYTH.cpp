#include<torch/extension.h>

void InvertCams_gpu(
    at::Tensor cameras,
    at::Tensor invKRs,
    at::Tensor camlocs
);

// void DepthReprojectionNeighbours_updateOutput_gpu(
//     at::Tensor *input_depth,
//     at::Tensor *output_depth,
//     at::Tensor *cameras,
//     at::Tensor *invKRs,
//     at::Tensor *camlocs
// );

// void DepthReprojectionNeighboursAlt_updateOutput_gpu(
//     at::Tensor *input_depth,
//     at::Tensor *output_depth,
//     at::Tensor *cameras,
//     at::Tensor *invKRs,
//     at::Tensor *camlocs
// );

void DepthColorAngleReprojectionNeighbours_updateOutput_gpu(
    at::Tensor input_depth,
    at::Tensor output_depth,
    at::Tensor input_color,
    at::Tensor output_color,
    at::Tensor output_angle,
    at::Tensor cameras,
    at::Tensor invKRs,
    at::Tensor camlocs
);

void DepthColorAngleReprojectionNeighbours_updateGradInput_gpu(
    at::Tensor input_depth,
    at::Tensor output_depth,
    at::Tensor dloss_input_color,
    at::Tensor dloss_output_color,
    at::Tensor cameras,
    at::Tensor invKRs,
    at::Tensor camlocs
);