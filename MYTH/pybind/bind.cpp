#include <torch/extension.h>
#include "src/MYTH.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("InvertCams_gpu", &InvertCams_gpu, "Inverse camera matrix");
    m.def("warp", &DepthColorAngleReprojectionNeighbours_updateOutput_gpu, "warping");
    m.def("grad_warping", &DepthColorAngleReprojectionNeighbours_updateGradInput_gpu, "compute gradient");
}