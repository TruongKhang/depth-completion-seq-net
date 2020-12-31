"""
Module providing wrappers to the MYTH FFI calls, in the form of simple functions,
torch.nn.Modules, or whatever else is applicable.
"""

import torch
import Warping


class DepthColorAngleReprojectionNeighbours(torch.autograd.Function):
    """
    Neighbour depth reprojection

    The first camera is assumed to be the center camera.
    All of the other view's points are reprojected onto that image plane.
    Zero-depth pixels are ignored, the first depth map is obviously left unchanged.

    Colour is also being reprojected along (following that zbuffer)

    Arguments:
        images -- the input views (B x N x C x H x W)
        depths -- the input views (B x N x 1 x H x W)
        cameras -- their cameras (B x N x 3 x 4)
        scale -- when the images have been scaled compared to the input camera matrices

    Outputs:
        output_color -- the reprojected colors (B x N x C x H x W)
        output_depth -- the reprojected depths (B x N x 1 x H x W)
    """

    @staticmethod
    def forward(ctx, depths, images, cameras, scale):
        B = depths.shape[0]
        N = depths.shape[1]
        C = images.shape[2]
        H = depths.shape[3]
        W = depths.shape[4]

        sentinel = 1e9
        output_depth = depths.new_full((B, N, 1, H, W), fill_value=sentinel)
        output_color = images.new_full((B, N, C, H, W), fill_value=0.0)
        output_angle = depths.new_full((B, N, 1, H, W), fill_value=0.0)

        camlocs = depths.new_empty(B, N, 3, 1)
        invKRs = depths.new_empty(B, N, 3, 3)
        if scale != 1.0:
            for b in range(B):
                for n in range(N):
                    cameras[b][n] = cameras[b][n].clone()
                    cameras[b][n][:2,:] = cameras[b][n][:2,:] * scale
                    invKRs[b][n] = torch.inverse(cameras[b][n][:3, :3]).contiguous()
                    camlocs[b][n] = - torch.mm(invKRs[b][n], cameras[b][n][:3, 3:4])
        else:
            invKRs = torch.inverse(cameras[:, :, :3, :3]).contiguous()
            camlocs = -torch.matmul(invKRs, cameras[:, :, :3, [3]])
            # Warping.InvertCams_gpu(cameras.reshape(-1,3,4), invKRs.reshape(-1,3,3), camlocs.reshape(-1,3,1))

        Warping.warp(depths, output_depth, images, output_color, output_angle, cameras, invKRs, camlocs)

        output_depth[output_depth > sentinel / 10] = 0
        output_angle[output_angle != output_angle] = 0

        ctx.save_for_backward(depths, output_depth, cameras.clone(), camlocs.clone(), invKRs.clone())

        return output_depth, output_color, output_angle

    @staticmethod
    def backward(ctx, grad_output_depth, grad_output_color, grad_output_angle):
        # note: backprop currently only implemented for the color
        depths, output_depth, cameras, camlocs, invKRs = ctx.saved_variables

        grad_input_color = grad_output_color.new_zeros(grad_output_color.shape)

        Warping.grad_warping(
            depths, output_depth, grad_input_color, grad_output_color, cameras, invKRs, camlocs
        )

        return None, grad_input_color, None, None


# class DepthReprojectionNeighbours(torch.autograd.Function):
#     """
#     Neighbour depth reprojection

#     The first camera is assumed to be the center camera.
#     All of the other view's points are reprojected onto that image plane.
#     Zero-depth pixels are ignored, the first depth map is obviously left unchanged.

#     Arguments:
#         depths -- the input views (B x N x 1 x H x W)
#         cameras -- their cameras (B x N x 3 x 4)
#         scale -- when the images have been scaled compared to the input camera matrices

#     Outputs:
#         output_depth -- the reprojected depths (B x N x 1 x H x W)
#     """

#     @staticmethod
#     def forward(ctx, depths, cameras, scale):
#         B = depths.shape[0]
#         N = depths.shape[1]
#         H = depths.shape[3]
#         W = depths.shape[4]

#         sentinel = 1e9
#         output_depth = depths.new_full((B, N, 1, H, W), fill_value=sentinel)

#         camlocs = depths.new_empty(B, N, 3, 1)
#         invKRs = depths.new_empty(B, N, 3, 3)
#         if scale != 1.0:
#             for b in range(B):
#                 for n in range(N):
#                     cameras[b][n] = cameras[b][n].clone()
#                     cameras[b][n][:2,:] = cameras[b][n][:2,:] * scale
#                     invKRs[b][n] = torch.inverse(cameras[b][n][:3, :3]).contiguous()
#                     camlocs[b][n] = - torch.mm(invKRs[b][n], cameras[b][n][:3, 3:4])
#         else:
#             MYTH.InvertCams_gpu(cameras.reshape(-1,3,4), invKRs.reshape(-1,3,3), camlocs.reshape(-1,3,1))

#         MYTH.DepthReprojectionNeighbours_updateOutput_gpu(depths, output_depth, cameras, invKRs, camlocs)

#         output_depth[ output_depth > sentinel / 10] = 0

#         return output_depth

#     @staticmethod
#     def backward(ctx, grad_output_depth):
#         # not differentiable right now
#         return None, None, None, None