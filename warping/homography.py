'''
Homography warping module: 
'''
import numpy as np
import torch.nn.functional as F
import torch

import MYTH


def warp_cfd(depths, cfds, Ks, R_ts, z_near=0.1, z_far=100, crop_at=None):
    b_size, h, w = depths.size(0), depths.size(2), depths.size(3)
    if crop_at is None:
        crop_at = torch.zeros(b_size, 2).cuda()
    crop_at = crop_at[:, [1, 0]] # swap height <--> width
    pos = torch.cat((crop_at, torch.zeros(b_size, 1).cuda()), dim=1)
    pos = pos.unsqueeze(-1).unsqueeze(-1).float()

    Rs, ts = R_ts[:, :3, :3], R_ts[:, :3, [3]]
    pixel_coord = np.indices([w, h]).astype(np.float32)
    pixel_coord = np.concatenate((pixel_coord, np.ones((1, w, h))), axis=0)
    pixel_coord = torch.from_numpy(pixel_coord)
    pixel_coords = torch.stack(b_size * [pixel_coord]).float().cuda() + pos #).cuda()
    KRKiUV_T = Ks.matmul(Rs.matmul(torch.inverse(Ks)))
    KRKiUV_T = KRKiUV_T.matmul(pixel_coords.view(b_size, 3, -1))
    KT_T = Ks.matmul(ts)
    depths = depths.permute(0, 1, 3, 2).reshape(b_size, -1) # size (b_size, 1, (w*h))
    mask_depths = (depths > z_near) & (depths < z_far)

    cfds = cfds.permute(0, 1, 3, 2).reshape(b_size, cfds.size(1), -1)
    warped_depths = torch.zeros((b_size, 1, w, h), dtype=torch.float32).cuda()
    warped_cfds = torch.zeros((b_size, cfds.size(1), w, h), dtype=torch.float32).cuda()  # - 1

    for i in range(b_size):
        if len(mask_depths[i].nonzero()) > 0:
            transformed = KRKiUV_T[i, :, mask_depths[i]] * depths[i, mask_depths[i]] + KT_T[i]
            trans_depths = transformed[2, :].unsqueeze(0)
            warp_uv = torch.div(transformed[0:2, :], (trans_depths + 1e-6)).round().long() # .float()

    # trans_depths, indices = trans_depths.sort(dim=2, descending=True)
    # print(trans_depths.size(), indices.size(), warp_uv.size())
    # for i in range(b_size):
    #     warp_uv[i, 0] = warp_uv[i, 0][indices[i]]
    #     warp_uv[i, 1] = warp_uv[i, 1][indices[i]]
    #     cfds[i, 0] = cfds[i, 0][indices[i]]
    #     cfds[i, 1] = cfds[i, 1][indices[i]]
    #     cfds[i, 2] = cfds[i, 2][indices[i]]
    # warp_uv = warp_uv[indices.repeat(1, 2, 1)]
            u, v = crop_at[i, 0].item(), crop_at[i, 1].item()
            mask = (warp_uv[0, :] >= u) & (warp_uv[0, :] < (u+w)) & (warp_uv[1, :] >= v) & (warp_uv[1, :] < (v+h))
    # for i in range(b_size):
            warped_depths[i, 0, warp_uv[0, mask]-u, warp_uv[1, mask]-v] = trans_depths[0, mask]
            cfd_i = cfds[i, :, mask_depths[i]]
            warped_cfds[i, :, warp_uv[0, mask]-u, warp_uv[1, mask]-v] = cfd_i[:, mask]

    # warp_uv = warp_uv.view(b_size, 2, w, h)
    # warped_cfds = F.grid_sample(cfds, warp_uv.permute(0, 3, 2, 1), mode='nearest')
    # warped_depths = warped_depths.view(b_size, 1, w, h)
    return warped_depths.permute(0, 1, 3, 2), warped_cfds.permute(0, 1, 3, 2)


def warping(depths_prev, cfds_prev, K_prev, E_prev, K_cur, E_cur):
    P_prev = torch.cat((K_prev.matmul(E_prev[:, :3, :3]), K_prev.matmul(E_prev[:, :3, [3]])), -1)
    P_cur = torch.cat((K_cur.matmul(E_cur[:, :3, :3]), K_cur.matmul(E_cur[:, :3, [3]])), -1)
    camera_params = torch.cat((P_cur.unsqueeze(1), P_prev.unsqueeze(1)), 1)
    depths = depths_prev.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    cfds = cfds_prev.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    warped_depths, warped_cfds, _ = MYTH.DepthColorAngleReprojectionNeighbours.apply(depths, cfds, camera_params, 1.0)
    warped_depths = warped_depths[:, -1]
    warped_cfds = warped_cfds[:, -1]
    return warped_depths, warped_cfds


def img_to_pointcloud(img, depth, K, Rt):
    import open3d as o3d
    # rgbd = o3d.geometry.RGBDImage()
    rgb = o3d.geometry.Image(img)
    depth = o3d.geometry.Image(depth)
    #rgbd.create_from_color_and_depth()
    rgbd = o3d.geometry.create_rgbd_image_from_color_and_depth(rgb, depth, depth_scale=1.0, depth_trunc=500.0,
                                                               convert_rgb_to_intensity=False)
    # pc = o3d.geometry.PointCloud()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(int(cx*2+1), int(cy*2+1), fx, fy, cx, cy)
    # pc = pc.create_from_rgbd_image(rgbd, intrinsic, Rt)
    pc = o3d.create_point_cloud_from_rgbd_image(rgbd, intrinsic, Rt)
    # pc = o3d.create_point_cloud_from_depth_image(depth, intrinsic, Rt, depth_scale=1.0, depth_trunc=100.0)
    print(pc.has_colors())
    o3d.visualization.draw_geometries([pc])


def get_rel_extrinsicM(ext_ref, ext_src):
    ''' Get the extrinisc matrix from ref_view to src_view '''
    return ext_src.dot(np.linalg.inv(ext_ref))


def test_warping(cur_sample, prev_sample):
    import matplotlib.pyplot as plt
    imgs, sdmaps, Es, Ks = cur_sample[0][:-1]
    crop_at = cur_sample[0][-1]
    gt_depths, masks = cur_sample[1]
    # if prev_sample is not None:
    # Es[0] = torch.zeros((4, 4), dtype=torch.float32)
    prev_imgs, _, prev_E, prev_K = prev_sample[0][:-1]
    prev_depths, masks = prev_sample[1]
    # prev_imgs[0] = torch.zeros((1, 352, 1216), dtype=torch.float32) - 1
    # gt_depths[0] = torch.zeros((1, 352, 1216), dtype=torch.float32)
    print('Previous trans_matrix: ', prev_E)
    print('Current trans_matrix', Es)
    warped_depths, warped_imgs = warping(prev_depths.cuda(), prev_imgs.cuda(),
                                         prev_K.cuda(), prev_E.cuda(), Ks.cuda(), Es.cuda())
    # warped_depths, warped_imgs = warp_cfd(prev_depths.cuda(), prev_imgs.cuda(), Ks.cuda(), Es.cuda(), crop_at=crop_at.cuda())
    print(warped_depths[0].min(), warped_depths[0].max())
    print(warped_imgs[0].min(), warped_imgs[0].max())
    warped_depths = warped_depths.cpu()
    warped_imgs = warped_imgs.cpu()
    print(warped_depths.size(), warped_imgs.size())
    for i in range(gt_depths.size(0)):
        img = imgs[i]
        img = img.permute(1, 2, 0).numpy()
        img = np.array(255 * img).astype(np.uint8)

        warped_img = warped_imgs[i]
        warped_img = warped_img.permute(1, 2, 0).numpy()
        warped_img = np.array(255 * warped_img).astype(np.uint8)
        # warped_img = warped_img / warped_img.max()
        # print(warped_img.min(), warped_img.max())

        prev_img = prev_imgs[i]
        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(3, 1, 1)
        plt.imshow(img)
        plt.xlabel('Current frame')
        ax2 = fig.add_subplot(3, 1, 2)
        plt.imshow(prev_img.permute(1, 2, 0).numpy())
        plt.xlabel('Previous frame')
        ax3 = fig.add_subplot(3, 1, 3)
        plt.imshow(warped_img)
        plt.xlabel('Warped image from previous frame to current frame')
        plt.show()


if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import matplotlib
    near_pose = np.loadtxt('test_warping/near_pose.txt')
    ref_pose = np.loadtxt('test_warping/ref_pose.txt')
    K = np.loadtxt('test_warping/cam_intrinsics.txt')
    K = torch.from_numpy(K)
    # h, w = 480, 752
    # K[0, 2] = w/2 #(w - 1) / 2.
    # K[1, 2] = h/2 #(h - 1) / 2.
    trans = transforms.Compose([transforms.ToTensor()])
    img_near = Image.open('test_warping/near_cfd.png')
    img_near = trans(img_near)
    img_ref = Image.open('test_warping/ref_cfd.png')
    img_ref = trans(img_ref)
    near_depth = Image.open('test_warping/near_depth.png')
    # ch, cw = gt_depth1.shape[0]//2, gt_depth1.shape[1]//2
    # gt_depth1 = gt_depth1[(ch-230):(ch+230), (cw-375):(cw+375)]
    near_depth = trans(near_depth)
    near_depth = near_depth / 100.
    # rel_E = torch.from_numpy(get_rel_extrinsicM(cam_pose1, cam_pose2))
    # warped_depth, warped_img = warp_cfd(gt_depth1.unsqueeze(0).float().cuda(), img1.unsqueeze(0).float().cuda(),
    #                                     K.unsqueeze(0).float().cuda(), rel_E.unsqueeze(0).float().cuda())#,
                                        #crop_at=np.array([[100, 200]], dtype=np.float32))
    K_torch = K.unsqueeze(0).float().cuda()
    near_pose = torch.from_numpy(near_pose)
    ref_pose = torch.from_numpy(ref_pose)
    warped_depth, warped_img = warping(near_depth.unsqueeze(0).float().cuda(), img_near.unsqueeze(0).float().cuda(),
                                       K_torch, near_pose.unsqueeze(0).float().cuda(), K_torch,
                                       ref_pose.unsqueeze(0).float().cuda())
    warped_depth *= 100
    warped_depth = warped_depth.cpu().squeeze(0).squeeze(0).numpy()
    warped_img = warped_img.cpu().squeeze(0).squeeze(0).numpy()
    warped_depth = warped_depth.astype(np.uint16)
    warped_img = (warped_img * 255).astype(np.uint8)
    warped_img[0, 0] = 0
    warped_img[1, 1] = 255
    warped_depth[warped_depth > 6400] = 6400
    warped_depth[0, 0] = 0
    cmap = matplotlib.cm.jet
    plt.imsave('warped_depth.eps', warped_depth, format='eps', dpi=1000, vmin=0, cmap=cmap)
    plt.imsave('warped_cfd.eps', warped_img, format='eps', dpi=1000, vmin=0)


    # d_min, d_max, n_d_candi = 0.5, 10.0, 64
    # idepth_base = 1.0 / d_max
    # idepth_step = (1.0 / d_min - 1.0 / d_max) / (n_d_candi - 1)
    # d_candi = np.arange(n_d_candi).astype(np.float32)
    # d_candi = 1.0 / (idepth_base + d_candi * idepth_step)
    # d_candi = torch.from_numpy(d_candi)
    # cost_vol = get_volume(img1.unsqueeze(0).float(), img2.unsqueeze(0).float(),
    #                     K.unsqueeze(0).float(), rel_E.unsqueeze(0).float(), d_candi)
    # kernel = torch.zeros(cost_vol.size(0), 1, 3, 3) + 1./9
    # cost_vol = F.conv2d(cost_vol.unsqueeze(0), kernel, groups=cost_vol.size(0)).squeeze(0)
    # cost_vol = F.softmax(-cost_vol, dim=0)
    # cost_vol = F.normalize(cost_vol, dim=0, p=1)
    # cfd, depth = torch.max(cost_vol, dim=0)
    # depth = depth.float().numpy()
    # depth = 1.0 / (idepth_base + depth * idepth_step)
    # print(img_near.size(), img_ref.size(), warped_depth.size(), warped_img.size())
    # fig = plt.figure(figsize=(20, 20))
    # ax1 = fig.add_subplot(2, 2, 1)
    # plt.imshow(near_depth.squeeze(0).numpy())
    # ax2 = fig.add_subplot(2, 2, 2)
    # plt.imshow(img_near.squeeze(0).numpy())
    # ax3 = fig.add_subplot(2, 2, 3)
    # plt.imshow(warped_depth.cpu().squeeze(0).squeeze(0).numpy())
    # # plt.colorbar()
    # ax4 = fig.add_subplot(2, 2, 4)
    # plt.imshow(warped_img.cpu().squeeze(0).squeeze(0).numpy())
    # # plt.colorbar()
    # plt.show()