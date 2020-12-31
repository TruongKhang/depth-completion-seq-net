'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

# data loader for KITTI dataset #
# We will use pykitti module to help to read the image and camera pose #

import pykitti

import numpy as np
import os, sys
import math
import os.path
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as tfv_transform
import torch.nn.functional as F

import warping.View as View
from warping import homography as homo
from data_loader.transformation import BottomCrop, Crop, RandomCrop

np.random.seed(2020)
random.seed(2020)


def _read_split_file(filepath):
    """
    Read data split txt file provided by KITTI dataset authors
    """
    with open(filepath) as f:
        trajs = f.readlines()
    trajs = [x.strip() for x in trajs]

    return trajs 


def _read_IntM_from_pdata(p_data,  out_size=None, cam_id=2):
    """
    Get the intrinsic camera info from pykitti object - pdata
    raw_img_size - [width, height]
    """

    # IntM = np.zeros((4,4))
    raw_img_size = p_data.get_cam2(0).size if cam_id == 2 else p_data.get_cam3(0).size
    width = int(raw_img_size[0])
    height = int(raw_img_size[1])
    IntM = p_data.calib.K_cam2 if cam_id == 2 else p_data.calib.K_cam3
    focal_length = np.mean([IntM[0, 0], IntM[1, 1]])
    h_fov = math.degrees(math.atan(IntM[0, 2] / IntM[0, 0]) * 2)
    v_fov = math.degrees(math.atan(IntM[1, 2] / IntM[1, 1]) * 2)

    if out_size is not None: # the depth map is re-scaled #
        camera_intrinsics = np.zeros((3, 3))
        pixel_width, pixel_height = out_size[0], out_size[1]
        camera_intrinsics[2, 2] = 1.
        camera_intrinsics[0, 0] = IntM[0, 0] # (pixel_width/2.0)/math.tan(math.radians(h_fov/2.0))
        camera_intrinsics[0, 2] = (pixel_width - 1)/2.0
        camera_intrinsics[1, 1] = IntM[1, 1] # (pixel_height/2.0)/math.tan(math.radians(v_fov/2.0))
        camera_intrinsics[1, 2] = (pixel_height - 1) /2.0

        IntM = camera_intrinsics
        focal_length = pixel_width / width * focal_length
        width, height = pixel_width, pixel_height


    # In scanenet dataset, the depth is perperdicular z, not ray distance #
    pixel_to_ray_array = View.normalised_pixel_to_ray_array(width=width, height=height,
                                                            hfov=h_fov, vfov=v_fov, normalize_z=True)

    pixel_to_ray_array_2dM = np.reshape(np.transpose( pixel_to_ray_array, axes=[2, 0, 1]), [3, -1])
    pixel_to_ray_array_2dM = torch.from_numpy(pixel_to_ray_array_2dM.astype(np.float32))
    cam_intrinsic = {
            'hfov': h_fov, 'vfov': v_fov,
            'unit_ray_array': pixel_to_ray_array,
            'unit_ray_array_2D': pixel_to_ray_array_2dM,
            'focal_length': focal_length,
            'intrinsic_M': IntM}  
    return cam_intrinsic


def get_paths_as_dict(depth_dir, rgb_dir, split_txt=None, mode='train', cam_id=None, output_size=None, window_size=None):
    assert split_txt is not None, 'split_txt file is needed'

    videos = dict()

    scene_names = _read_split_file(split_txt)
    for idx, sname in enumerate(scene_names):
        name_contents = sname.split('_')
        date = name_contents[0] + '_' + name_contents[1] + '_' + name_contents[2]
        drive = name_contents[4]
        p_data = pykitti.raw(rgb_dir, date, drive)
        nimg = len(p_data)
        #
        # assume: the depth frames for one traj. is always nimg - 10 (ignoring the first and last 5 frames)
        p_data = pykitti.raw(rgb_dir, date, drive, frames=range(5, nimg - 5))
        nimg = len(p_data)
        if type(cam_id) is int:
            cam_ids = [cam_id]
        elif cam_id is None:
            cam_ids = [2]
        else:
            cam_ids = cam_id
        for idc in cam_ids:
            cam_intrinsics = _read_IntM_from_pdata(p_data, out_size=output_size, cam_id=idc)
            K = cam_intrinsics['intrinsic_M']
            img_paths = []
            gt_dmap_paths = []
            sdmap_paths = []
            poses = []

            M_imu2cam = getattr(p_data.calib, 'T_cam%d_imu' % idc)
            all_imgs = getattr(p_data, 'cam%d_files' % idc)
            for i_img in range(nimg):
                img_path = all_imgs[i_img]
                imgname = img_path.split('/')[-1]
                gt_file = '%s/data_depth_annotated/%s/%s/proj_depth/groundtruth/image_%.2d/%s' \
                          % (depth_dir, mode, sname, idc, imgname)
                if os.path.exists(gt_file):
                    img_paths.append(img_path)
                    gt_dmap_paths.append(gt_file)
                    input_file = '%s/data_depth_velodyne/%s/%s/proj_depth/velodyne_raw/image_%.2d/%s' \
                                 % (depth_dir, mode, sname, idc, imgname)
                    sdmap_paths.append(input_file)
                    pose_imu = p_data.oxts[i_img].T_w_imu
                    extM = np.matmul(M_imu2cam, np.linalg.inv(pose_imu))
                    poses.append(extM)
            video_name = sname + '_cam%d' % idc
            videos[video_name] = dict(intrinsic_matrix=K, cam_poses=poses, img_paths=img_paths,
                                gt_paths=gt_dmap_paths, sdmap_paths=sdmap_paths)

    return videos


def get_selected_valid_videos(root_dir, rgb_dir, full_img_size=(370, 1240)):
    image_dir = os.path.join(root_dir, 'image')
    videos = {} #{'image': dict(), 'ground_truth': dict(), 'velodyne_raw': dict()}
    for iii, f in enumerate(os.listdir(image_dir)):
        if os.path.exists(os.path.join(image_dir, f)):
            parser = f.split('_')
            date = '_'.join(parser[:3])
            drive = parser[4]
            cam_id = parser[-1][:2]
            name_video = 'x'.join([date, drive, cam_id])
            if name_video in videos:
                videos[name_video]['img_paths'].append(os.path.join(image_dir, f))
            else:
                videos[name_video] = {'img_paths': [os.path.join(image_dir, f)], 'gt_paths': [], 'sdmap_paths': [],
                                      'cam_poses': [], 'intrinsic_matrix': None}

    for name_video in videos:
        videos[name_video]['img_paths'] = sorted(videos[name_video]['img_paths'])

    for name_video in videos:
        date, drive, cam_id = tuple(name_video.split('x'))
        cam_id = int(cam_id)
        p_data = pykitti.raw(rgb_dir, date, drive)
        K = _read_IntM_from_pdata(p_data, (full_img_size[1], full_img_size[0]), cam_id=cam_id)['intrinsic_M']
        videos[name_video]['intrinsic_matrix'] = K

        img_name2id = dict()
        for idx in range(len(p_data)):
            all_cam_files = getattr(p_data, 'cam%d_files' % cam_id)
            img_name = all_cam_files[idx].split('/')[-1].split('.')[0]
            img_name2id[img_name] = idx

        M_imu2cam = getattr(p_data.calib, 'T_cam%d_imu' % cam_id) #p_data.calib.T_cam2_imu if cam_id == 2 else p_data.calib.T_cam3_imu

        for path_img in videos[name_video]['img_paths']:
            gt_path = path_img[:-12].replace('image', 'groundtruth_depth') + path_img[-12:]
            sdmap_path = path_img[:-12].replace('image', 'velodyne_raw') + path_img[-12:]

            img_name = path_img.split('/')[-1].split('_')[-3]
            idx = img_name2id[img_name]
            pose_imu = p_data.oxts[idx].T_w_imu
            extM = np.matmul(M_imu2cam, np.linalg.inv(pose_imu))
            videos[name_video]['gt_paths'].append(gt_path)
            videos[name_video]['sdmap_paths'].append(sdmap_path)
            videos[name_video]['cam_poses'].append(extM)

    return videos

def get_rgb_near(raw_loader, id_ref, cam_id=2):
    # count = 0
    max_frame_diff = 3
    candidates = [
        (i - max_frame_diff + id_ref) for i in range(max_frame_diff * 2 + 1)
        if i - max_frame_diff != 0
    ]
    candidates = [idx for idx in candidates if (idx >= 0) and (idx < len(raw_loader))]
    id_near = np.random.choice(candidates)
    img_near = raw_loader.get_cam2(id_near) if cam_id == 2 else raw_loader.get_cam3(id_near)
    # print(id_ref, id_near)
    return img_near, id_near


class KittiDataset(data.Dataset):

    def __init__(self, mode, depth_dir, rgb_dir, data_file=None, img_size=(352, 1216), patch_size=None, batch_size=1,
                 scale_factor=1, depth_max=100., cam_id=None, img_resize=(352, 1216)):
        self.cam_id = cam_id
        self.mode = mode
        self.depth_dir = depth_dir
        self.rgb_dir = rgb_dir
        self.depth_max = depth_max
        # self.img_size = img_size # the raw input image size (used for resizing the input images)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.data_file = data_file
        self.batch_size = batch_size
        self.scale = 1.

        # if self.mode == 'selval':
        #     full_img_size = (352, 1216)
        # else:
        #     full_img_size = (370, 1240)

        if self.mode == 'train':
            self.all_paths = get_paths_as_dict(depth_dir, rgb_dir, split_txt=data_file, mode='train',
                                               cam_id=cam_id, output_size=(1216, 352)#(full_img_size[1], full_img_size[0])
                                               )
        elif self.mode == 'val':
            self.all_paths = get_paths_as_dict(depth_dir, rgb_dir, split_txt=data_file, mode='val',
                                               cam_id=cam_id, output_size=(1216, 352) #(full_img_size[1], full_img_size[0])
                                               )
            # self.batch_size = 1
        else:
            self.all_paths = get_selected_valid_videos(data_file, rgb_dir,
                                                       full_img_size=img_size #full_img_size
                                                       )
        self.generate_img_index = []
        for name in self.all_paths:
            for idx in range(len(self.all_paths[name]['img_paths'])):
                self.generate_img_index.append((name, idx))

        print("Number samples of %s dataset: " % self.mode, len(self.generate_img_index))

        self.rescale = float(img_resize[0] / img_size[0])
        full_img_size = (int(img_size[0] * self.rescale), int(img_size[1] * self.rescale))
        self.img_size = img_resize
        self.top = (full_img_size[0] - self.img_size[0]) // 2
        self.left = (full_img_size[1] - self.img_size[1]) // 2
        if self.mode == 'train':
            # degree = np.random.uniform(-5.0, 5.0)
            self.transform_rgb = tfv_transform.Compose([
                tfv_transform.CenterCrop((352, 1216)),
                tfv_transform.Resize(self.img_size[0], interpolation=Image.NEAREST),
                Crop(self.left, self.img_size[1]+self.left, self.top, self.img_size[0]+self.top),
                                                        tfv_transform.ColorJitter(brightness=0.4, contrast=0.4,
                                                                                saturation=0.4),
                                                        tfv_transform.ToTensor(),
                                                        ])
            self.transform_depth = tfv_transform.Compose([
                tfv_transform.CenterCrop((352, 1216)),
                tfv_transform.Resize(self.img_size[0], interpolation=Image.NEAREST),
                Crop(self.left, self.img_size[1]+self.left, self.top, self.img_size[0]+self.top),
                                                          tfv_transform.ToTensor()])
        else:
            self.transform_rgb = tfv_transform.Compose([
                tfv_transform.CenterCrop((352, 1216)),
                tfv_transform.Resize(self.img_size[0], interpolation=Image.NEAREST),
                Crop(self.left, self.img_size[1]+self.left, self.top, self.img_size[0]+self.top),
                                                        tfv_transform.ToTensor(),
                                                        ])
            self.transform_depth = tfv_transform.Compose([
                tfv_transform.CenterCrop((352, 1216)),
                tfv_transform.Resize(self.img_size[0], interpolation=Image.NEAREST),
                Crop(self.left, self.img_size[1]+self.left, self.top, self.img_size[0]+self.top),
                                                          tfv_transform.ToTensor()])

    def __getitem__(self, indx):
        '''
        outputs:
        img, dmap, extM, scene_path , as entries in a dic.
        '''
        name, id_data = self.generate_img_index[indx]
        video_data = self.all_paths[name]
        dmap_path = video_data['sdmap_paths'][id_data]
        gt_path = video_data['gt_paths'][id_data]
        img_path = video_data['img_paths'][id_data]
        cam_pose = video_data['cam_poses'][id_data]
        K = np.array(video_data['intrinsic_matrix'])
        K = np.array([[K[0, 0]*self.rescale, 0, self.img_size[1] / 2.],
                      [0, K[1, 1]*self.rescale, self.img_size[0] / 2.],
                      [0, 0, 1]])

        # read rgb image #
        img = Image.open(img_path)
        img = self.transform_rgb(img)

        # read sparse depth map and GT depth map
        sdepth = Image.open(dmap_path)
        sdepth = self.transform_depth(sdepth)
        gt = Image.open(gt_path)
        gt = self.transform_depth(gt)
        # normalize
        sdepth = sdepth.float() / 256
        gt = gt.float() / 256
        if self.mode == 'train':
            sdepth[sdepth > self.depth_max] = self.depth_max
            gt[gt > self.depth_max] = self.depth_max
        if self.scale_factor == 0:
            max_depth = max(sdepth.max(), 1.)
            scale = 10. / max_depth
            sdepth *= scale
            gt *= scale
        else:
            sdepth /= self.scale_factor
            gt /= self.scale_factor
            scale = 1. / self.scale_factor

        return img, sdepth, torch.from_numpy(cam_pose).float(), torch.from_numpy(K).float(), \
               torch.tensor(scale, dtype=torch.float32), gt, gt > 0, True

    def __len__(self):
        return len(self.generate_img_index)

class KittiDatasetSeq(data.Dataset):

    def __init__(self, mode, depth_dir, rgb_dir, data_file=None, img_size=(352, 1216), patch_size=None, batch_size=1,
                 seq_size=2, skip_step=5, scale_factor=1, depth_max=100., cam_id=None, shuffle=False,
                 img_resize=(352, 1216)):
        self.cam_id = cam_id
        self.mode = mode
        self.depth_dir = depth_dir
        self.rgb_dir = rgb_dir
        self.depth_max = depth_max
        # self.img_size = img_size # the raw input image size (used for resizing the input images)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.data_file =data_file
        self.seq_size = seq_size
        self.skip_step = skip_step
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scale = 1.

        # if self.mode == 'selval':
        #     full_img_size = (352, 1216)
        # else:
        #     full_img_size = (370, 1240)

        if self.mode == 'train':
            self.all_paths = get_paths_as_dict(depth_dir, rgb_dir, split_txt=data_file, mode='train',
                                               cam_id=cam_id, output_size=(1216, 352)
                                               )
        elif self.mode == 'val':
            self.all_paths = get_paths_as_dict(depth_dir, rgb_dir, split_txt=data_file, mode='val',
                                               cam_id=cam_id, output_size=(1216, 352) #(full_img_size[1], full_img_size[0])
                                               )
            # self.batch_size = 1
        else:
            self.all_paths = get_selected_valid_videos(data_file, rgb_dir,
                                                       full_img_size=img_size #full_img_size
                                                       )
            # self.batch_size = 1

        self.generate_img_index = []
        self.list_begin = []
        self.list_crop_coords = []
        self.spliter = []
        # for name in self.all_paths.keys():
        #     num_imgs = len(self.all_paths[name]['img_paths'])
        #     if num_imgs < self.seq_size:
        #         self.seq_size = num_imgs

        print('Number of frames per video: ', self.seq_size)
        total_imgs = 0
        keys = sorted(list(self.all_paths.keys()))
        for name in keys:
            num_imgs = len(self.all_paths[name]['img_paths'])
            total_imgs += num_imgs
            # print(name, num_imgs, seq_size)
            if self.mode == 'train':
                indices = np.arange(num_imgs)
                for ptr in range(0, num_imgs, seq_size):
                    self.spliter.append((name, indices[ptr:(ptr+seq_size)]))
            else:
                self.spliter.append((name, np.arange(num_imgs)))

        self.generate_indices()

        self.rescale = float(img_resize[0] / img_size[0])
        full_img_size = (int(img_size[0] * self.rescale), int(img_size[1] * self.rescale))
        self.img_size = img_resize
        self.top = (full_img_size[0] - self.img_size[0]) // 2
        self.left = (full_img_size[1] - self.img_size[1]) // 2
        if self.mode == 'train':
            # degree = np.random.uniform(-5.0, 5.0)
            self.transform_rgb = tfv_transform.Compose([
                tfv_transform.CenterCrop((352, 1216)),
                tfv_transform.Resize(self.img_size[0], interpolation=Image.NEAREST),
                Crop(self.left, self.img_size[1]+self.left, self.top, self.img_size[0]+self.top),
                                                        # tfv_transform.ColorJitter(brightness=0.4, contrast=0.4,
                                                        #                         saturation=0.4),
                                                        #tfv_transform.RandomHorizontalFlip(),
                                                        # tfv_transform.CenterCrop(img_size),
                                                        tfv_transform.ToTensor(),
                                                        # tfv_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                        ])
            self.transform_depth = tfv_transform.Compose([
                tfv_transform.CenterCrop((352, 1216)),
                tfv_transform.Resize(self.img_size[0], interpolation=Image.NEAREST),
                Crop(self.left, self.img_size[1]+self.left, self.top, self.img_size[0]+self.top),
                                                          # tfv_transform.RandomHorizontalFlip(),
                                                          # tfv_transform.CenterCrop(img_size),
                                                          tfv_transform.ToTensor()])
        else:
            self.transform_rgb = tfv_transform.Compose([
                tfv_transform.CenterCrop((352, 1216)),
                tfv_transform.Resize(self.img_size[0], interpolation=Image.NEAREST),
                Crop(self.left, self.img_size[1]+self.left, self.top, self.img_size[0]+self.top),
                                                        # tfv_transform.CenterCrop(img_size),
                                                        tfv_transform.ToTensor(),
                                                        # tfv_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                        ])
            self.transform_depth = tfv_transform.Compose([
                tfv_transform.CenterCrop((352, 1216)),
                tfv_transform.Resize(self.img_size[0], interpolation=Image.NEAREST),
                Crop(self.left, self.img_size[1]+self.left, self.top, self.img_size[0]+self.top),
                                                          # tfv_transform.CenterCrop(img_size),
                                                          tfv_transform.ToTensor()])

    def generate_indices(self):
        self.generate_img_index = []
        self.list_begin = []
        self.list_crop_coords = []

        if self.shuffle:
            random.shuffle(self.spliter)
        if self.mode == 'train':
            if self.patch_size is not None:
                range_h = np.arange(0, self.img_size[0] - self.patch_size[0] + 1, 24)
                range_w = np.arange(0, self.img_size[1] - self.patch_size[1] + 1, 24)
            else:
                range_h, range_w = [0], [0]

            idx = self.batch_size - 1
            batch_ptrs = list(range(self.batch_size))
            id_ptrs = np.zeros(self.batch_size, dtype=np.uint8)
            batch_crop_coords = [(np.random.choice(range_h), np.random.choice(range_w)) for i in range(self.batch_size)]
            while idx < len(self.spliter):
                for i in range(len(batch_ptrs)):
                    if id_ptrs[i] == 0:
                        self.list_begin.append(True)
                    else:
                        self.list_begin.append(False)
                    name, id_data = self.spliter[batch_ptrs[i]]
                    self.generate_img_index.append((name, id_data[id_ptrs[i]]))
                    self.list_crop_coords.append(batch_crop_coords[i])
                    id_ptrs[i] += 1
                    if id_ptrs[i] >= len(id_data):
                        idx += 1
                        batch_ptrs[i] = idx
                        id_ptrs[i] = 0
                        batch_crop_coords[i] = (np.random.choice(range_h), np.random.choice(range_w))
                    if idx >= len(self.spliter):
                        if i < len(batch_ptrs)-1:
                            self.generate_img_index = self.generate_img_index[:-(i+1)]
                            self.list_begin = self.list_begin[:-(i+1)]
                            self.list_crop_coords = self.list_crop_coords[:-(i+1)]
                        break
        else:
            for ptr in range(len(self.spliter)):
                name, indices = self.spliter[ptr]
                for i, idx in enumerate(indices):
                    self.generate_img_index.append((name, idx))
                    if i == 0:
                        self.list_begin.append(True)
                    else:
                        self.list_begin.append(False)

        print("Number samples of %s dataset: " % self.mode, len(self.generate_img_index))

    def __getitem__(self, indx):
        '''
        outputs:
        img, dmap, extM, scene_path , as entries in a dic.
        '''
        name, id_data = self.generate_img_index[indx]
        video_data = self.all_paths[name]
        dmap_path = video_data['sdmap_paths'][id_data]
        gt_path = video_data['gt_paths'][id_data]
        img_path = video_data['img_paths'][id_data]
        cam_pose = video_data['cam_poses'][id_data]
        K = np.array(video_data['intrinsic_matrix'])
        K = np.array([[K[0, 0]*self.rescale, 0, self.img_size[1] / 2.],
                      [0, K[1, 1]*self.rescale, self.img_size[0] / 2.],
                      [0, 0, 1]])

        # read rgb image #
        img = Image.open(img_path)
        img = self.transform_rgb(img)

        # read sparse depth map and GT depth map
        sdepth = Image.open(dmap_path)
        sdepth = self.transform_depth(sdepth)
        gt = Image.open(gt_path)
        gt = self.transform_depth(gt)
        # normalize
        sdepth = sdepth.float() / 256
        gt = gt.float() / 256
        if self.mode == 'train':
            sdepth[sdepth > self.depth_max] = self.depth_max
            gt[gt > self.depth_max] = self.depth_max
        if self.scale_factor == 0:
            max_depth = max(sdepth.max(), 1.)
            scale = 10. / max_depth
            sdepth *= scale
            gt *= scale
        else:
            sdepth /= self.scale_factor
            gt /= self.scale_factor
            scale = 1. / self.scale_factor

        return img, sdepth, torch.from_numpy(cam_pose).float(), torch.from_numpy(K).float(), \
               torch.tensor(scale, dtype=torch.float32), gt, gt > 0, self.list_begin[indx]

    def __len__(self):
        return len(self.generate_img_index)


if __name__ == '__main__':
    # n_traj, p_data, sdmap_paths, \
    # gt_dmap_paths, poses, _ = get_paths(0, '/home/khangtg/Documents/lab/code/nconv/data/kitti_depth',
    #                                     '/home/khangtg/Documents/lab/code/nconv/data/kitti_raw',
    #                                     split_txt='training.txt')
    # dataset = KittiDataset(True, p_data, sdmap_paths, gt_dmap_paths, poses, img_size=(352, 1216))
    # K = dataset.cam_intrinsics['intrinsic_M']
    # sample = dataset[10]
    # img, img_near = sample['img'], sample['img_near']
    # E, gt_depth = sample['rel_extM'], sample['gt_depth']
    # warped_depths, warped_imgs = homo.warp_cfd(gt_depth.cuda(), img_near.cuda(), torch.from_numpy(K).unsqueeze(0).float().cuda(), E.cuda())
    # print(warped_depths.size(), warped_imgs.size())
    # img = img.squeeze(0).permute(1, 2, 0).numpy()
    # img = np.array(255 * img).astype(np.uint8)
    # warped_img = warped_imgs.cpu().squeeze(0).permute(1, 2, 0).numpy()
    # warped_img = np.array(255 * warped_img).astype(np.uint8)
    # # warped_img = warped_img / warped_img.max()
    # # print(warped_img.min(), warped_img.max())
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(20, 20))
    # ax1 = fig.add_subplot(3, 1, 1)
    # plt.imshow(img)
    # ax2 = fig.add_subplot(3, 1, 2)
    # plt.imshow(img_near.squeeze(0).permute(1, 2, 0).numpy())
    # ax3 = fig.add_subplot(3, 1, 3)
    # plt.imshow(warped_img)
    # plt.show()
    from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
    dataset = KittiDatasetSeq('val', '/home/khangtg/Documents/lab/code/data/kitti_depth',
                              '/home/khangtg/Documents/lab/code/data/kitti_raw',
                              '/home/khangtg/Documents/lab/corona-net/data_loader/small_training.txt',
                              img_size=(352, 1216), batch_size=4, seq_size=10, skip_step=1, cam_id=2, shuffle=True,
                              # img_resize=(272, 928)
                              )
    # dataset = KittiDatasetSeq('selval', '/home/khangtg/Documents/lab/code/data/kitti_depth',
    #                           '/home/khangtg/Documents/lab/code/data/kitti_raw',
    #                           data_file='/home/khangtg/Documents/lab/code/data/kitti_depth/val_selection_cropped',
    #                           img_size=(352, 1216), batch_size=4, seq_size=10, skip_step=1, cam_id=2, shuffle=True,
    #                           img_resize=(272, 928))
    print(len(dataset))
    for i in range(100):
        img, sdepth, E, K, coord, gt, mask, is_begin = dataset[i]
        print(img.size(), sdepth.size(), coord)
    # dataset.generate_indices()
    # print(len(dataset))
    # for i in range(201):
    #     print('Batch no: ', i)
    #     for j in range(4):
    #         print(dataset.generate_img_index[i*4+j], dataset.list_begin[i*4+j])
    # sampler = BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True)
    # dataloader = DataLoader(dataset, sampler=sampler)
    # for batch in dataloader:
    #     print(batch[0].size(), batch[1].size())
