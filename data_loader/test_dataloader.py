import os, copy
import pykitti
import numpy as np
from torchvision import transforms as tfv_transform
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
import glob

from data_loader.kitti_dataset import _read_IntM_from_pdata
from data_loader.transformation import Crop, RandomCrop
from warping import homography as homo


class TestDataLoader():
    def __init__(self, root_dir, path_to_rgb, img_size=None, shuffle=False, batch_size=1,
                 patch_size=None, num_workers=1, thresh_gt_depth=0):
        self.path_images, self.path_gt_depths, self.path_sdepths = get_videos(root_dir)
        self.path_to_rgb = path_to_rgb
        self.img_size = img_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_worker = num_workers
        self.thresh_gt_depth = thresh_gt_depth

        self.intrinsic_data, self.extrinsic_data = self.get_cam_calib()

        self.transform_rgb = tfv_transform.Compose([Crop(10, 1210, 130, 370),
                                                    tfv_transform.CenterCrop(img_size),
                                                    tfv_transform.ToTensor(),
                                                    tfv_transform.Normalize((0.485, 0.456, 0.406),
                                                                            (0.229, 0.224, 0.225))])
        self.transform_depth = tfv_transform.Compose([Crop(10, 1210, 130, 370),
                                                      tfv_transform.CenterCrop(img_size),
                                                      tfv_transform.ToTensor()])
        self.num_videos = len(self.path_images)
        self.idx_videos = np.arange(self.num_videos)
        self.video_names = list(self.path_images.keys())
        self.ptr_frame = 0

    def get_cam_calib(self):
        intrinsics, extrinsics = dict(), dict()
        for name in self.path_images.keys():
            date, drive, cam_id = tuple(name.split('x'))
            cam_id = int(cam_id)
            p_data = pykitti.raw(self.path_to_rgb, date, drive)
            K = _read_IntM_from_pdata(p_data, (370, 1240), cam_id=cam_id)['intrinsic_M']
            intrinsics[name] = K

            img_name2id = dict()
            for idx in range(len(p_data)):
                if cam_id == 2:
                    img_name = p_data.cam2_files[idx].split('/')[-1].split('.')[0]
                elif cam_id == 3:
                    img_name = p_data.cam3_files[idx].split('/')[-1].split('.')[0]
                img_name2id[img_name] = idx
            intrinsics[name], extrinsics[name] = list(), list()
            M_imu2cam = p_data.calib.T_cam2_imu if cam_id == 2 else p_data.calib.T_cam3_imu
            for path_img in self.path_images[name]:
                img_name = path_img.split('/')[-1].split('_')[-3]
                idx = img_name2id[img_name]
                pose_imu = p_data.oxts[idx].T_w_imu
                extM = np.matmul(M_imu2cam, np.linalg.inv(pose_imu))
                extrinsics[name].append(extM)
        return intrinsics, extrinsics

    def initialize(self):
        if self.shuffle:
            np.random.seed(0)
            np.random.shuffle(self.idx_videos)
        self.id_batch = 0
        self.num_processed_videos = 0
        self.num_processed_frames = 0
        self.dataset_batch = None
        self.ptrs = None
        self.new_added_videos = None
        self.new_batch = None
        self.prev_poses = None
        self.init_batch()

    def init_batch(self):
        end = self.id_batch+self.batch_size
        end = self.num_videos if end > self.num_videos else end
        self.dataset_batch = list()
        self.ptrs = list()
        self.new_added_videos = list()
        self.prev_poses = []
        for i in range(self.id_batch, end):
            self.dataset_batch.append(self.video_names[self.idx_videos[i]])
            self.ptrs.append(0)
            self.new_added_videos.append(True)
            self.prev_poses.append(None)
        self.new_added_videos = np.array(self.new_added_videos)
        self.id_batch = end-1

    def load_new_video_to_batch(self, position):
        self.dataset_batch[position] = self.video_names[self.idx_videos[self.id_batch]]
        self.ptrs[position] = 0
        self.new_added_videos[position] = True
        self.prev_poses[position] = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.id_batch < self.num_videos:
            for i in range(len(self.dataset_batch)):
                name_video = self.dataset_batch[i]
                num_frames = len(self.path_images[name_video])
                if self.ptrs[i] >= num_frames:
                    self.num_processed_videos += 1
                    self.id_batch += 1
                    if self.id_batch < self.num_videos:
                        self.load_new_video_to_batch(i)
                    else:
                        raise StopIteration

            indices = np.arange(len(self.dataset_batch))
            arr_indices = np.array_split(indices, self.num_worker)
            results = list()
            with ThreadPoolExecutor(max_workers=self.num_worker) as executor:
                for sub_indices in arr_indices:
                    if len(sub_indices) > 0:
                        t = executor.submit(self.load_sample, sub_indices)
                        results.append(t.result())

            samples = tuple(zip(*results))
            inputs, gt_depths = (), torch.cat(samples[-1])
            for i in range(len(samples)-1):
                inputs += (torch.cat(samples[i]),)
            return inputs, (gt_depths, gt_depths > self.thresh_gt_depth)
        else:
            raise StopIteration

    def load_sample(self, list_ids):
        imgs, sdepths, cfds, Es, Ks, crops, gts = [], [], [], [], [], [], []
        for i in list_ids:
            name_video = self.dataset_batch[i]
            crop_at = torch.zeros(2)
            crop_at[0] = 130 + (240 - self.img_size[0]) // 2
            crop_at[1] = (1200 - self.img_size[1]) // 2

            K = self.intrinsic_data[name_video]
            K = torch.from_numpy(K).unsqueeze(0).float()
            Rt_cur = self.extrinsic_data[name_video][self.ptrs[i]]
            Rt_prev2cur = torch.eye(4).unsqueeze(0)
            if self.prev_poses[i] is not None:
                Rt_prev2cur = homo.get_rel_extrinsicM(self.prev_poses[i], Rt_cur)
                Rt_prev2cur = torch.from_numpy(Rt_prev2cur).unsqueeze(0).float()

            self.ptrs[i] += 1
            self.prev_poses[i] = Rt_cur
            self.num_processed_frames += 1

            path_img = self.path_images[name_video][self.ptrs[i]]
            img = Image.open(path_img)
            img = self.transform_rgb(img)

            path_sdepth = self.path_sdepths[name_video][self.ptrs[i]]
            sdepth = Image.open(path_sdepth)
            sdepth = self.transform_depth(sdepth)

            path_gt_depth = self.path_gt_depths[name_video][self.ptrs[i]]
            gt = Image.open(path_gt_depth)
            gt = self.transform_depth(gt)

            sdepth = sdepth.type(torch.FloatTensor) / 256
            gt = gt.type(torch.FloatTensor) / 256
            c = (sdepth > 0)

            imgs.append(img.unsqueeze(0).float())
            sdepths.append(sdepth.unsqueeze(0).float())
            cfds.append(c.unsqueeze(0).float())
            Es.append(Rt_prev2cur)
            Ks.append(K)
            crops.append(crop_at.unsqueeze(0))
            gts.append(gt.unsqueeze(0).float())

        return torch.cat(imgs), torch.cat(sdepths), torch.cat(cfds), torch.cat(Es), torch.cat(Ks), torch.cat(crops), torch.cat(gts)

    def set_device(self, device):
        self.device = device

    def __len__(self):
        num_images = 0
        for name in self.video_names:
            num_images += len(self.path_images[name])
        return num_images


if __name__ == '__main__':
    videos = get_videos('/home/khangtg/Documents/lab/code/data/kitti_depth/val_selection_cropped',
                        '/home/khangtg/Documents/lab/code/data/kitti_raw')
    total_samples = 0
    for name in videos.keys():
        total_samples += len(videos[name]['img_paths'])
        print(videos[name]['img_paths'][0])
        print(videos[name]['sdmap_paths'][0])
        print(videos[name]['gt_paths'][0])
        print(videos[name]['cam_poses'][0])
        print(videos[name]['intrinsic_matrix'])
    print("Number of test cases: ", total_samples)
    # test_loader = TestDataLoader('/home/khangtg/Documents/lab/code/data/kitti_depth/val_selection_cropped',
    #                              '/home/khangtg/Documents/lab/code/data/kitti_raw', img_size=(224, 912))
    # test_loader.initialize()
    # test_loader.set_device(torch.device('cuda:0'))
    # # print(len(dataloader.dataset_batch))
    # prev_batch = None
    # for i, batch in enumerate(test_loader):
    #     # if i%100 == 0:
    #     data, target = batch
    #     for elem in data:
    #         print(elem.size())
    #     for elem in target:
    #         print(elem.size())
    #     if i == 0:
    #         break
        # print(i, batch[0][-1], batch[1][0].size(), 'num processed frames: %d' % dataloader.num_processed_frames)
        # homo.test_warping(batch, prev_batch)
        # prev_batch = batch
        # for idx, is_new in enumerate(dataloader.new_added_videos):
        #     if is_new:
        #         dataloader.new_added_videos[idx] = False
    # print(test_loader.num_videos)