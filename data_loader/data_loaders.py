import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from data_loader.kitti_dataset import KittiDataset, KittiDatasetSeq
from data_loader.aerial_dataset import AerialDatasetSeq
from data_loader.scene7_dataset import SevenSceneDatasetSeq
from warping import homography as homo

path_to_dir = os.path.abspath(os.path.dirname(__file__))
training_videos = os.path.join(path_to_dir, 'training.txt')
val_videos = os.path.join(path_to_dir, 'test.txt')

np.random.seed(1234)

class KittiLoader(DataLoader):

    def __init__(self, kitti_depth_dir, kitti_raw_dir, batch_size,
                 img_size=None, patch_size=None, num_workers=1, mode='train', scale_factor=1,
                 depth_max=100., seq_size=None, cam_ids=None, img_resize=(352, 1216)):
        if mode == 'train':
            file_videos = training_videos
        elif mode == 'val':
            file_videos = val_videos
        else:
            file_videos = os.path.join(kitti_depth_dir, 'val_selection_cropped')
        self.kitti_dataset = KittiDataset(mode, kitti_depth_dir, kitti_raw_dir, file_videos, img_size=img_size,
                                          patch_size=patch_size, batch_size=batch_size, scale_factor=scale_factor,
                                          depth_max=depth_max, cam_id=cam_ids, img_resize=img_resize)
        super().__init__(self.kitti_dataset, batch_size=batch_size, shuffle=True,
                         num_workers=num_workers, pin_memory=True)
        self.n_samples = len(self.kitti_dataset)
        self.device = None

    def shuffle_and_crop(self):
        pass

    def set_device(self, device):
        self.device = device


class KittiLoaderv2(DataLoader):

    def __init__(self, kitti_depth_dir, kitti_raw_dir, batch_size, shuffle=True,
                 img_size=None, patch_size=None, num_workers=1, mode='train', scale_factor=1,
                 depth_max=100., seq_size=20, skip_step=1, cam_ids=None, img_resize=(352, 1216)):
        if mode == 'train':
            file_videos = training_videos
        elif mode == 'val':
            file_videos = val_videos
        else:
            file_videos = os.path.join(kitti_depth_dir, 'val_selection_cropped')
        self.kitti_dataset = KittiDatasetSeq(mode, kitti_depth_dir, kitti_raw_dir, file_videos, img_size=img_size,
                                        patch_size=patch_size, batch_size=batch_size, seq_size=seq_size,
                                        skip_step=skip_step, scale_factor=scale_factor, depth_max=depth_max,
                                        shuffle=shuffle, cam_id=cam_ids, img_resize=img_resize)
        sampler = SequentialSampler(self.kitti_dataset)
        super().__init__(self.kitti_dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                         num_workers=num_workers, pin_memory=True)

        self.n_samples = len(self.kitti_dataset)
        self.device = None

    def shuffle_and_crop(self):
        self.kitti_dataset.generate_indices()

    def set_device(self, device):
        self.device = device

    def get_num_samples(self):
        return len(self.kitti_dataset)


class VISIMLoader(DataLoader):

    def __init__(self, root_dir, mode='train', shuffle=False, img_size=(480, 752), batch_size=1,
                 num_workers=-1, seq_size=100, patch_size=None, scale_factor=1, depth_max=-1, depth_min=0,
                 img_resize=(480, 752), sparsifier='fix', num_samples=1000):
        self.visim_dataset = AerialDatasetSeq(mode, root_dir, img_size=img_size, patch_size=patch_size,
                                              batch_size=batch_size, seq_size=seq_size, shuffle=shuffle,
                                              scale_factor=scale_factor, depth_max=depth_max, depth_min=depth_min,
                                              img_resize=img_resize, sparsifier=sparsifier, num_samples=num_samples)
        sampler = SequentialSampler(self.visim_dataset)
        super().__init__(self.visim_dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                         num_workers=num_workers, pin_memory=True)

        self.n_samples = len(self.visim_dataset)
        self.device = None

    def shuffle_and_crop(self):
        self.visim_dataset.generate_indices()

    def set_device(self, device):
        self.device = device

    def get_num_samples(self):
        return len(self.visim_dataset)


class SevenSceneLoader(DataLoader):

    def __init__(self, root_dir, mode='train', shuffle=False, img_size=(480, 640), batch_size=1,
                 num_workers=-1, seq_size=100, patch_size=None, scale_factor=1, depth_max=65., depth_min=0,
                 img_resize=(480, 640)):
        self.scene7_dataset = SevenSceneDatasetSeq(mode, root_dir, img_size=img_size, patch_size=patch_size,
                                              batch_size=batch_size, seq_size=seq_size, shuffle=shuffle,
                                              scale_factor=scale_factor, depth_max=depth_max,
                                              depth_min=depth_min, img_resize=img_resize)
        sampler = SequentialSampler(self.scene7_dataset)
        super().__init__(self.scene7_dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                         num_workers=num_workers, pin_memory=True)

        self.n_samples = len(self.scene7_dataset)
        self.device = None

    def shuffle_and_crop(self):
        self.scene7_dataset.generate_indices()

    def set_device(self, device):
        self.device = device


if __name__ == '__main__':
    # dataloader = KittiDepthDataloader('/home/khangtg/Documents/lab/code/data/kitti_depth',
    #                                   '/home/khangtg/Documents/lab/code/data/kitti_raw',
    #                                   4, img_size=(352, 1216), shuffle=False, num_workers=4)
    dataloader = KittiLoaderv2('/home/khangtg/Documents/lab/code/data/kitti_dataset/kitti_depth',
                               '/home/khangtg/Documents/lab/code/data/kitti_dataset/kitti_raw',
                               4, img_size=(352, 1216), shuffle=True, num_workers=4, mode='val',
                               seq_size=10, cam_ids=[2, 3]
                               )

    # dataloader = KittiLoader('/home/khangtg/Documents/lab/code/data/kitti_depth',
    #                            '/home/khangtg/Documents/lab/code/data/kitti_raw',
    #                            4, img_size=(352, 1216), num_workers=4, mode='val',
    #                            cam_ids=[2, 3],
    #                            )

    # dataloader = VISIMLoader('/home/khangtg/Documents/lab/mvs/dataset/mvs/test', mode='train',
    #                          num_workers=4, batch_size=4,
    #                          seq_size=100,
    #                          shuffle=False)
    # dataloader = SevenSceneLoader('/home/khangtg/Documents/lab/mvs/dataset/mvs/7scene', num_workers=4, batch_size=4,
    #                                 seq_size=100,
    #                                 shuffle=False)
    # print(len(dataloader))
    # dataloader.shuffle_and_crop()
    print(hasattr(dataloader, 'n_samples'))
    # dataloader.set_device(torch.device('cuda:0'))
    # print(len(dataloader.dataset_batch))
    prev_batch = None
    prev_pose = None
    cur_batch = None
    prev_imgs, prev_gt = torch.zeros(4, 3, 480, 640), torch.zeros(4, 1, 480, 640)
    for i, data in enumerate(dataloader):
        # if i%100 == 0:
        # data, target = batch
        # for elem in data:
        #     print(elem.size())
        # for elem in target:
        #     print(elem.size())
        # print("Batch no: %d, beginning of video: " % i)
        # print(batch[-1], len(batch[-1].nonzero()))
        # for elem in batch:
        #     print(elem.size())
        # if i > 30:
        #     break
        # print(i, batch[0][-1], batch[1][0].size(), 'num processed frames: %d' % dataloader.num_processed_frames)
        imgs, sdmaps, E, K, crop_at, gt, mask, is_begin_video = data
        print(imgs.size())
        print(E.size())
        print(K.size())
        print(len(crop_at))
        print(gt.size())
        print(mask.size())
        print(is_begin_video.size())
        # if prev_batch is None:
        #     prev_pose = E
        # else:
        #     prev_pose[is_begin_video] = E[is_begin_video]
        # print(E.size(), K.size())
        # if prev_pose is not None:
        # rel_E = E.matmul(torch.inverse(prev_pose))
        # else:
        #     rel_E = prev_pose
        batch = (imgs, sdmaps, E, K, crop_at), (gt, mask)
        if i == 0:
            prev_batch = batch #(prev_imgs, sdmaps, E, K, crop_at), (prev_gt, mask) # batch
            # name, id_data = dataloader.visim_dataset.generate_img_index[i]
            # video = dataloader.visim_dataset.all_paths[name]
            # img_path = video['img_paths'][id_data]
            # print(img_path)
        elif i == 1:
            cur_batch = batch
            # name, id_data = dataloader.visim_dataset.generate_img_index[i]
            # video = dataloader.visim_dataset.all_paths[name]
            # img_path = video['img_paths'][id_data]
            # print(img_path)
            break
        # if i > 60 and (i < 63):
        # homo.test_warping(batch, prev_batch)
        # prev_batch = batch
        # prev_pose = E

    # print(len(dataloader))
    (imgs_prev, sdmaps_prev, E_prev, K_prev, crop_at_prev), (gt_prev, mask_prev) = prev_batch
    (imgs_cur, sdmaps_cur, E_cur, K_cur, crop_at_cur), (gt_cur, mask_cur) = cur_batch
    # E_prev[:, [1, 2], :] = - E_prev[:, [1, 2], :] #torch.inverse(E_prev)
    # E_cur[:, [1, 2], :] = - E_cur[:, [1, 2], :] # torch.inverse(E_cur)
    # rel_E = E_cur.matmul(torch.inverse(E_prev))
    # cur_batch = (imgs_cur, sdmaps_cur, rel_E, K_cur, crop_at_cur), (gt_cur, mask_cur)
    # homo.test_warping(cur_batch, prev_batch)
    homo.test_warping(cur_batch, prev_batch)


