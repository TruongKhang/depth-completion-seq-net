import os
import os.path
import torch.utils.data as data
from torchvision import transforms as tfv_transform
import torch
import numpy as np
import random
from scipy.spatial.transform import Rotation
from PIL import Image

from data_loader.transformation import Crop
from data_loader.dense_to_sparse import UniformSampling, SimulatedStereo

epsilon = np.finfo(float).eps
np.random.seed(1234)
random.seed(1995)


def get_all_paths(root_dir, mode='train'):
    all_datasets = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    all_paths = dict()
    for d in all_datasets:
        dataset_dir = os.path.join(root_dir, d)
        if mode == 'train':
            data_names = os.path.join(dataset_dir, 'TrainSplit.txt')
        else:
            data_names = os.path.join(dataset_dir, 'TestSplit.txt')
        with open(data_names) as fp:
            for line in fp:
                idx = int(line.strip().replace('sequence', ''))
                path_scene = os.path.join(dataset_dir, 'seq-%.2d' % idx)
                img_paths = [os.path.join(path_scene, f) for f in os.listdir(path_scene) if 'color' in f]
                img_paths = sorted(img_paths)
                gt_paths = [f.replace('color', 'depth') for f in img_paths]
                sdepth_paths = [f.replace('color', 'sparse') for f in img_paths]
                poses, masks = [], []
                for f in img_paths:
                    file_pose = f.replace('color.png', 'pose.txt')
                    poses.append(np.linalg.inv(np.loadtxt(file_pose)))
                all_paths['%s_seq-%.2d' % (d, idx)] = dict(img_paths=img_paths, gt_paths=gt_paths,
                                                           sdmap_paths=sdepth_paths, cam_poses=poses, masks=masks)
    return all_paths


def create_sparse_depth(root_dir, out_dir, num_samples=5000):
    import cv2
    all_datasets = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for name in all_datasets:
        path_dataset = os.path.join(root_dir, name)
        sub_folders = [f for f in os.listdir(path_dataset) if 'seq' in f]
        for seq in sub_folders:
            path_seq = os.path.join(path_dataset, seq)
            print(path_seq)
            gt_files = [f for f in os.listdir(path_seq) if 'depth' in f]
            for file in gt_files:
                gt_path = os.path.join(path_seq, file)
                gt_depth = np.array(Image.open(gt_path))
            # if sparsifier_type == UniformSampling.name:  # uar
                sparsifier = UniformSampling(num_samples=num_samples)
            # elif sparsifier_type == SimulatedStereo.name:  # sim_stereo
            #     sparsifier = SimulatedStereo(num_samples=num_samples, max_depth=depth_max)
                mask_keep = sparsifier.dense_to_sparse(None, gt_depth)
                sparse_depth = np.zeros(gt_depth.shape)
                sparse_depth[mask_keep] = gt_depth[mask_keep]
                out_folder = path_seq #os.path.join(out_dir, name)
                # if not os.path.exists(out_folder):
                #     os.makedirs(out_folder)
                sparse_path = os.path.join(out_folder, file.replace('depth', 'sparse'))
                cv2.imwrite(sparse_path, sparse_depth.astype(np.uint16))
            # img = np.array(Image.open(sparse_path))


class SevenSceneDatasetSeq(data.Dataset):

    def __init__(self, mode, root_dir, img_size=(480, 640), patch_size=None, batch_size=1,
                 seq_size=2, skip_step=5, scale_factor=1, shuffle=False, depth_max=100, depth_min=1,
                 img_resize=(480, 640)):
        self.mode = mode
        self.root_dir = root_dir
        # self.img_size = img_size # the raw input image size (used for resizing the input images)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.seq_size = seq_size
        self.skip_step = skip_step
        self.batch_size = batch_size
        self.shuffle = shuffle
        # full_img_size = self.img_size
        self.depth_max = depth_max if depth_max >= 0.0 else np.inf
        self.depth_min = depth_min
        # self.img_resize = img_resize

        self.all_paths = get_all_paths(root_dir, mode)
        rescale = float(img_resize[0] / img_size[0])
        full_img_size = (img_size[0]*rescale, img_size[1]*rescale)
        self.img_size = img_resize
        self.intrinsic_matrix = np.array([[585.0*rescale, 0, (img_resize[1] - 1)/2.],
                                          [0, 585.0*rescale, (img_resize[0] -1 )/2.],
                                          [0, 0, 1.0]])

        self.generate_img_index = []
        self.list_begin = []
        self.list_crop_coords = []
        self.spliter = []

        print('Number of videos: ', len(self.all_paths))
        print('Number of frames per video: ', self.seq_size)
        total_imgs = 0
        keys = sorted(list(self.all_paths.keys()))
        for name in keys:
            num_imgs = len(self.all_paths[name]['img_paths'])
            total_imgs += num_imgs
            # print(name, num_imgs, seq_size)
            indices = np.arange(num_imgs)
            for ptr in range(0, num_imgs, seq_size):
                self.spliter.append((name, indices[ptr:(ptr+seq_size)]))
            # if self.is_training:
            #     self.spliter[-1] = name, indices[-seq_size:]
        self.generate_indices()

        print("Number samples of %s dataset: " % self.mode, len(self.generate_img_index))
        self.top = (full_img_size[0] - self.img_size[0]) // 2
        self.left = (full_img_size[1] - self.img_size[1]) // 2
        if self.mode == 'train':
            self.transform_rgb = tfv_transform.Compose([
                                                        tfv_transform.Resize(self.img_size[0], interpolation=Image.NEAREST),
                                                        Crop(self.left, self.img_size[1]+self.left,
                                                             self.top, self.img_size[0]+self.top),
                                                        tfv_transform.ColorJitter(brightness=0.4, contrast=0.4,
                                                                                saturation=0.4),
                                                        tfv_transform.ToTensor(),
                                                        # tfv_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                        ])
            self.transform_depth = tfv_transform.Compose([
                tfv_transform.Resize(self.img_size[0], interpolation=Image.NEAREST),
                                                        Crop(self.left, self.img_size[1]+self.left,
                                                             self.top, self.img_size[0]+self.top),
                                                          tfv_transform.ToTensor()])
        else:
            self.transform_rgb = tfv_transform.Compose([
                tfv_transform.Resize(self.img_size[0], interpolation=Image.NEAREST),
                                                        Crop(self.left, self.img_size[1]+self.left,
                                                             self.top, self.img_size[0]+self.top),
                                                        tfv_transform.ToTensor(),
                                                        # tfv_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                        ])
            self.transform_depth = tfv_transform.Compose([
                tfv_transform.Resize(self.img_size[0], interpolation=Image.NEAREST),
                                                        Crop(self.left, self.img_size[1]+self.left,
                                                             self.top, self.img_size[0]+self.top),
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

        K = self.intrinsic_matrix

        # read rgb image #
        img = Image.open(img_path)

        # read sparse depth map and GT depth map
        gt = Image.open(gt_path)

        # if self.sparsifier is None:
        sdepth = Image.open(dmap_path)
        # else:
        #     sdepth = self.create_sparse_depth(np.array(img), np.array(gt))
        #     sdepth = Image.fromarray(sdepth)

        img = self.transform_rgb(img)
        gt = self.transform_depth(gt)
        sdepth = self.transform_depth(sdepth)

        # normalize
        sdepth = sdepth.float() / 1000
        gt = gt.float() / 1000
        #print(sdepth.max(), gt.max())
        # if self.mode == 'train':
        #     sdepth[sdepth > self.depth_max] = self.depth_max
        #     gt[gt > self.depth_max] = self.depth_max
        sdepth[sdepth > self.depth_max] = 0
        gt[gt > self.depth_max] = 0
        if len(video_data['masks']) > 0:
            path_mask = video_data['masks'][id_data]
            mask = Image.open(path_mask)
            mask = self.transform_depth(mask)
            mask = mask > 0
        else:
            mask = (gt > self.depth_min) & (gt < self.depth_max)
        if self.scale_factor == 0:
            # max_depth = max(sdepth.max(), 1.)
            max_depth = max(sdepth[mask].max(), 1.)
            scale = 10. / max_depth
            sdepth *= scale
            gt *= scale
        else:
            sdepth /= self.scale_factor
            gt /= self.scale_factor
            scale = 1. / self.scale_factor
        # final_coord_top, final_coord_left = self.top, self.left
        # if self.patch_size is not None:
        #     final_coord_top = self.top + self.list_crop_coords[indx][0]
        #     final_coord_left = self.left + self.list_crop_coords[indx][1]
        #     dl = self.list_crop_coords[indx][1]
        #     dr = self.img_size[1] - self.patch_size[1] - dl
        #     dt = self.list_crop_coords[indx][0]
        #     db = self.img_size[0] - self.patch_size[0] - dt
        #     img = F.pad(img, [-dl, -dr, -dt, -db])
        #     sdepth = F.pad(sdepth, [-dl, -dr, -dt, -db])
        #     gt = F.pad(gt, [-dl, -dr, -dt, -db])

        return img, sdepth, torch.from_numpy(cam_pose).float(), torch.from_numpy(K).float(), \
               torch.tensor(scale, dtype=torch.float32), gt, mask, self.list_begin[indx]

    def __len__(self):
        return len(self.generate_img_index)


if __name__ == '__main__':
    dataset = SevenSceneDatasetSeq('val', '/home/khangtg/Documents/lab/mvs/dataset/mvs/7scene',
                               batch_size=1, seq_size=100, img_resize=(480, 640))
    print(len(dataset))
    for i in range(10):
        data = dataset[i]
        img, sdepth, pose, K, scale, gt, mask, list_begin = data
        print('Image: ', img.size())
        print('Sparse depth: ', sdepth.size())
        print('camera pose: ', pose)
        print('intrinsic: ', K)
        print('Ground truth: ', gt.size())
        print('Mask: ', mask.size())
        print('List begin: ', list_begin)
    # create_sparse_depth('/home/khangtg/Documents/lab/mvs/dataset/mvs/7scene', None, num_samples=10000)
    # create_sparse_depth('/home/khangtg/Documents/lab/mvs/dataset/mvs/aerial_dataset',
    #                     '/home/khangtg/Documents/lab/mvs/dataset/mvs/sparse_depths', mode='val', num_samples=10000)
    # create_sparse_depth('/home/khangtg/Documents/lab/mvs/dataset/mvs/dataset',
    #                     '/home/khangtg/Documents/lab/mvs/dataset/mvs/sparse_depths', mode='train', num_samples=10000)
    # create_sparse_depth('/home/khangtg/Documents/lab/mvs/dataset/mvs/dataset',
    #                     '/home/khangtg/Documents/lab/mvs/dataset/mvs/sparse_depths', mode='val', num_samples=10000)
