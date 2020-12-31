import os
import os.path
import torch.utils.data as data
from torchvision import transforms as tfv_transform
import torch
import numpy as np
import random
from scipy.spatial.transform import Rotation
from PIL import Image
import cv2

from data_loader.transformation import Crop
from data_loader.dense_to_sparse import UniformSampling, SimulatedStereo

epsilon = np.finfo(float).eps
np.random.seed(1234)
random.seed(1995)


def read_cam_poses(file_cam_poses):
    poses = list()
    indices = list()
    print(file_cam_poses)
    fp = open(file_cam_poses)
    for line in fp:
        parser = line.strip().split(',')
        idx, x, y, z = float(parser[0]), float(parser[1]), float(parser[2]), float(parser[3])
        qx, qy, qz, qw = float(parser[4]), float(parser[5]), float(parser[6]), float(parser[7])
        r = Rotation.from_quat([qx, qy, qz, qw])
        matrix = r.as_matrix()
        t = np.array([x, y, z]).reshape(-1, 1) #
        # t = - np.dot(matrix, np.array([x, y, z]).reshape(-1, 1))
        trans_matrix = np.eye(4)
        trans_matrix[:3, :3] = matrix
        trans_matrix[:3, [3]] = t
        poses.append(np.linalg.inv(trans_matrix))
        indices.append(int(idx*1000))
    indices, poses = np.array(indices).astype(np.int16), np.array(poses)
    #isorted = np.argsort(indices)
    #indices = indices[isorted]
    #poses = poses[isorted]
    return poses, indices


def read_cam_poses_v2(file_cam_poses):
    poses = {}
    fp = open(file_cam_poses)
    for idx, line in enumerate(fp):
        parser = line.strip().split(',')
        indx = int(parser[0])
        arr = np.array(parser[1:], dtype=np.float32)
        trans_matrix = arr.reshape(4, 4)
        poses[indx] = trans_matrix
    return poses


def get_all_paths(root_dir, mode='train', load_sparse=True):
    data_dir = os.path.join(root_dir, mode)
    all_videos = [f for f in os.listdir(data_dir)]
    # all_videos = all_videos[:2] if mode == 'train' else all_videos[:1]
    all_paths = dict()
    for name in all_videos:
        path_video = os.path.join(data_dir, name)
        files = [f for f in os.listdir(path_video) if 'image' in f]
        indices = [int(f.split('.')[0][5:]) for f in files]
        indices = sorted(indices)
        # poses, indices = read_cam_poses_v2(file_cam_poses) #read_cam_poses(file_cam_poses)
        poses = list()
        img_paths, gt_paths, sdepth_paths, masks = list(), list(), list(), list()
        for id_in_data in indices:
            pose = np.loadtxt(os.path.join(path_video, 'pose%d.txt' % id_in_data))
            poses.append(np.linalg.inv(pose))
            img_paths.append(os.path.join(path_video, 'image%d.png' % id_in_data))
            gt_paths.append(os.path.join(path_video, 'depth%d.png' % id_in_data))
            if load_sparse:
                sdepth_paths.append(os.path.join(path_video, 'sparse%d.png' % id_in_data))
                #path_mask = os.path.join(path_video, 'mask%d.png' % id_in_data)
                #if os.path.exists(path_mask):
                #    masks.append(path_mask)

        all_paths[name] = dict(img_paths=img_paths, gt_paths=gt_paths,
                               sdmap_paths=sdepth_paths, cam_poses=poses, masks=masks)
    return all_paths


def create_sparse_depth(root_dir, out_dir, mode='train', num_samples=5000):
    data_dir = os.path.join(root_dir, mode)
    all_videos = [f for f in os.listdir(data_dir)]
    for name in all_videos:
        path_video = os.path.join(data_dir, name)
        gt_files = [f for f in os.listdir(path_video) if ('depth' in f) and ('mvs' not in f)]
        for file in gt_files:
            gt_path = os.path.join(path_video, file)
            gt_depth = np.array(Image.open(gt_path))
            # if sparsifier_type == UniformSampling.name:  # uar
            sparsifier = UniformSampling(num_samples=num_samples)
            # elif sparsifier_type == SimulatedStereo.name:  # sim_stereo
            #     sparsifier = SimulatedStereo(num_samples=num_samples, max_depth=depth_max)
            mask_keep = sparsifier.dense_to_sparse(None, gt_depth)
            sparse_depth = np.zeros(gt_depth.shape)
            sparse_depth[mask_keep] = gt_depth[mask_keep]
            out_folder = os.path.join(out_dir, name)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            sparse_path = os.path.join(out_folder, file.replace('depth', 'sparse'))
            cv2.imwrite(sparse_path, sparse_depth.astype(np.uint16))
            # img = np.array(Image.open(sparse_path))


class AerialDatasetSeq(data.Dataset):

    def __init__(self, mode, root_dir, img_size=(480, 752), patch_size=None, batch_size=1,
                 seq_size=2, skip_step=5, scale_factor=1, shuffle=False, depth_max=100, depth_min=1,
                 img_resize=(480, 752), sparsifier='fix', num_samples=None):
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
        self.sparsifier = sparsifier
        self.num_samples = num_samples

        if sparsifier != 'fix':
            self.all_paths = get_all_paths(root_dir, mode, load_sparse=False)
        else:
            self.all_paths = get_all_paths(root_dir, mode, load_sparse=True)
        rescale = float(img_resize[0] / img_size[0])
        full_img_size = (img_size[0]*rescale, img_size[1]*rescale)
        self.img_size = img_resize
        self.intrinsic_matrix = np.array([[455.0*rescale, 0, (img_resize[1] + 1)/2.],
                                          [0, 455.0*rescale, (img_resize[0] + 1 )/2.],
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

            if self.mode == 'train':
                indices = np.arange(num_imgs)
                for ptr in range(0, num_imgs, seq_size):
                    self.spliter.append((name, indices[ptr:(ptr+seq_size)]))
            else:
                self.spliter.append((name, np.arange(num_imgs)))

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

    def create_sparse_depth(self, gt_depth, num_samples=500):
        # if sparsifier_type == UniformSampling.name:  # uar
        sparsifier = UniformSampling(num_samples=num_samples)
        # elif sparsifier_type == SimulatedStereo.name:  # sim_stereo
        #     sparsifier = SimulatedStereo(num_samples=num_samples, max_depth=depth_max)
        mask_keep = sparsifier.dense_to_sparse(None, gt_depth)
        sparse_depth = np.zeros(gt_depth.shape)
        sparse_depth[mask_keep] = gt_depth[mask_keep]
        return sparse_depth

    def __getitem__(self, indx):
        '''
        outputs:
        img, dmap, extM, scene_path , as entries in a dic.
        '''
        name, id_data = self.generate_img_index[indx]
        video_data = self.all_paths[name]
        gt_path = video_data['gt_paths'][id_data]
        img_path = video_data['img_paths'][id_data]
        cam_pose = video_data['cam_poses'][id_data]


        K = self.intrinsic_matrix

        # read rgb image #
        img = Image.open(img_path)

        # read sparse depth map and GT depth map
        gt = Image.open(gt_path) if os.path.exists(gt_path) else None

        if self.sparsifier != 'fix':
            sdepth = self.create_sparse_depth(np.array(gt), num_samples=self.num_samples)
            sdepth = Image.fromarray(sdepth.astype(np.uint16))
        else:
            dmap_path = video_data['sdmap_paths'][id_data]
            sdepth = Image.open(dmap_path)

        img = self.transform_rgb(img)
        gt = self.transform_depth(gt) if gt is not None else None
        sdepth = self.transform_depth(sdepth)

        # normalize
        sdepth = sdepth.float() / 100
        gt = gt.float() / 100 if gt is not None else None

        if len(video_data['masks']) > 0:
            path_mask = video_data['masks'][id_data]
            mask = Image.open(path_mask)
            mask = self.transform_depth(mask)
            mask = mask > 0
        else:
            mask = (gt > self.depth_min) & (gt <= self.depth_max) if gt is not None else None
        if self.scale_factor == 0:
            max_depth = max(sdepth[mask].max(), 1.)
            scale = 10. / max_depth
            sdepth *= scale
            gt = gt * scale if gt is not None else None
        else:
            sdepth /= self.scale_factor
            gt = gt / self.scale_factor if gt is not None else None
            scale = 1. / self.scale_factor

        if gt is not None:
            return img, sdepth, torch.from_numpy(cam_pose).float(), torch.from_numpy(K).float(), \
                   torch.tensor(scale, dtype=torch.float32), gt, mask, self.list_begin[indx]
        else:
            return img, sdepth, torch.from_numpy(cam_pose).float(), torch.from_numpy(K).float(), \
                   torch.tensor(scale, dtype=torch.float32), self.list_begin[indx]

    def __len__(self):
        return len(self.generate_img_index)


if __name__ == '__main__':
    # dataset = AerialDatasetSeq('val', '/home/khangtg/Documents/lab/mvs/dataset/mvs/dataset',
    #                            batch_size=4, seq_size=100, img_resize=(240, 320))
    # print(len(dataset))
    # for i in range(30):
    #     data = dataset[i]
    #     for elem in data:
    #         print(elem.size())
    # create_sparse_depth('/home/khangtg/Documents/lab/mvs/dataset/mvs/aerial_dataset',
    #                     '/home/khangtg/Documents/lab/mvs/dataset/mvs/sparse_depths', mode='train', num_samples=10000)
    create_sparse_depth('/home/khangtg/Documents/lab/mvs/dataset/mvs/aerial/',
                        '/home/khangtg/Documents/lab/mvs/dataset/mvs/sparse_depths', mode='val', num_samples=1000)
    # create_sparse_depth('/home/khangtg/Documents/lab/mvs/dataset/mvs/dataset',
    #                     '/home/khangtg/Documents/lab/mvs/dataset/mvs/sparse_depths', mode='train', num_samples=10000)
    # create_sparse_depth('/home/khangtg/Documents/lab/mvs/dataset/mvs/dataset',
    #                     '/home/khangtg/Documents/lab/mvs/dataset/mvs/sparse_depths', mode='val', num_samples=10000)
