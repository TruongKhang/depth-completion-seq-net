import importlib, os
from datetime import datetime
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from warping import homography as homo

import torchvision.transforms as tf
import matplotlib.pyplot as plt
import matplotlib
from skimage.transform import rescale

class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install either TensorboardX with 'pip install tensorboardx', upgrade " \
                    "PyTorch to version >= 1.1 for using 'torch.utils.tensorboard' or turn off the option in " \
                    "the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


def read_cam_poses(file_cam_poses):
    poses = dict()
    print(file_cam_poses)
    fp = open(file_cam_poses)
    for line in fp:
        parser = line.strip().split(',')
        idx, x, y, z = int(parser[0]), float(parser[1]), float(parser[2]), float(parser[3])
        qx, qy, qz, qw = float(parser[4]), float(parser[5]), float(parser[6]), float(parser[7])
        r = Rotation.from_quat([qx, qy, qz, qw])
        matrix = r.as_dcm()
        t = np.array([x, y, z]).reshape(-1, 1) #- np.dot(matrix, np.array([x, y, z]).reshape(-1, 1))
        trans_matrix = np.eye(4)
        trans_matrix[:3, :3] = matrix
        trans_matrix[:3, [3]] = t
        poses[idx] = trans_matrix
    return poses


def visualize_pointcloud(depth_folder, cfd_folder, img_folder, gt_depth_folder, idx):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    # poses = read_cam_poses(os.path.join(gt_depth_folder, 'pose_orb_s.txt'))
    file_depth = '%s/%d.png' % (depth_folder, idx)
    # file_depth = '%s/depth965.png' % depth_folder
    # file_depth = '%s/depth_mvs%d.png' % (gt_depth_folder, idx)
    depth = Image.open(file_depth)
    # depth = tf.Compose([tf.Resize(240, interpolation=Image.NEAREST),
    #             tf.CenterCrop((240, 320))])(depth)
    # depth = tf.Compose([tf.Resize(240, interpolation=Image.NEAREST),
    #                     tf.CenterCrop((240, 320))])(depth)
    depth = np.array(depth, dtype=np.float32) / 100
    # plt.imshow(depth)
    # plt.show()
    # file_cfd = '%s/%d.png' % (cfd_folder, idx)
    # cfd = Image.open(file_cfd)
    # cfd = np.array(cfd).astype(np.float32)  # / 255
    # depth[cfd < 0.1] = 100
    # depth[depth < 1] = 100

    img_path = os.path.join(img_folder, 'image%d.png' % idx)
    # imgs, sdmaps, Rt, K, crop_at = data
    img = Image.open(img_path)
    img = tf.Compose([tf.Resize(240, interpolation=Image.NEAREST),
                      tf.CenterCrop((240, 320))])(img)
    img = np.array(img).astype(np.uint8)
    # img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
    # pil = Image.fromarray(img, 'RGB')
    # pil.save('image.png')
    # pil = Image.open('image.png')  # .convert('L')
    # img = np.array(pil)
    # img = mpimg.imread('image.png')
    print(img.shape, depth.shape)
    K = np.array([[455.0/2, 0, (320-1)/2.0], [0, 455.0/2, (240-1)/2.0], [0, 0, 1]])
    E = np.eye(4) #poses[idx]
    homo.img_to_pointcloud(img, depth, K, E)


def read_file_log(file_log, metric):
    fp = open(file_log)
    start = False
    line = fp.readline()
    list_epoch = list()
    list_train_loss = list()
    list_test_loss = list()
    while line:
        if 'Trainable' in line:
            start = True
            line = fp.readline().strip()
        if start:
            while line:
                print(line)
                if ' epoch ' in line:
                    epoch = int(line.split(':')[-1])
                    list_epoch.append(epoch)
                if ' %s ' % metric in line:
                    mae = float(line.split(':')[-1])
                    list_train_loss.append(mae)
                if ' val_%s ' % metric in line:
                    val_mae = float(line.split(':')[-1])
                    list_test_loss.append(val_mae)
                if 'Saving' in line:
                    break
                line = fp.readline().strip()
        line = fp.readline()
    return list_epoch, list_train_loss, list_test_loss


def plot(**kwargs):
    import matplotlib.pyplot as plt
    plt.figure()
    colors = ['r', 'g', 'b', 'k', 'y']
    for idx, (name, m) in enumerate(kwargs.items()):
        epoch1, train1, test1 = m
        print(len(epoch1), len(train1))
        # epoch2, train2, test2 = m2
        #print(len(list_epoch), len(list_train_loss), len(list_test_loss))

        plt.plot(epoch1, train1, '%s-' %colors[idx+1], label='Training %s' % name)
        plt.plot(epoch1, test1, '%s-' % colors[idx], label='Test %s' % name)
        # plt.plot(epoch2, train2, 'b*', label='Training (RGB-D image)')
        # plt.plot(epoch2, test2, 'b-', label='RGB-D images')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title("Evaluation")
    plt.show()


def plot_image(img_path, output_path, rmin=0, rmax=0, cmin=0, cmax=0):
    # import matplotlib.pyplot as plt
    # img = Image.open(img_path)
    # img = np.array(img)
    # plt.imsave(output_path, img, format='eps', dpi=1000)
    draw_with_rectangle(img_path, output_path, (rmin, rmax, cmin, cmax), is_color=True, linewidth=10)


def plot_error_map(gt_path, pred_path, output_path, dataset='kitti', subset='sub', rmin=162, rmax=271, cmin=840, cmax=1043):
    import matplotlib.pyplot as plt
    import matplotlib
    from skimage.transform import rescale

    gt = Image.open(gt_path)
    if dataset == 'kitti':
        gt = tf.CenterCrop((352, 1216))(gt)
    gt = np.array(gt, dtype=np.float32)
    gt = gt / 256 if dataset == 'kitti' else gt / 100
    pred = np.array(Image.open(pred_path), dtype=np.float32)
    pred = pred / 100
    error = np.abs(gt - pred) * 100
    mask = gt > 0
    print("RMSE: ", np.sqrt(np.mean((gt[mask] - pred[mask])**2)))
    error[error > 100] = 100

    if subset == 'sub':
        error = error[rmin:rmax, cmin:cmax]
        gt = gt[rmin:rmax, cmin:cmax]
        error = rescale(error, 3, anti_aliasing=False, order=0)
        gt = rescale(gt, 3, anti_aliasing=False, order=0)
    # error += 200
    # error[gt == 0] = 0

    # error_masked = error[gt > 0]
    # error_masked = error_masked[error_masked < ]
    # print(error_masked.max(), error_masked.min())
    error = error.astype(np.uint16)
    row, col = np.where(gt > 0)
    error[row[0], col[0]] = 1
    error[row[1], col[1]] = 100

    cmap = matplotlib.cm.plasma
    plt.figure()
    mbp = plt.imshow(error/100.0, cmap=cmap)
    fig, ax = plt.subplots()
    cbar = fig.colorbar(mbp, ax=ax, orientation='vertical', ticks=[0.01] + list(np.arange(0.1, 1.01, 0.1)))
    ax.remove()
    # tick_labels = ['<0.01']
    # for i, val in enumerate(np.arange(0.1, 1, 0.1)):
    #     if (i+1)%2 == 0:
    #         tick_labels.append('%.1f' % val)
    #     else:
    #         tick_labels.append(' ')
    # tick_labels.append('    >1 (m)')
    cbar.ax.set_yticklabels(['0.01'] + ['%.1f' % i for i in np.arange(0.1, 1, 0.1)] + ['>1 m'])
    cbar.ax.tick_params(labelsize=18)
    plt.savefig(output_path, bbox_inches='tight', format='eps', dpi=1000)

    # mask = (gt == 0)
    # draw_with_rectangle(None, output_path, (rmin, rmax, cmin, cmax), max_depth=100, cmap='plasma', mask_outlier=mask,
    #                     data_input=error, linewidth=10)


    # mask = np.ma.masked_where(gt == 0, error)
    # cmap = matplotlib.cm.plasma
    # cmap.set_bad(color='gray')
    # plt.imsave(output_path, mask, cmap=cmap, format='eps', dpi=1000, vmin=1)


def plot_depth(gt_path, pred_path, output_path, dataset='kitti', subset='sub', rmin=162, rmax=271, cmin=840, cmax=1043):

    gt = Image.open(gt_path)
    if dataset == 'kitti':
        gt = tf.CenterCrop((352, 1216))(gt)
    gt = np.array(gt, dtype=np.float32)
    gt = gt / 256 if dataset == 'kitti' else gt / 100
    pred = np.array(Image.open(pred_path))
    pred = pred.astype(np.uint16)

    if subset == 'sub':
        pred = pred[rmin:rmax, cmin:cmax]
        gt = gt[rmin:rmax, cmin:cmax]
        pred = rescale(pred, 2, anti_aliasing=False, order=0)
        gt = rescale(gt, 2, anti_aliasing=False, order=0)

    # print("showing")
    # cmap = matplotlib.cm.jet
    # plt.imsave(output_path, pred, format='eps', dpi=1000, vmin=0, cmap=cmap)
    draw_with_rectangle(pred_path, output_path, (rmin, rmax, cmin, cmax), subtract=500, linewidth=10, max_depth=50000)


def plot_confidence(cfd_path, output_path, dataset='kitti', subset='sub', rmin=162, rmax=271, cmin=840, cmax=1043,
                    a=1.0, b=0.0):
    import matplotlib.pyplot as plt
    import matplotlib
    from skimage.transform import rescale

    cfd = np.array(Image.open(cfd_path), dtype=np.float32) / 255

    if subset == 'sub':
        cfd = cfd[rmin:rmax, cmin:cmax]
        cfd = rescale(cfd, 3, anti_aliasing=False, order=0)
    # cfd = 1 / (1 + np.exp(-a*(cfd - b)))
    cfd[0, 0] = 0
    cfd[0, 1] = 1

    print("showing")
    # plt.figure(figsize=(9.3, 5))
    # plt.imshow(mask, cmap=cmap, vmin=1)
    # plt.colorbar()
    # plt.show()
    # plt.savefig(output_path, format='eps', dpi=1200)
    # cmap = matplotlib.cm.plasma

    plt.figure()
    mbp = plt.imshow(cfd)
    fig, ax = plt.subplots()
    cbar = fig.colorbar(mbp, ax=ax, orientation='horizontal')
    ax.remove()
    cbar.ax.tick_params(labelsize=18)
    plt.savefig(output_path, bbox_inches='tight', format='eps', dpi=1000)

    # plt.imsave(output_path, cfd, format='eps', dpi=1000, vmin=0)


def draw_with_rectangle(file_input, file_output, coord, linewidth=3, subtract=0, max_depth=6021,
                        is_color=False, cmap='jet', mask_outlier=None, data_input=None):
    input_map = Image.open(file_input) if file_input is not None else data_input
    input_map = np.array(input_map)
    input_map[input_map > max_depth] = max_depth
    input_map[0, 0] = 400
    input_map[1, 1] = max_depth
    input_map -= subtract
    x0, x1, y0, y1 = coord

    # sub_map = input_map[x0:x1, y0:y1]
    # sub_map = rescale(sub_map, 4, order=1, multichannel=True)
    # plt.imsave(file_output.replace('color_image', 'sub_color_image'), sub_map, format='eps', dpi=1000)

    mask = np.zeros(input_map.shape)

    w = linewidth // 2
    if is_color:
        input_map = np.concatenate((input_map[:, :, [2]], input_map[:, :, [1]], input_map[:, :, [0]]), axis=2)
        # mask[x0:x1, (y0 - w):(y0 + w), 0] = 1
        # mask[x0:x1, (y1 - w):(y1 + w), 0] = 1
        # mask[(x0 - w):(x0 + w), y0:y1, 0] = 1
        # mask[(x1 - w):(x1 + w), y0:y1, 0] = 1
        #
        # mask[x0:x1, (y0 - w):(y0 + w), 1] = 2
        # mask[x0:x1, (y1 - w):(y1 + w), 1] = 2
        # mask[(x0 - w):(x0 + w), y0:y1, 1] = 2
        # mask[(x1 - w):(x1 + w), y0:y1, 1] = 2
        #
        # mask[x0:x1, (y0 - w):(y0 + w), 2] = 3
        # mask[x0:x1, (y1 - w):(y1 + w), 2] = 3
        # mask[(x0 - w):(x0 + w), y0:y1, 2] = 3
        # mask[(x1 - w):(x1 + w), y0:y1, 2] = 3
        # input_map[mask == 1] = 128
        # input_map[mask == 2] = 128
        # input_map[mask == 3] = 128
        plt.imsave(file_output, input_map, format='eps', dpi=1000)
    else:
        # mask[x0:x1, (y0 - w):(y0 + w)] = 1
        # mask[x0:x1, (y1 - w):(y1 + w)] = 1
        # mask[(x0 - w):(x0 + w), y0:y1] = 1
        # mask[(x1 - w):(x1 + w), y0:y1] = 1
        # if mask_outlier is not None:
        #     mask = mask + np.array(mask_outlier, dtype=np.float32)
        # mask = np.ma.masked_where(mask > 0, input_map)
        cmap = matplotlib.cm.jet if cmap == 'jet' else matplotlib.cm.plasma
        plt.figure()
        mbp = plt.imshow(input_map/100, cmap=cmap)
        fig, ax = plt.subplots()
        cbar = fig.colorbar(mbp, ax=ax, orientation='horizontal', ticks=[4] + list(range(10, 81, 10)))
        ax.remove()
        cbar.ax.set_xticklabels(['4'] + [str(i) for i in range(10, 80, 10)] + ['     >80 (m)'])
        cbar.ax.tick_params(labelsize=18)
        plt.savefig(file_output, bbox_inches='tight', format='eps', dpi=1000)
        # cmap.set_bad(color='gray')
        # plt.imsave(file_output, input_map, format='eps', dpi=1000, cmap=cmap)


if __name__ == '__main__':
    # dataset_name = 'dsq-gremi-fortress-30'
    # fp = open("/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results/DepthRefinementModel/final_depth_maps/%s/highest_rmse.txt" %dataset_name)
    # indices = list()
    # for line in fp:
    #     indices.append(int(line.strip()))
    # print(indices)
    # visualize_pointcloud('/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results/DepthRefinementModel/final_depth_maps/%s' %dataset_name,
    #                      '/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results/DepthRefinementModel/final_cfd_maps/%s' %dataset_name,
    #                      "/home/khangtg/Documents/lab/mvs/dataset/mvs/dataset/val/%s" % dataset_name,
    #                      "/home/khangtg/Documents/lab/mvs/dataset/mvs/dataset/val/%s" % dataset_name, indices[0])

    # file_log1 = '/home/khangtg/Documents/lab/depth-completion-seq-net/saved/log/SeqModel/0309_194319/info.log'  # 1229_022912/info.log'

    # Upgraded model
    # file_log1 = '/home/khangtg/Documents/lab/mvs-dl/saved/log/UpgradedModel/1218_173506/info.log'
    # file_log2 = '/home/khangtg/Desktop/mvs-dl/saved/log/UpgradedModel/0109_163437/info.log'  # 0101_163942/info.log'
    # file_log3 = '/home/khangtg/Desktop/mvs-dl/saved/log/UpgradedModel/0112_130948/info.log'
    # file_log4 = '/home/khangtg/Desktop/mvs-dl/saved/log/Unet_depth/0111_135406/info.log'  # 1229_143018/info.log'
    # file_log5 = '/home/khangtg/Desktop/mvs-dl/saved/log/UpgradedModel/0109_164301/info.log'

    # m1 = read_file_log(file_log1, 'rmse')
    # m2 = read_file_log(file_log2, 'mae2')
    # m3 = read_file_log(file_log3, 'mae2')
    # m4 = read_file_log(file_log4, 'mae')
    # m5 = read_file_log(file_log5, 'mae2')
    # plot(rmse=m1)#, unet_depth_normal=m4, new_model_kernel3=m2, new_model_small_patch=m3,
         #new_model_dilation2=m5)
    # gt_path = '/home/khangtg/Documents/lab/code/data/kitti_dataset/kitti_depth/data_depth_annotated/val/' \
    #           '2011_09_26_drive_0095_sync/proj_depth/groundtruth/image_02/0000000100.png'
    # pred_path = '/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results_color_kitti_removed_outliers/' \
    #             'final_depth_maps/2011_09_26_drive_0095_sync_cam2/0000000100.png'
    # pred_path = '/home/khangtg/Documents/lab/code/nconv/workspace/exp_guided_nconv_cnn_l2/saved_results/' \
    #             '2011_09_26_drive_0005_sync_image_02/depth0000000030.png'
    # color_image = '/home/khangtg/Documents/lab/code/data/kitti_dataset/kitti_raw/2011_09_28/' \
    #               '2011_09_28_drive_0037_sync/image_02/data/0000000064.png'
    # gt_path = '/home/khangtg/Documents/lab/code/data/kitti_dataset/kitti_depth/data_depth_annotated/val/' \
    #           '2011_09_28_drive_0037_sync/proj_depth/groundtruth/image_02/0000000064.png'
    # pred_path = '/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results_gray_kitti/' \
    #             'final_depth_maps/2011_09_28_drive_0037_sync_cam2/0000000064.png'
    # cfd_path = '/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results_gray_kitti/' \
    #            'final_cfd_maps/2011_09_28_drive_0037_sync_cam2/0000000064.png'
    # pred_path_s2d = '/home/khangtg/Documents/lab/code/self-supervised-depth-completion/depth_kitti_gray/val_output/' \
    #                 '2011_09_28_drive_0037_sync_image_02/depth0000000064.png'
    # pred_path_nconv = '/home/khangtg/Documents/lab/code/nconv/workspace/exp_guided_nconv_cnn_l2/' \
    #                   'saved_results_kitti_gray/2011_09_28_drive_0037_sync_image_02/depth0000000064.png'
    # pred_path_dcnet = "/home/khangtg/Documents/lab/code/aerial-depth-completion/results/kitti_depth_cfd_gray/" \
    #                   "2011_09_28_drive_0037_sync_image_02/depth0000000064.png"
    # cfd_path_dcnet = "/home/khangtg/Documents/lab/code/aerial-depth-completion/results/kitti_depth_cfd_gray/" \
    #                  "2011_09_28_drive_0037_sync_image_02/cfd0000000064.png"

    # color_image = '/home/khangtg/Documents/lab/code/data/kitti_dataset/kitti_raw/2011_09_26/' \
    #               '2011_09_26_drive_0005_sync/image_02/data/0000000017.png'
    # gt_path = '/home/khangtg/Documents/lab/code/data/kitti_dataset/kitti_depth/data_depth_annotated/val/' \
    #           '2011_09_26_drive_0095_sync/proj_depth/groundtruth/image_02/0000000017.png'
    # pred_path = '/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results_gray_kitti/' \
    #             'final_depth_maps/2011_09_26_drive_0095_sync_cam2/0000000017.png'
    # cfd_path = '/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results_gray_kitti/' \
    #            'final_cfd_maps/2011_09_26_drive_0095_sync_cam2/0000000017.png'
    # pred_path_s2d = '/home/khangtg/Documents/lab/code/self-supervised-depth-completion/depth_kitti_gray/val_output/' \
    #                 '2011_09_26_drive_0095_sync_image_02/depth0000000017.png'
    # pred_path_nconv = '/home/khangtg/Documents/lab/code/nconv/workspace/exp_guided_nconv_cnn_l2/' \
    #                   'saved_results_kitti_gray/2011_09_26_drive_0095_sync_image_02/depth0000000017.png'
    # pred_path_dcnet = "/home/khangtg/Documents/lab/code/aerial-depth-completion/results/kitti_depth_cfd_gray/" \
    #                   "2011_09_26_drive_0095_sync_image_02/depth0000000017.png"
    # cfd_path_dcnet = "/home/khangtg/Documents/lab/code/aerial-depth-completion/results/kitti_depth_cfd_gray/" \
    #                  "2011_09_26_drive_0095_sync_image_02/cfd0000000017.png"
    #
    # dataset_name = '2011_09_28_drive_0037_sync_image_02' #'dsq-50-house' #'dsq-monteriggioni-b'
    # img_id = 64
    # color_image = '/home/khangtg/Documents/lab/mvs/dataset/mvs/aerial/val/%s/image%d.png' % (dataset_name, img_id)
    # gt_path = "/home/khangtg/Documents/lab/mvs/dataset/mvs/aerial/val/%s/depth%d.png" % (dataset_name, img_id)
    # pred_path = "/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results_gray_aerial/final_depth_maps/" \
    #             "%s/%d.png" % (dataset_name, img_id)
    # cfd_path = "/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results_gray_aerial/final_cfd_maps/" \
    #             "%s/%d.png" % (dataset_name, img_id)
    # pred_path_s2d = "/home/khangtg/Documents/lab/code/self-supervised-depth-completion/aerial_gray/val_output/" \
    #                 "%s/depth%d.png" % (dataset_name, img_id)
    # pred_path_nconv = "/home/khangtg/Documents/lab/code/nconv/workspace/exp_guided_nconv_cnn_l2/" \
    #                   "saved_results_aerial_gray/%s/depth%d.png" % (dataset_name, img_id)
    # pred_path_dcnet = "/home/khangtg/Documents/lab/code/aerial-depth-completion/results/aerial_depth_cfd_map_gray/" \
    #                   "%s/depth%d.png" % (dataset_name, img_id)
    # cfd_path_dcnet = "/home/khangtg/Documents/lab/code/aerial-depth-completion/results/aerial_depth_cfd_map_gray/" \
    #                  "%s/cfd%d.png" % (dataset_name, img_id)
    #
    # dataset = 'kitti'
    # mapping = {'nconv': pred_path_nconv, 's2d': pred_path_s2d,
    #            'our': pred_path, 'dcnet': pred_path_dcnet}
    # saved_folder = 'saved_images/%s/sample_%s' % (dataset, dataset_name)
    # if not os.path.exists(saved_folder):
    #     os.makedirs(saved_folder)
    # # plot_image(color_image, '%s/color_image%d.eps' %(saved_folder, img_id), rmin=184, rmax=453, cmin=8, cmax=215) #, rmin=77, rmax=240, cmin=101, cmax=346)
    # for mode in ['sub']:
    #     for name in ['our']: #['nconv', 's2d', 'dcnet', 'our']:
    #         # output_path = '%s/%s_error_map_%d_%s.eps' % (saved_folder, mode, img_id, name)
    #         output_path = '%s/colorbar_error.eps' % saved_folder
    #         plot_error_map(gt_path, mapping[name], output_path, dataset=dataset, subset=mode, rmin=210, rmax=310, cmin=550, cmax=640) #rmin=77, rmax=240, cmin=101, cmax=346)

            # output_path = '%s/%s_depth_%d_%s.eps' % (saved_folder, mode, img_id, name)
            # plot_depth(gt_path, mapping[name], output_path, dataset=dataset, subset=mode, rmin=184, rmax=453, cmin=8, cmax=215) #rmin=77, rmax=240, cmin=101, cmax=346)
    # # # plot confidence
    # mapping_cfd = {'dcnet': cfd_path_dcnet, 'our': cfd_path}
    # a = {'dcnet': 40, 'our': 20}
    # b = {'dcnet': 15.0/255.0, 'our': 200.0/255.0}
    # for mode in ['sub']:
    #     for name in mapping_cfd:
    #         output_path = '%s/%s_cfd_%d_%s.eps' % (saved_folder, mode, img_id, name)
    #         plot_confidence(mapping_cfd[name], output_path, dataset=dataset, subset=mode, rmin=210, rmax=310,
    #                         cmin=550, cmax=640, a=a[name], b=b[name])

    # gt_path = '/home/khangtg/Documents/lab/mvs/dataset/mvs/aerial/val/dsq-monteriggioni-b/depth4056.png'
    # pred_path = '/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results_gray_aerial/' \
    #             'final_depth_maps/dsq-monteriggioni-b/4056.png'

    # gt_path = '/home/khangtg/Documents/lab/mvs/dataset/mvs/aerial/val/dsq-gremi-fortress-30/depth1180.png'
    # pred_path = '/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results_gray_aerial/' \
    #             'final_depth_maps/dsq-gremi-fortress-30/1180.png'

    # gt_path = '/home/khangtg/Documents/lab/mvs/dataset/mvs/aerial/val/dsq-rifugio-antistress-30/depth47.png'
    # pred_path = '/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results_gray_aerial/' \
    #             'final_depth_maps/dsq-rifugio-antistress-30/47.png'

    # gt_path = '/home/khangtg/Documents/lab/mvs/dataset/mvs/aerial/val/dsq-twycross-bldg/depth54.png'
    # pred_path = '/home/khangtg/Documents/lab/depth-completion-seq-net/saved/results_gray_aerial/' \
    #             'final_depth_maps/dsq-twycross-bldg/54.png'

    # saved_folder = 'saved_images/aerial'
    # if not os.path.exists(saved_folder):
    #     os.makedirs(saved_folder)
    # output_path = '%s/error_map_54.png' % saved_folder
    # plot_error_map(gt_path, pred_path, output_path, dataset='aerial')

    draw_with_rectangle('/home/khangtg/Documents/depth260.png',
                        '/home/khangtg/Documents/colorbar_depth260.eps',
                        (100, 254, 602, 870), 6, subtract=0, max_depth=8000)
    plot_confidence('/home/khangtg/Documents/confidence260.png', '/home/khangtg/Documents/colorbar_confidence260.eps',
                    dataset='aerial', subset='full', a=20, b=200.0/255.0)
