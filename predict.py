import argparse, os
import torch
import numpy as np
import data_loader.data_loaders as module_data
import models.loss as module_loss
import models.metric as module_metric
import models.model as module_arch
from parse_config import ConfigParser
from trainer.trainer import to_device
from utils.util import MetricTracker
from utils import util
from warping import homography as homo


def main(config, saved_folder=None):
    logger = config.get_logger('predict')

    # setup data_loader instances
    if 'KittiLoader' in config['data_loader']['type']:
        init_kwags = {
            "kitti_depth_dir": config['data_loader']['args']['kitti_depth_dir'],
            "kitti_raw_dir": config['data_loader']['args']['kitti_raw_dir'],
            # "root_dir": config['data_loader']['args']['root_dir'],
            "batch_size": 1,
            "shuffle": False,
            "img_size": config['val_img_size'],
            "num_workers": config['data_loader']['args']['num_workers'],
            "mode": "val",
            "scale_factor": config['data_loader']['args']['scale_factor'],
            "seq_size": config['data_loader']['args']['seq_size'],
            "cam_ids": config['data_loader']['args']['cam_ids']
        }
        data_loader = getattr(module_data, config['data_loader']['type'])(**init_kwags)
    else:
        init_kwags = {
            "root_dir": config['data_loader']['args']['root_dir'],
            "batch_size": 1,
            "shuffle": False,
            "img_size": config['val_img_size'],
            "num_workers": config['data_loader']['args']['num_workers'],
            "mode": "test",
            "scale_factor": config['data_loader']['args']['scale_factor'],
            "seq_size": config['data_loader']['args']['seq_size'],
            "img_resize": config['data_loader']['args']['img_resize']
        }
        data_loader = getattr(module_data, config['data_loader']['type'])(**init_kwags)

    # build models architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    name_metrics = list()
    for m in metric_fns:
        if m.__name__ != 'deltas':
            name_metrics.append(m.__name__)
        else:
            for i in range(1, 4):
                name_metrics.append("delta_%d" % i)
    total_metrics = MetricTracker('loss', *name_metrics)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(str(config.resume))
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, val in state_dict.items():
        new_state_dict[key.replace('module.', '')] = val
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(new_state_dict)

    # prepare models for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    itg_state = None

    if saved_folder is not None:
        path_idepth = os.path.join(saved_folder, 'final_depth_maps')
        if not os.path.exists(path_idepth):
            os.makedirs(path_idepth)
        path_icfd = os.path.join(saved_folder, 'final_cfd_maps')
        if not os.path.exists(path_icfd):
            os.makedirs(path_icfd)

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            data = to_device(tuple(data), device)
            print(len(data))
            imgs, sdmaps, E, K, scale, is_begin_video = data
            is_begin_video = is_begin_video.type(torch.uint8)
            if itg_state is None:
                init_depth = torch.zeros(sdmaps.size(), dtype=torch.float32)
                init_cfd = torch.zeros(sdmaps.size(),
                                       dtype=torch.float32)
                itg_state = init_depth, init_cfd
                itg_state = to_device(itg_state, device)
                prev_E = E
            else:
                if config['trainer']['seq']:
                    itg_state[0][is_begin_video] = 0.
                    itg_state[1][is_begin_video] = 0.
                    prev_E[is_begin_video] = E[is_begin_video]
                else:
                    itg_state[0].zero_()
                    itg_state[1].zero_()

            warped_depth, warped_cfd = homo.warping(itg_state[0], itg_state[1], K, prev_E, K, E)
            warped_depth *= scale.view(-1, 1, 1, 1)
            prev_E = E

            final_depth, final_cfd, _, _ = model((imgs, sdmaps), prev_state=(warped_depth, warped_cfd))
            d = final_depth.detach()
            c = final_cfd.detach()

            iscale = 1. / scale.view(-1, 1, 1, 1)
            itg_state = (d * iscale, c)

            if config['data_loader']['type'] == 'KittiLoaderv2':
                name, id_data = data_loader.kitti_dataset.generate_img_index[batch_idx]
                video_data = data_loader.kitti_dataset.all_paths[name]
            elif config['data_loader']['type'] == 'VISIMLoader':
                name, id_data = data_loader.visim_dataset.generate_img_index[batch_idx]
                video_data = data_loader.visim_dataset.all_paths[name]
            elif config['data_loader']['type'] == 'SevenSceneLoader':
                name, id_data = data_loader.scene7_dataset.generate_img_index[batch_idx]
                video_data = data_loader.scene7_dataset.all_paths[name]
            img_path = video_data['img_paths'][id_data]
            if config['data_loader']['type'] == 'KittiLoaderv2':
                id_img = img_path.split('/')[-1].split('.')[0]
            elif config['data_loader']['type'] == 'VISIMLoader':
                id_img = img_path.split('/')[-1].split('.')[0][5:]
            elif config['data_loader']['type'] == 'SevenSceneLoader':
                id_img = img_path.split('/')[-1].split('.')[0].split('-')[-1]

            if saved_folder is not None:
                if (batch_idx+1) % 1 == 0:
                    final_depth = itg_state[0].squeeze(0).squeeze(0).cpu().numpy() * 100
                    if config['data_loader']['type'] == 'KittiLoaderv2':
                        subfolder_idepth = os.path.join(path_idepth, '_'.join(name.split('_')))
                    elif config['data_loader']['type'] == 'VISIMLoader':
                        subfolder_idepth = os.path.join(path_idepth, name.split('_')[0])
                    elif config['data_loader']['type'] == 'SevenSceneLoader':
                        subfolder_idepth = os.path.join(path_idepth, '/'.join(name.split('_')))
                    if not os.path.exists(subfolder_idepth):
                        os.makedirs(subfolder_idepth)
                    util.save_image(subfolder_idepth, '%s.png' % id_img, final_depth.astype(np.uint16), saver='opencv')
                    c_new = c.squeeze(0).squeeze(0).cpu().numpy() * 255
                    if config['data_loader']['type'] == 'KittiLoaderv2':
                        subfolder_cfd = os.path.join(path_icfd, '_'.join(name.split('_')))
                    elif config['data_loader']['type'] == 'VISIMLoader':
                        subfolder_cfd = os.path.join(path_icfd, name.split('_')[0])
                    elif config['data_loader']['type'] == 'SevenSceneLoader':
                        subfolder_cfd = os.path.join(path_icfd, '/'.join(name.split('_')))
                    if not os.path.exists(subfolder_cfd):
                        os.makedirs(subfolder_cfd)
                    util.save_image(subfolder_cfd, '%s.png' % id_img, c_new.astype(np.uint8), saver='opencv')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--save_folder', default=None, type=str,
                      help='path to save the results')

    config = ConfigParser.from_args(args)
    parse_args = args.parse_args()
    main(config, saved_folder=parse_args.save_folder)
