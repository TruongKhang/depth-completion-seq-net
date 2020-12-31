import numpy as np
import os
import torch
from models.loss import smooth_l1_loss, cfd_loss_decay
from base import BaseTrainer
from utils import inf_loop, MetricTracker, util
from warping import homography as homo


def to_device(data, device):
    new_data = ()
    if type(data) is tuple:
        for i in range(len(data)):
            if data[i] is not None:
                new_data += (to_device(data[i], device),)
            else:
                new_data += (None, )
    else:
        new_data = data.to(device)
    return new_data


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.skipping_frames = config['trainer']['skipping_frames']
        self.data_loader = data_loader
        self.data_loader.set_device(self.device)
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer']['logging_every'] # int(np.sqrt(data_loader.batch_size))
        self.scale_factor = config['data_loader']['args']['scale_factor']

        name_metrics = list()
        for m in self.metric_ftns:
            if m.__name__ != 'deltas':
                name_metrics.append(m.__name__)
            else:
                for i in range(1, 4):
                    name_metrics.append("delta_%d" % i)
        self.train_metrics = MetricTracker('loss', *name_metrics, writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *name_metrics, writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        itg_state = None
        self.data_loader.initialize()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            if itg_state is None:
                init_depth = torch.zeros(target[0].size(), dtype=torch.float32)
                init_cfd = torch.zeros(target[0].size(),
                                       dtype=torch.float32) #- 1.  # to forget the first frame of each video
                itg_state = init_depth, init_cfd
                itg_state = to_device(itg_state, self.device)
            else:
                for idx, is_new in enumerate(self.data_loader.new_added_videos):
                    if is_new:
                        itg_state[0][idx] = 0
                        itg_state[1][idx] = 0 #-1
            for idx in self.data_loader.new_added_videos.nonzero()[0]:
                if self.data_loader.ptrs[idx] >= self.skipping_frames:
                    self.data_loader.new_added_videos[idx] = False

            data, target = to_device(data, self.device), to_device(target, self.device) # data.to(self.device), target.to(self.device)
            imgs, sdmaps, Rt, K, crop_at = data
            warped_depth, warped_cfd = homo.warp_cfd(itg_state[0], itg_state[1], K, Rt, crop_at=crop_at)
            warped_depth /= self.scale_factor

            self.optimizer.zero_grad()
            final_depth, final_cfd = self.model((imgs, sdmaps), prev_state=(warped_depth, warped_cfd)) #, crop_at=crop_at)
            d = final_depth.detach()
            loss = self.criterion(final_depth, final_cfd, target, d)
            # final_depth = self.model((imgs, sdmaps))
            # loss = smooth_l1_loss(final_depth, target)
            loss.backward()
            self.optimizer.step()

            c = final_cfd.detach()
            # convert confidence to range (0, 1) which is the input for prediction of next frame
            # c = torch.exp(- torch.exp(c / 2) * self.scale_factor)
            c = torch.exp(-c * self.scale_factor)
            itg_state = (d*self.scale_factor, c)

            # itg_state = self.data_loader.update_valid_samples(itg_state, valid_itg_state)

            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), n=target[0].size(0))
            for met in self.metric_ftns:
                if met.__name__ != 'deltas':
                    self.train_metrics.update(met.__name__, met(d, target, scale_factor=self.scale_factor).item(),
                                              n=target[0].size(0))
                else:
                    for i in range(1, 4):
                        self.train_metrics.update('delta_%d' % i, met(d, target, i, scale_factor=self.scale_factor).item(),
                                                  n=target[0].size(0))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}, #processed_videos: {}, #processed_frames: {} Loss: {:.6f}, RMSE: {:.6f}'.format(
                    epoch,
                    self.data_loader.num_processed_videos,
                    self._progress(self.data_loader.num_processed_frames),
                    self.train_metrics.avg('loss'),
                    self.train_metrics.avg('rmse')))

        log = self.train_metrics.result()

        if self.do_validation:
            if epoch%2 == 0:
                save_folder = '/home/khang/project/kitti_dataset/results/no_integration'
            else:
                save_folder = None
            val_log = self._valid_epoch(epoch, save_folder=save_folder)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch, save_folder=None):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        print("Validation at epoch %d, size of validation set: %d, batch_size: %d" % (epoch, len(self.valid_data_loader),
                                                                                     self.valid_data_loader.batch_size))
        if save_folder is not None:
            path_depth = os.path.join(save_folder, 'depth_maps')
            if not os.path.exists(path_depth):
                os.makedirs(path_depth)
            path_cfd = os.path.join(save_folder, 'confidence')
            if not os.path.exists(path_cfd):
                os.makedirs(path_cfd)

        self.model.eval()
        self.valid_metrics.reset()
        itg_state = None
        self.valid_data_loader.initialize()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                if itg_state is None:
                    init_depth = torch.zeros(target[0].size(), dtype=torch.float32)
                    init_cfd = torch.zeros(target[0].size(),
                                           dtype=torch.float32)  # - 1.  # to forget the first frame of each video
                    itg_state = init_depth, init_cfd
                    itg_state = to_device(itg_state, self.device)
                else:
                    for idx, is_new in enumerate(self.valid_data_loader.new_added_videos):
                        if is_new:
                            itg_state[0][idx] = 0
                            itg_state[1][idx] = 0  # -1
                            self.valid_data_loader.new_added_videos[idx] = False

                # crop_at = data[-1]
                data, target = to_device(data, self.device), to_device(target, self.device)
                imgs, sdmaps, Rt, K, crop_at = data
                warped_depth, warped_cfd = homo.warp_cfd(itg_state[0], itg_state[1], K, Rt, crop_at=crop_at)
                warped_depth /= self.scale_factor

                final_depth, final_cfd = self.model((imgs, sdmaps), prev_state=(warped_depth, warped_cfd))
                d = final_depth.detach()
                loss = self.criterion(final_depth, final_cfd, target, d)
                # final_depth = self.model((imgs, sdmaps))
                # loss = smooth_l1_loss(final_depth, target)

                c = final_cfd.detach()
                # convert confidence to range (0, 1) which is the input for prediction of next frame
                # c = torch.exp(- torch.exp(c / 2) * self.scale_factor)
                c = torch.exp(-c * self.scale_factor)
                itg_state = (d * self.scale_factor, c)

                if save_folder is not None:
                    util.save_image(path_depth, '%d.png' % batch_idx, d.squeeze(0).squeeze(0).cpu().numpy())
                    # final_cfd = torch.exp(final_cfd)
                    # final_cfd[final_cfd > 1] = 1
                    # final_cfd = torch.add(-final_cfd, 1)
                    util.save_image(path_cfd, '%d.png' % batch_idx,
                                    c.squeeze(0).squeeze(0).cpu().numpy())

                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item(), n=target[0].size(0))
                for met in self.metric_ftns:
                    if met.__name__ != 'deltas':
                        self.valid_metrics.update(met.__name__, met(d, target, scale_factor=self.scale_factor).item(),
                                                  n=target[0].size(0))
                    else:
                        for i in range(1, 4):
                            self.valid_metrics.update('delta_%d' % i, met(d, target, i, scale_factor=self.scale_factor).item(),
                                                      n=target[0].size(0))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
