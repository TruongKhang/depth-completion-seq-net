import numpy as np
import os
import torch
import torch.nn.functional as F
from base import BaseTrainer
from utils import inf_loop, MetricTracker, util
from warping import homography as homo
from models.loss import cfd_loss_decay


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


class KittiTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.seq_size = config['data_loader']['args']['seq_size']
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
        self.log_step = config['trainer']['logging_every']
        self.thresh = config['trainer']['cfd_thresh']

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
        self.data_loader.shuffle_and_crop()
        itg_state = None
        prev_E = None

        for batch_idx, data in enumerate(self.data_loader):
            data = to_device(tuple(data), self.device)
            imgs, sdmaps, E, K, scale, gt, mask, is_begin_video = data
            target = (gt, mask)
            is_begin_video = is_begin_video.type(torch.uint8)
            if itg_state is None:
                init_depth = torch.zeros(target[0].size(), dtype=torch.float32)
                init_cfd = torch.zeros(target[0].size(),
                                       dtype=torch.float32)
                itg_state = init_depth, init_cfd
                itg_state = to_device(itg_state, self.device)
                prev_E = E #torch.eye(4).unsqueeze(0).repeat(target[0].size(0), 1, 1)
            else:
                if self.config['trainer']['seq']:
                    itg_state[0][is_begin_video] = 0.
                    itg_state[1][is_begin_video] = 0.
                    prev_E[is_begin_video] = E[is_begin_video]
                else:
                    itg_state[0].zero_()
                    itg_state[1].zero_()

            # warped_depth, warped_cfd = homo.warp_cfd(itg_state[0], itg_state[1], K, rel_E, crop_at=crop_at)
            warped_depth, warped_cfd = homo.warping(itg_state[0], itg_state[1], K, prev_E, K, E)
            warped_depth *= scale.view(-1, 1, 1, 1) # self.scale_factor
            prev_E = E

            self.optimizer.zero_grad()
            if self.config['arch']['type'] == 'DepthCompletionNet':
                final_depth, final_cfd, init_depth, init_cfd = self.model((imgs, sdmaps), prev_state=(warped_depth, warped_cfd))
                d = final_depth.detach()
                loss, gt_cfd = self.criterion(final_depth, final_cfd, target, init_depth, init_cfd, scale.view(-1, 1, 1, 1),
                                              self.thresh)
            elif self.config['arch']['type'] == 'ResUnet':
                final_depth, _ = self.model((imgs, sdmaps))
                d = final_depth.detach()
                final_cfd = None
                loss = self.criterion(final_depth, None, target, d)
            elif self.config['arch']['type'] == 'NormCNN':
                final_depth, final_cfd = self.model(sdmaps, (sdmaps > 0).float())
                d = final_depth.detach()
                loss = cfd_loss_decay(final_depth, final_cfd, target, epoch)
            else:
                final_depth, final_cfd = self.model((imgs, sdmaps), prev_state=(warped_depth, warped_cfd))
                d = final_depth.detach()
                loss = self.criterion(final_depth, final_cfd, target, d)

            loss.backward()
            self.optimizer.step()

            c = final_cfd.detach()
            iscale = 1 / scale.view(-1, 1, 1, 1)
            itg_state = (d*iscale, c)

            self.train_metrics.update('loss', loss.item(), n=target[0].size(0))
            for met in self.metric_ftns:
                if met.__name__ != 'deltas':
                    self.train_metrics.update(met.__name__, met(d, target, scale_factor=iscale).item(),
                                              n=target[0].size(0))
                else:
                    for i in range(1, 4):
                        self.train_metrics.update('delta_%d' % i, met(d, target, i, scale_factor=iscale).item(),
                                                  n=target[0].size(0))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}, #processed_frames: {} Loss: {:.6f}, RMSE: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    self.train_metrics.avg('loss'),
                    self.train_metrics.avg('rmse')))

        log = self.train_metrics.result()

        if self.do_validation:
            # if epoch%2 == 0:
            #     save_folder = '/home/khang/results/'
            # else:
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
        prev_E = None
        # self.valid_data_loader.shuffle_and_crop()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                data = to_device(tuple(data), self.device)
                imgs, sdmaps, E, K, scale, gt, mask, is_begin_video = data
                target = (gt, mask)
                is_begin_video = is_begin_video.type(torch.uint8)
                if itg_state is None:
                    init_depth = torch.zeros(target[0].size(), dtype=torch.float32)
                    init_cfd = torch.zeros(target[0].size(),
                                           dtype=torch.float32)  # - 1.  # to forget the first frame of each video
                    itg_state = init_depth, init_cfd
                    itg_state = to_device(itg_state, self.device)
                    prev_E = E #torch.eye(4).unsqueeze(0).repeat(target[0].size(0), 1, 1)
                else:
                    if self.config['trainer']['seq']:
                        itg_state[0][is_begin_video] = 0.
                        itg_state[1][is_begin_video] = 0.
                        prev_E[is_begin_video] = E[is_begin_video]
                    else:
                        itg_state[0].zero_()
                        itg_state[1].zero_()

                # warped_depth, warped_cfd = homo.warp_cfd(itg_state[0], itg_state[1], K, rel_E, crop_at=crop_at)
                warped_depth, warped_cfd = homo.warping(itg_state[0], itg_state[1], K, prev_E, K, E)
                warped_depth *= scale.view(-1, 1, 1, 1)
                prev_E = E

                if self.config['arch']['type'] == 'DepthCompletionNet':
                    final_depth, final_cfd, init_depth, init_cfd = self.model((imgs, sdmaps), prev_state=(warped_depth, warped_cfd))
                    d = final_depth.detach()
                    loss, gt_cfd = self.criterion(final_depth, final_cfd, target, init_depth, init_cfd,
                                                  scale.view(-1, 1, 1, 1), self.thresh)
                elif self.config['arch']['type'] == 'ResUnet':
                    final_depth, _ = self.model((imgs, sdmaps))
                    d = final_depth.detach()
                    final_cfd = None
                    loss = self.criterion(final_depth, None, target, d)
                elif self.config['arch']['type'] == 'NormCNN':
                    final_depth, final_cfd = self.model(sdmaps, (sdmaps > 0).float())
                    d = final_depth.detach()
                    loss = cfd_loss_decay(final_depth, final_cfd, target, epoch)
                else:
                    final_depth, final_cfd = self.model((imgs, sdmaps), prev_state=(warped_depth, warped_cfd))
                    d = final_depth.detach()
                    loss = self.criterion(final_depth, final_cfd, target, d)

                c = final_cfd.detach()
                iscale = 1 / scale.view(-1, 1, 1, 1)
                if self.config['arch']['type'] == 'DepthCompletionNet3':
                    c = torch.exp(- self.thresh * torch.exp(c) * iscale)
                itg_state = (d*iscale, c)

                self.valid_metrics.update('loss', loss.item(), n=target[0].size(0))
                for met in self.metric_ftns:
                    if met.__name__ != 'deltas':
                        self.valid_metrics.update(met.__name__, met(d, target, scale_factor=iscale).item(),
                                                  n=target[0].size(0))
                    else:
                        for i in range(1, 4):
                            self.valid_metrics.update('delta_%d' % i, met(d, target, i, scale_factor=iscale).item(),
                                                      n=target[0].size(0))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.get_num_samples()
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
