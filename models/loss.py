import math
import torch
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def input_output_loss(outputs, target, cout, epoch_num, inputs, *args):
    val_pixels = torch.ne(target, 0).float().cuda()
    err = F.smooth_l1_loss(outputs * val_pixels, target * val_pixels, reduction='none')

    val_pixels = torch.ne(inputs, 0).float().cuda()
    inp_loss = F.smooth_l1_loss(outputs * val_pixels, inputs * val_pixels, reduction='none')

    loss = err + 0.1 * inp_loss

    return torch.mean(loss)


def cfd_loss_decay(outputs, cout, target, epoch_num, *args):
    # val_pixels = torch.ne(target, 0).float() #.cuda()
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    err = F.smooth_l1_loss(outputs * val_pixels, gt_depth * val_pixels, reduction='none')
    cert = cout * val_pixels - err * cout * val_pixels
    loss = err - np.exp(-0.1*(epoch_num-1)) * cert
    return torch.sum(loss) / torch.sum(val_pixels)


def cfd_loss_decay_mse(outputs, cout, target, epoch_num, *args):
    # val_pixels = torch.ne(target, 0).float() #.cuda()
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    err = F.mse_loss(outputs * val_pixels, gt_depth * val_pixels, reduction='none')
    cert = cout * val_pixels - err * cout * val_pixels
    loss = err - (1 / epoch_num) * cert
    return torch.sum(loss) / torch.sum(val_pixels)


def cfd_loss(outputs, target, cout, *args):
    val_pixels = torch.ne(target, 0).float().cuda()
    err = F.smooth_l1_loss(outputs * val_pixels, target * val_pixels, reduction='none')
    loss = err - cout * val_pixels + err * cout * val_pixels
    return torch.mean(loss)


def smooth_l1_loss(outputs, target, *args):
    # val_pixels = torch.ne(target, 0) #.float().cuda()
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    loss = F.smooth_l1_loss(outputs*val_pixels, gt_depth*val_pixels, reduction='none')
    return torch.sum(loss) / torch.sum(val_pixels)


def rmse_loss(outputs, target, *args):
    val_pixels = (target > 0).float().cuda()
    err = (target * val_pixels - outputs * val_pixels) ** 2
    loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
    cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
    return torch.mean(torch.sqrt(loss / cnt))


def mse_loss(outputs, target, *args):
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    loss = gt_depth * val_pixels - outputs * val_pixels
    return torch.mean(loss ** 2)


def total_loss_l1(outputs1, outputs2, target, cout, epoch_num, ld=0.1):
    loss1 = cfd_loss_decay(outputs1, target, cout, epoch_num)
    loss2 = smooth_l1_loss(outputs2, target)
    return ld * loss1 + (1 - ld) * loss2


def total_loss_mse(outputs1, outputs2, target, cout, epoch_num, ld=0.5):
    loss1 = cfd_loss_decay_mse(outputs1, target, cout, epoch_num)
    loss2 = mse_loss(outputs2, target)
    return ld * loss1 + (1 - ld) * loss2


def masked_l1_gauss(means, log_vars, targets, *args):
    # (means has shape: (batch_size, 1, h, w))
    # (log_vars has shape: (batch_size, 1, h, w))
    # (targets has shape: (batch_size, 1, h, w))
    gt_depths, valid_mask = targets
    valid_mask = valid_mask.float()
    # cnt = gt_depths.size(0) * gt_depths.size(2) * gt_depths.size(3)
    coarse_depth = args[0]

    # gt_depths = gt_depths[valid_mask]
    # means = means[valid_mask]
    # log_vars = log_vars[valid_mask]
    loss1 = torch.mean(torch.exp(-log_vars) * torch.abs(gt_depths - means) + log_vars)
    grad_loss = L1_Gradient_loss()(means, gt_depths, valid_mask)
    loss2 = F.l1_loss(coarse_depth * valid_mask, gt_depths * valid_mask)

    return loss1 + grad_loss + 0.01*loss2


class L1_Gradient_loss(torch.nn.Module):
    def __init__(self):
        super(L1_Gradient_loss, self).__init__()
        self.eps = 1e-6
        self.crit = torch.nn.L1Loss(reduction='none')

    def forward(self, X, Y, mask):
        mask_x = mask[:, :, 1:, 1:] - mask[:, :, 0:-1, 1:]
        mask_y = mask[:, :, 1:, 1:] - mask[:, :, 1:, 0:-1]
        valid_mask = (mask_x.eq(0) * mask_y.eq(0)).float() * F.pad(mask, (-1, 0, -1, 0))
        xgin = X[:, :, 1:, 1:] - X[:, :, 0:-1, 1:]
        ygin = X[:, :, 1:, 1:] - X[:, :, 1:, 0:-1]
        xgtarget = Y[:, :, 1:, 1:] - Y[:, :, 0:-1, 1:]
        ygtarget = Y[:, :, 1:, 1:] - Y[:, :, 1:, 0:-1]

        xl = self.crit(xgin, xgtarget)
        yl = self.crit(ygin, ygtarget)
        grad_loss = (xl + yl) * 0.5 * valid_mask
        return torch.mean(grad_loss)


def multi_losses_kitti(depths, cfds, targets, *args):
    gt_depths, valid_mask = targets
    valid_mask = valid_mask.float()
    const_depths = args[0]
    coarse_depth = args[1]
    scale_factor = args[2]
    thresh = args[3]

    #gt_depths[gt_depths == 0] = 1e-5
    #error = torch.abs(gt_depths - const_depths)
    #mask = error > thresh * scale_factor
    #error[mask] = gt_depths[mask]
    #gt_cfds = error / gt_depths
    #gt_cfds[gt_cfds > 1] = 1
    #gt_cfds = 1 - gt_cfds
    error = torch.abs(gt_depths - const_depths) / scale_factor
    gt_cfds = torch.exp(-error)

    loss1 = F.mse_loss(depths*valid_mask, gt_depths*valid_mask)
    loss2 = F.mse_loss(cfds*valid_mask, gt_cfds*valid_mask)
    # penalty = torch.mean(cfds*valid_mask ** 2)
    # loss3 = F.l1_loss(coarse_depth*valid_mask, gt_depths*valid_mask)
    # loss4 = F.smooth_l1_loss(itg_depth * valid_mask, gt_depths*valid_mask)
    loss = loss1 + loss2 #+ 0.3*loss3  # + 0.5*loss4
    return loss, gt_cfds


def multi_losses(depths, cfds, targets, *args):
    gt_depths, valid_mask = targets
    valid_mask = valid_mask.float()
    init_depth = args[0]
    init_cfd = args[1]
    scale_factor = args[2]
    thresh = args[3]
    const_depths = depths.detach()
    # gt_depths[gt_depths == 0] = 1e-5
    error = torch.abs(gt_depths - const_depths) / scale_factor
    gt_cfds = torch.exp(-error)

    loss1 = F.mse_loss(depths*valid_mask, gt_depths*valid_mask)
    # grad_loss = L1_Gradient_loss()(depths, gt_depths, valid_mask)
    # loss2 = F.binary_cross_entropy(cfds * valid_mask, gt_cfds * valid_mask) #
    loss2 = F.mse_loss(cfds*valid_mask, gt_cfds*valid_mask)
    # loss3 = F.l1_loss(coarse_depth*valid_mask, gt_depths*valid_mask)
    # loss4 = F.smooth_l1_loss(itg_depth * valid_mask, gt_depths*valid_mask)
    init_error = torch.abs(gt_depths - init_depth.detach()) / scale_factor
    gt_init_cfd = torch.exp(-init_error)
    loss3 = F.mse_loss(init_depth*valid_mask, gt_depths*valid_mask)
    loss4 = F.mse_loss(init_cfd*valid_mask, gt_init_cfd*valid_mask)

    loss = loss1 + loss2 + 0.3*loss3 + loss4
    # cfds = cfds[valid_mask > 0]
    # gt_cfds = gt_cfds[valid_mask > 0]
    # print(cfds.max().item(), cfds.min().item(), cfds.mean().item())
    # print(gt_cfds.max().item(), gt_cfds.min().item(), gt_cfds.mean().item())
    # print(loss1.item(), loss2.item())
    return loss, gt_cfds


def sum_multi_losses(means, stds, targets, *args):
    gt_depths, valid_mask = targets
    valid_mask = valid_mask.float()
    const_means = args[0]
    coarse_depth = args[1]
    epoch = args[2]
    itg_depth = args[3]

    # gt_depths = gt_depths[valid_mask]
    # means = means[valid_mask]
    # stds = stds[valid_mask]
    # const_means = const_means[valid_mask]
    gt_stds = torch.abs(gt_depths - const_means) / gt_depths
    gt_stds[gt_stds > 1] = 1
    gt_stds = 1 - gt_stds
    loss1 = F.smooth_l1_loss(means*valid_mask, gt_depths*valid_mask)
    grad_loss = L1_Gradient_loss()(means, gt_depths, valid_mask)
    loss2 = F.smooth_l1_loss(stds*valid_mask, gt_stds*valid_mask)
    loss3 = F.smooth_l1_loss(coarse_depth*valid_mask, gt_depths*valid_mask)
    loss4 = F.smooth_l1_loss(itg_depth * valid_mask, gt_depths*valid_mask)
    #base2 = loss2.detach()
    loss = loss1 + grad_loss + loss2 + 0.1*loss3 + 0.1*loss4 # math.exp(-0.1*epoch) * loss3
    return loss


def sum_multi_losses_kitti(means, stds, targets, *args):
    gt_depths, valid_mask = targets
    valid_mask = valid_mask.float()
    const_means = args[0]
    coarse_depth = args[1]
    epoch = args[2]
    itg_depth = args[3]

    # gt_depths = gt_depths[valid_mask]
    # means = means[valid_mask]
    # stds = stds[valid_mask]
    # const_means = const_means[valid_mask]
    gt_stds = torch.abs(gt_depths - const_means) / gt_depths
    gt_stds[gt_stds > 1] = 1
    gt_stds = 1 - gt_stds
    loss1 = F.smooth_l1_loss(means*valid_mask, gt_depths*valid_mask)
    # grad_loss = L1_Gradient_loss()(means, gt_depths) * valid_mask
    loss2 = F.smooth_l1_loss(stds*valid_mask, gt_stds*valid_mask)
    loss3 = F.smooth_l1_loss(coarse_depth*valid_mask, gt_depths*valid_mask)
    loss4 = F.smooth_l1_loss(itg_depth * valid_mask, gt_depths*valid_mask)
    # base2 = loss2.detach()
    loss = loss1 + loss2 + 0.1*loss3 + 0.3*loss4 # math.exp(-0.1*epoch) * loss3
    return loss


def im_gradient_loss(d_batch, n_pixels):
    a = torch.Tensor([[[[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]]]])

    b = torch.Tensor([[[[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]]])
    a = a.to(device)
    b = b.to(device)

    G_x = F.conv2d(d_batch, a, padding=1).to(device)
    G_y = F.conv2d(d_batch, b, padding=1).to(device)

    G = torch.pow(G_x, 2) + torch.pow(G_y, 2)

    return G.view(-1, n_pixels).mean(dim=1).mean()


def scale_inv_loss(preds, actual_depth, n_pixels, valid_mask, grad=False):
    preds[preds <= 0] = 1e-5
    d = (torch.log(preds) - torch.log(actual_depth)) * valid_mask

    term_1 = torch.pow(d.view(-1, n_pixels), 2).mean(dim=1).mean()  # pixel wise mean, then batch mean
    term_2 = (torch.pow(d.view(-1, n_pixels).sum(dim=1), 2) / (2 * (n_pixels ** 2))).mean()
    if grad:
        grad_loss_term = im_gradient_loss(d, n_pixels)
        return term_1 - term_2 + grad_loss_term
    else:
        return term_1 - term_2, d.view(-1, n_pixels).sum(dim=1) / valid_mask.view(-1, n_pixels).sum(dim=1)


def total_scale_inv_loss(pred_depths, cfds, targets, *args):
    gt_depths, valid_mask = targets
    n_pixels = gt_depths.shape[2] * gt_depths.shape[3]
    valid_mask = valid_mask.float()
    const_pred_depths = args[0]
    scale_factor = args[1]
    # epoch = args[2]
    # itg_depth = args[3]

    gt_depths[gt_depths == 0] = 1e-5

    loss1, est_scales = scale_inv_loss(pred_depths, gt_depths, n_pixels, valid_mask) #, grad=True)
    est_scales = est_scales.detach()
    error = torch.abs(gt_depths - const_pred_depths * torch.exp(-est_scales.view(-1, 1, 1, 1)))
    mask = error > 1.0 * scale_factor
    error[mask] = gt_depths[mask]
    gt_cfds = error / gt_depths
    gt_cfds[gt_cfds > 1] = 1
    gt_cfds = 1 - gt_cfds
    loss2 = F.mse_loss(cfds*valid_mask, gt_cfds*valid_mask)

    # loss3 = scale_inv_loss(coarse_pred_depths, gt_depths, n_pixels, valid_mask)
    # loss4 = scale_inv_loss(itg_depth, gt_depths, n_pixels, valid_mask)
    #loss = loss1 + loss2 # + 0.1*loss3 + 0.3*loss4 # math.exp(-0.1*epoch) * loss3
    return loss1 + loss2
