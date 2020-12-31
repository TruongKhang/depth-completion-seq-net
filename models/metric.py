import torch
import torch.nn.functional as F


def mae(outputs, target, scale_factor=1):
    # val_pixels = (target > 0.1) #.float() #.cuda()
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    # gt_depth *= scale_factor
    # outputs *= scale_factor
    err = torch.abs(gt_depth * val_pixels * scale_factor - outputs * val_pixels * scale_factor)
    loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
    cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
    return torch.mean(loss / cnt)


def rmse(outputs, target, scale_factor=1):
    # val_pixels = (target > 0.1) #.float() #.cuda()
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    # gt_depth *= scale_factor
    # outputs *= scale_factor
    err = (gt_depth * val_pixels * scale_factor - outputs * val_pixels * scale_factor) ** 2
    loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
    cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
    return torch.mean(torch.sqrt(loss / cnt))


def rmse_log(outputs, target, scale_factor=1):
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    outputs[outputs <= 0] = 1e-5
    gt_depth[gt_depth <= 0] = 1e-5
    err = torch.log(outputs * scale_factor) - torch.log(gt_depth * scale_factor)
    err = (err * val_pixels) ** 2
    loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
    cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
    return torch.mean(loss / cnt)


def scale_inv_error(preds, target, scale_factor=1):
    actual_depth, valid_mask = target
    valid_mask = valid_mask.float()
    preds[preds <= 0] = 1e-5
    actual_depth[actual_depth <= 0] = 1e-5
    n_pixels = actual_depth.size(2) * actual_depth.size(3)
    d = (torch.log(preds*scale_factor) - torch.log(actual_depth*scale_factor)) * valid_mask
    cnt = torch.sum(valid_mask.view(-1, n_pixels), 1)
    term_1 = torch.pow(d.view(-1, n_pixels), 2).sum(dim=1)  # pixel wise mean, then batch mean
    term_1 /= cnt
    term_2 = torch.pow(d.view(-1, n_pixels).sum(dim=1), 2)
    term_2 /= torch.pow(cnt, 2)
    return torch.mean(term_1 - term_2)


def mre(outputs, target, scale_factor=1):
    # val_pixels = (target > 0.1)# .float()# .cuda()
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    # gt_depth *= scale_factor
    # outputs *= scale_factor
    err = torch.abs(gt_depth * val_pixels * scale_factor - outputs * val_pixels * scale_factor)
    r = err / (gt_depth * val_pixels * scale_factor + 1e-6)
    cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
    mre = torch.sum(r.view(r.size(0), 1, -1), -1, keepdim=True) / cnt
    return torch.mean(mre)


def deltas(outputs, target, i, scale_factor=1):
    # val_pixels = (target > 0.1).float().cuda()
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    scaled_gt_depth = gt_depth * scale_factor
    scaled_outputs = outputs * scale_factor
    rel = torch.max((scaled_gt_depth * val_pixels) / (scaled_outputs * val_pixels + 1e-3),
                    (scaled_outputs * val_pixels) / (scaled_gt_depth * val_pixels))

    cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)

    def del_i(val):
        r = (rel < 1.01 ** val).float()
        delta = torch.sum(r.view(r.size(0), 1, -1), -1, keepdim=True) / cnt
        return torch.mean(delta)

    return del_i(i) #, del_i(2), del_i(3)


def huber(outputs, target, delta=5):
    l1_loss = F.l1_loss(outputs, target, reduce=False)
    mse_loss = F.mse_loss(outputs, target, reduce=False)

    mask = (l1_loss < delta).float()

    loss = (0.5 * mse_loss) * mask + delta * (l1_loss - 0.5 * delta) * (1 - mask)

    return torch.mean(loss)



