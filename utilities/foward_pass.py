import torch

from utilities.misc import NestedTensor
# from torch.cuda.amp import autocast as autocast, GradScaler

from dataset.stereo_albumentation import random_crop

downsample = 0


def set_downsample(args):
    global downsample
    downsample = args.downsample


def write_summary(stats, summary, epoch, mode):
    """
    write the current epoch result to tensorboard
    """
    summary.writer.add_scalar(mode + '/rr', stats['rr'], epoch)
    summary.writer.add_scalar(mode + '/l1', stats['l1'], epoch)
    summary.writer.add_scalar(mode + '/l1_pcw', stats['l1_pcw'], epoch)
    summary.writer.add_scalar(
        mode + '/l1_combine', stats['l1_combine'], epoch)
    summary.writer.add_scalar(mode + '/l1_raw', stats['l1_raw'], epoch)
    summary.writer.add_scalar(mode + '/occ_be', stats['occ_be'], epoch)

    summary.writer.add_scalar(mode + '/epe_combine',
                              stats['epe_combine'], epoch)
    summary.writer.add_scalar(mode + '/D1_combine', stats['D1_combine'], epoch)
    summary.writer.add_scalar(mode + '/iou', stats['iou'], epoch)
    summary.writer.add_scalar(
        mode + '/3px_error', stats['px_error_rate'], epoch)


def forward_pass(model, data, device, criterion, stats, idx=0, logger=None, stage='train'):
    """
    forward pass of the model given input
    """
    # read data
    data = random_crop(360, 640, data, stage)
    left, right = data['left'].to(device), data['right'].to(device)
    disp, occ_mask, occ_mask_right = data['disp'].to(device), data['occ_mask'].to(device), \
        data['occ_mask_right'].to(device)

    # if need to downsample, sample with a provided stride
    bs, _, h, w = left.size()
    if downsample <= 0:
        sampled_cols = None
        sampled_rows = None
    else:
        col_offset = int(downsample / 2)
        row_offset = int(downsample / 2)
        sampled_cols = torch.arange(col_offset, w, downsample)[
            None, ].expand(bs, -1).to(device)
        sampled_rows = torch.arange(row_offset, h, downsample)[
            None, ].expand(bs, -1).to(device)

    # build the input
    inputs = NestedTensor(left.cuda(), right.cuda(), sampled_cols=sampled_cols.cuda(), sampled_rows=sampled_rows.cuda(), disp=disp.cuda(),
                          occ_mask=occ_mask.cuda(), occ_mask_right=occ_mask_right.cuda())

    # forward pass
    outputs = model(inputs)
    # compute loss
    losses = criterion(inputs, outputs)

    if losses is None:
        return outputs, losses, disp

    # get the loss
    stats['rr'] += losses['rr'].item()
    stats['l1_combine'] += losses['l1_combine'].item()
    stats['l1_pcw'] += losses['l1_pcw'].item()
    stats['l1_raw'] += losses['l1_raw'].item()
    stats['l1'] += losses['l1'].item()
    stats['occ_be'] += losses['occ_be'].item()

    stats['iou'] += losses['iou'].item()
    stats['epe'] += losses['epe'].item()

    stats['error_px'] += losses['error_px']
    stats['total_px'] += losses['total_px']

    stats['epe_pcw'] += losses['epe_pcw'].item()
    stats['D1_pcw'] += losses['D1_pcw'].item()
    stats['error_px_pcw'] += losses['error_px_pcw']
    stats['total_px_pcw'] += losses['total_px_pcw']

    stats['epe_combine'] += losses['epe_combine'].item()
    stats['D1_combine'] += losses['D1_combine'].item()
    stats['error_px_combine'] += losses['error_px_combine']
    stats['total_px_combine'] += losses['total_px_combine']

    # log for eval only
    if logger is not None:
        logger.info('Index %d, l1_raw %.4f, rr %.4f, l1 %.4f, l1_pcw %.4f, l1_combine %.4f, occ_be %.4f, \n epe %.4f, epe_pcw %.4f, epe_combine: %.4f, px error %.4f, px error_pcw %.4f, px error_combine %.4f, iou %.4f,\n D1 %.4f, D1_pcw %.4f, D1_combine %.4f, aleatoric_uncertainty %.4f, epistemic_uncertainty %.4f' %
                    (idx, losses['l1_raw'].item(), losses['rr'].item(), losses['l1'].item(), losses['l1_pcw'].item(), losses['l1_combine'].item(), losses['occ_be'].item(),
                        losses['epe'].item(), losses['epe_pcw'].item(), losses['epe_combine'].item(), (losses['error_px'] / losses['total_px']
                                                                                                       ), (losses['error_px_pcw'] / losses['total_px_pcw']), (losses['error_px_combine'] / losses['total_px_combine']), losses['iou'].item(),
                        losses['D1'].item(), losses['D1_pcw'].item(), losses['D1_combine'].item(), outputs['aleatoric'].mean(), outputs['epistemic'].mean()))

    return outputs, losses, disp
