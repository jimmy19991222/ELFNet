import math
import sys
from typing import Iterable

import torch
from tqdm import tqdm

from utilities.foward_pass import forward_pass, write_summary
from utilities.summary_logger import TensorboardSummary

# from torch.cuda.amp import autocast as autocast, GradScaler


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    criterion: torch.nn.Module, device: torch.device, epoch: int, summary: TensorboardSummary,
                    max_norm: float = 0):
    """
    train model for 1 epoch
    """
    model.train()
    criterion.train()

    # initialize stats

    train_stats = {'l1_combine': 0.0, 'l1_pcw': 0.0, 'l1': 0.0, 'occ_be': 0.0, 'l1_raw': 0.0, 'iou': 0.0, 'rr': 0.0,
                   'epe': 0.0, 'error_px': 0.0, 'total_px': 0.0, 'epe_pcw': 0.0, 'error_px_pcw': 0.0, 'total_px_pcw': 0.0,
                   'epe_combine': 0.0, 'error_px_combine': 0.0, 'total_px_combine': 0.0, 'D1': 0.0, 'D1_pcw': 0.0, 'D1_combine': 0.0}

    # scaler = GradScaler()

    tbar = tqdm(data_loader)
    for idx, data in enumerate(tbar):

        # with autocast():
        #     # forward pass
        #     _, losses, sampled_disp = forward_pass(
        #         model, data, device, criterion, train_stats)

        # forward pass
        _, losses, sampled_disp = forward_pass(
            model, data, device, criterion, train_stats)
            
        if losses is None:
            continue

        # terminate training if exploded
        if not math.isfinite(losses['aggregated'].item()):
            print("Loss is {}, stopping training".format(
                losses['aggregated'].item()))
            sys.exit(1)

        # backprop
        optimizer.zero_grad()

        losses['aggregated'].backward()
        # scaler.scale(losses['aggregated']).backward()

        # clip norm
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # step optimizer
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()

        print('pixel_error', losses['error_px'].item(
        ) / losses['total_px'])

        print('pixel_error_pcw',
              losses['error_px_pcw'].item() / losses['total_px_pcw'])
        print('pixel_error_combine',
              losses['error_px_combine'].item() / losses['total_px_combine'])

        # clear cache
        torch.cuda.empty_cache()

    # compute avg
    train_stats['px_error_rate'] = train_stats['error_px_combine'] / \
        train_stats['total_px_combine']

    # log to tensorboard
    write_summary(train_stats, summary, epoch, 'train')

    print('Training loss', train_stats['l1_combine'],
          'pixel error rate combine', train_stats['px_error_rate'])

    return
