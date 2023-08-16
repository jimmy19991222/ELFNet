from typing import Iterable

import torch
from tqdm import tqdm

from utilities.foward_pass import forward_pass, write_summary
from utilities.misc import save_and_clear
from utilities.summary_logger import TensorboardSummary


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, device: torch.device,
             epoch: int, summary: TensorboardSummary, save_output: bool):
    model.eval()
    criterion.eval()

    # initialize stats
    eval_stats = {'l1_combine': 0.0, 'l1_pcw': 0.0, 'l1': 0.0, 'occ_be': 0.0, 'l1_raw': 0.0, 'iou': 0.0, 'rr': 0.0,
                  'epe': 0.0, 'error_px': 0.0, 'total_px': 0.0, 'epe_pcw': 0.0, 'error_px_pcw': 0.0, 'total_px_pcw': 0.0,
                  'epe_combine': 0.0, 'error_px_combine': 0.0, 'total_px_combine': 0.0,
                  'D1': 0.0, 'D1_pcw': 0.0, 'D1_combine': 0.0, 'aleatoric': 0.0, 'epistemic': 0.0}
    # config text logger
    logger = summary.config_logger(epoch)
    # init output file
    if save_output:
        output_idx = 0
        output_file = {'left': [], 'right': [], 'disp': [],
                       'disp_pred': [], 'occ_mask': [], 'occ_pred': [], 'la': [], 'alpha': [], 'beta': []}

    tbar = tqdm(data_loader)
    valid_samples = len(tbar)
    for idx, data in enumerate(tbar):
        # if (idx == 10): break
        # forward pass
        outputs, losses, sampled_disp = forward_pass(
            model, data, device, criterion, eval_stats, idx, logger, 'eval')

        if losses is None:
            valid_samples -= 1
            continue
        # clear cache
        torch.cuda.empty_cache()

        # save output
        if save_output:
            output_file['left'].append(data['left'][0])
            output_file['right'].append(data['right'][0])
            output_file['disp'].append(data['disp'][0])
            output_file['occ_mask'].append(data['occ_mask'][0].cpu())
            output_file['disp_pred'].append(outputs['disp_pred'].data[0].cpu())
            output_file['occ_pred'].append(outputs['occ_pred'].data[0].cpu())

            output_file['la'].append(outputs['la'].data[0].cpu())
            output_file['alpha'].append(outputs['alpha'].data[0].cpu())
            output_file['beta'].append(outputs['beta'].data[0].cpu())

            output_file['disp_pcw'].append(
                outputs['disp_pcw'].data[0].cpu())
            output_file['la_pcw'].append(outputs['la_pcw'].data[0].cpu())
            output_file['alpha_pcw'].append(
                outputs['alpha_pcw'].data[0].cpu())
            output_file['beta_pcw'].append(
                outputs['beta_pcw'].data[0].cpu())

            output_file['disp_combine'].append(
                outputs['disp_combine'].data[0].cpu())
            output_file['la_combine'].append(
                outputs['la_combine'].data[0].cpu())
            output_file['alpha_combine'].append(
                outputs['alpha_combine'].data[0].cpu())
            output_file['beta_combine'].append(
                outputs['beta_combine'].data[0].cpu())

            output_file['aleatoric'].append(
                outputs['aleatoric'].data[0].cpu())
            output_file['epistemic'].append(
                outputs['epistemic'].data[0].cpu())

            # save to file
            if len(output_file['left']) >= 50:
                output_idx = save_and_clear(output_idx, output_file)

    # save to file
    if save_output:
        save_and_clear(output_idx, output_file)

    # compute avg
    eval_stats['epe'] = eval_stats['epe'] / valid_samples
    eval_stats['D1'] = eval_stats['D1'] / valid_samples

    eval_stats['iou'] = eval_stats['iou'] / valid_samples
    eval_stats['px_error_rate'] = eval_stats['error_px'] / \
        eval_stats['total_px']

    eval_stats['aleatoric'] = outputs['aleatoric'].mean()
    eval_stats['epistemic'] = outputs['epistemic'].mean()

    eval_stats['epe_pcw'] = eval_stats['epe_pcw'] / valid_samples
    eval_stats['D1_pcw'] = eval_stats['D1_pcw'] / valid_samples
    eval_stats['px_error_rate_pcw'] = eval_stats['error_px_pcw'] / \
        eval_stats['total_px_pcw']

    eval_stats['epe_combine'] = eval_stats['epe_combine'] / valid_samples
    eval_stats['D1_combine'] = eval_stats['D1_combine'] / valid_samples
    eval_stats['px_error_rate_combine'] = eval_stats['error_px_combine'] / \
        eval_stats['total_px_combine']

    # write to tensorboard
    write_summary(eval_stats, summary, epoch, 'eval')

    # log to text

    logger.info('Epoch %d, epe %.4f, iou %.4f, px error %.4f \n epe_pcw %.4f, px error_pcw %.4f,\n epe_combine %.4f px error_combine %.4f D1 %.4f D1_pcw %.4f D1_combine %.4f aleatoric_uncertainty %.4f epistemic_uncertainty %.4f' %
                (epoch, eval_stats['epe'], eval_stats['iou'], eval_stats['px_error_rate'], eval_stats['epe_pcw'], eval_stats['px_error_rate_pcw'], eval_stats['epe_combine'], eval_stats['px_error_rate_combine'], eval_stats['D1'], eval_stats['D1_pcw'], eval_stats['D1_combine'], eval_stats['aleatoric'], eval_stats['epistemic']))
    print('Epoch %d, epe %.4f, iou %.4f, px error %.4f \n epe_pcw %.4f, px error_pcw %.4f,\n epe_combine %.4f px error_combine %.4f D1 %.4f D1_pcw %.4f D1_combine %.4f aleatoric_uncertainty %.4f epistemic_uncertainty %.4f' %
          (epoch, eval_stats['epe'], eval_stats['iou'], eval_stats['px_error_rate'], eval_stats['epe_pcw'], eval_stats['px_error_rate_pcw'], eval_stats['epe_combine'], eval_stats['px_error_rate_combine'], eval_stats['D1'], eval_stats['D1_pcw'], eval_stats['D1_combine'], eval_stats['aleatoric'], eval_stats['epistemic']))

    return eval_stats
