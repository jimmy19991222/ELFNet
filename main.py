import argparse
import os
import random

import numpy as np
import torch

from dataset import build_data_loader
from module.elfnet import ELFNet
from utilities.checkpoint_saver import Saver
from utilities.eval import evaluate
from utilities.summary_logger import TensorboardSummary
from utilities.train import train_one_epoch
from utilities.foward_pass import set_downsample
from module.loss import build_criterion


def get_args_parser():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser('ELFNet', add_help=False)

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--ft', action='store_true',
                        help='load model from checkpoint, but discard optimizer state')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--checkpoint', type=str, default='dev',
                        help='checkpoint name for current experiment')
    parser.add_argument('--pre_train', action='store_true')
    parser.add_argument('--downsample', default=3, type=int,
                        help='Ratio to downsample width/height')

    # * Transformer-based part
    parser.add_argument('--lr_sttr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-4, type=float)
    parser.add_argument('--lr_regression', default=2e-4, type=float)
    parser.add_argument('--channel_dim', default=128, type=int,
                        help="Size of the embeddings (dimension of the transformer)")

    # * Positional Encoding
    parser.add_argument('--position_encoding', default='sine1d_rel', type=str, choices=('sine1d_rel', 'none'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--num_attn_layers', default=6, type=int,
                        help="Number of attention layers in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # * Regression Head
    parser.add_argument('--regression_head', default='ot', type=str, choices=('softmax', 'ot'),
                        help='Normalization to be used')
    parser.add_argument('--context_adjustment_layer',
                        default='cal', choices=['cal', 'none'], type=str)
    parser.add_argument('--cal_num_blocks', default=8, type=int)
    parser.add_argument('--cal_feat_dim', default=16, type=int)
    parser.add_argument('--cal_expansion_ratio', default=4, type=int)

    # * Dataset parameters
    parser.add_argument('--dataset', default='sceneflow',
                        type=str, help='dataset to train/eval on')
    parser.add_argument('--dataset_directory', default='',
                        type=str, help='directory to dataset')
    parser.add_argument('--validation', default='validation', type=str, choices={'validation', 'validation_all'},
                        help='If we validate on all provided training images')

    # * Loss
    parser.add_argument('--px_error_threshold', type=int, default=3,
                        help='Number of pixels for error computation (default 3 px)')
    parser.add_argument('--loss_weight', type=str, default='rr:1.0, l1_raw:1.0, l1:1.0, occ_be:1.0, l1_pcw:1.0, l1_combine:2.0',
                        help='Weight for losses')
    parser.add_argument('--validation_max_disp', type=int, default=192)

    # uncertainty loss
    parser.add_argument('--weight_reg', default=0.05, type=float)

    # * Cost-volume-based part
    parser.add_argument('--lr_pcw', default=2e-3, type=float)
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maximum disparity')
    parser.add_argument('--pcnet', action='store_true',
                        help='use pcnet instead of pcwnet')
    parser.add_argument('--wo_combine', action='store_true',
                        help='')

    parser.add_argument('--name', default='demo',
                        type=str, help='expriment name')
    parser.add_argument('--lrepochs', type=str, default='4,8,10,12',
                        help='the epochs to decay lr: the downscale rate')

    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    return parser


def save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, best, amp=None):
    """
    Save current state of training
    """

    # save model
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_pred': prev_best
    }
    if amp is not None:
        checkpoint['amp'] = amp.state_dict()
    if best:
        checkpoint_saver.save_checkpoint(
            checkpoint, 'model.pth.tar', write_best=False)
    else:
        checkpoint_saver.save_checkpoint(
            checkpoint, 'epoch_' + str(epoch) + '_model.pth.tar', write_best=False)


def print_param(model):
    """
    print number of parameters in the model
    """

    n_parameters = sum(p.numel() for n, p in model.named_parameters(
    ) if 'backbone' in n and p.requires_grad)
    print('number of params in backbone:', f'{n_parameters:,}')
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if
                       'transformer' in n and 'regression' not in n and p.requires_grad)
    print('number of params in transformer:', f'{n_parameters:,}')
    n_parameters = sum(p.numel() for n, p in model.named_parameters(
    ) if 'tokenizer' in n and p.requires_grad)

    print('number of params in tokenizer:', f'{n_parameters:,}')
    n_parameters = sum(p.numel() for n, p in model.named_parameters(
    ) if 'regression' in n and p.requires_grad)
    print('number of params in regression:', f'{n_parameters:,}')

    n_parameters = sum(p.numel() for n, p in model.named_parameters(
    ) if 'pcw' in n and p.requires_grad)
    print('number of params in pcw:', f'{n_parameters:,}')


def main(args):
    # get device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = ELFNet(args)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    print_param(model)

    # set learning rate
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if
                    "backbone" not in n and "regression" not in n and 'pcw' not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if "regression" in n and p.requires_grad],
            "lr": args.lr_regression,
        },
        {
            "params": [p for n, p in model.named_parameters() if "pcw" in n and p.requires_grad],
            "lr": args.lr_pcw,
        },
    ]

    # define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr_sttr, weight_decay=args.weight_decay)
    downscale_epochs = [int(eid_str) for eid_str in args.lrepochs.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, downscale_epochs, gamma=0.5)

    # load checkpoint if provided
    prev_best = np.inf
    if args.resume != '':
        if not os.path.isfile(args.resume):
            raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
        checkpoint = torch.load(args.resume)

        pretrained_dict = checkpoint['state_dict']
        missing, unexpected = model.load_state_dict(
            pretrained_dict, strict=False)
        # check missing and unexpected keys
        if len(missing) > 0:
            print("Missing keys: ", ','.join(missing))
            raise Exception("Missing keys.")
        unexpected_filtered = [k for k in unexpected if
                               'running_mean' not in k and 'running_var' not in k]  # skip bn params
        if len(unexpected_filtered) > 0:
            print("Unexpected keys: ", ','.join(unexpected_filtered))
            raise Exception("Unexpected keys.")
        print("Pre-trained model successfully loaded.")

        # if not ft/eval, load states for optimizer, lr_scheduler, amp and prev best
        if not (args.ft or args.eval):
            if len(unexpected) > 0:  # loaded checkpoint has bn parameters, legacy resume, skip loading
                raise Exception("Resuming legacy model with BN parameters. Not possible due to BN param change. " +
                                "Do you want to finetune? If so, check your arguments.")
            else:
                args.start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                prev_best = checkpoint['best_pred']
                print(
                    "Pre-trained optimizer, lr scheduler and stats successfully loaded.")

    # initiate saver and logger
    checkpoint_saver = Saver(args)
    summary_writer = TensorboardSummary(checkpoint_saver.experiment_dir)

    # build dataloader
    data_loader_train, data_loader_val, _ = build_data_loader(args)

    # build loss criterion
    criterion = build_criterion(args)

    # set downsample rate
    set_downsample(args)

    # eval
    if args.eval:
        print("Start evaluation")
        evaluate(model, criterion, data_loader_val,
                 device, 0, summary_writer, False)
        return

    # train
    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        # train
        print("Epoch: %d" % epoch)
        train_one_epoch(model, data_loader_train, optimizer, criterion, device, epoch, summary_writer,
                        args.clip_max_norm)

        if not args.pre_train:
            lr_scheduler.step()
            print("current learning rate", lr_scheduler.get_lr())

        # empty cache
        torch.cuda.empty_cache()

        # save if pretrain, save every 50 epochs
        if args.pre_train or epoch % 50 == 0:
            save_checkpoint(epoch, model, optimizer, lr_scheduler,
                            prev_best, checkpoint_saver, False)

        # validate
        eval_stats = evaluate(model, criterion, data_loader_val,
                              device, epoch, summary_writer, False)
        # save if best
        if prev_best > eval_stats['epe_combine'] and 0.5 > eval_stats['px_error_rate_combine']:
            save_checkpoint(epoch, model, optimizer, lr_scheduler,
                            prev_best, checkpoint_saver, True)

    # save final model
    save_checkpoint(epoch, model, optimizer,
                    prev_best, checkpoint_saver, False)

    return


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        'ELFNet training and evaluation script', parents=[get_args_parser()])
    args_ = ap.parse_args()
    main(args_)
