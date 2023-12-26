from collections import OrderedDict

import torch
from torch import nn, Tensor
import numpy as np

from utilities.misc import batched_index_select, NestedTensor

from utilities.metrics import D1_metric, Thres_metric, EPE_metric


class Criterion(nn.Module):
    """
    Compute loss and evaluation metrics
    """

    def __init__(self, threshold: int = 3, validation_max_disp: int = -1, loss_weight: list = None, weight_reg: float = 0.05):
        super(Criterion, self).__init__()

        if loss_weight is None:
            loss_weight = {}

        self.px_threshold = threshold
        self.validation_max_disp = validation_max_disp
        self.weights = loss_weight

        self.weight_reg = weight_reg

        self.l1_criterion = nn.SmoothL1Loss()
        self.epe_criterion = nn.L1Loss()

    @torch.no_grad()
    def calc_px_error(self, pred: Tensor, disp: Tensor, loss_dict: dict, invalid_mask: Tensor, name=''):
        """
        compute px error

        :param pred: disparity prediction [N,H,W]
        :param disp: ground truth disparity [N,H,W]
        :param loss_dict: dictionary of losses
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        """

        # computing threshold-px error
        loss_dict['error_px' +
                  name] = Thres_metric(pred, disp, ~invalid_mask, self.px_threshold)
        loss_dict['total_px'+name] = torch.sum(~invalid_mask).item()
        return

    @torch.no_grad()
    def compute_epe(self, pred: Tensor, disp: Tensor, loss_dict: dict, invalid_mask: Tensor, name=''):
        """
        compute EPE

        :param pred: disparity prediction [N,H,W]
        :param disp: ground truth disparity [N,H,W]
        :param loss_dict: dictionary of losses
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        """
        loss_dict['epe'+name] = EPE_metric(pred, disp, ~invalid_mask)
        return

    @torch.no_grad()
    def compute_D1(self, pred: Tensor, disp: Tensor, loss_dict: dict, invalid_mask: Tensor, name=''):
        loss_dict['D1'+name] = D1_metric(pred, disp, ~invalid_mask)
        return

    @torch.no_grad()
    def compute_iou(self, pred: Tensor, occ_mask: Tensor, loss_dict: dict, invalid_mask: Tensor):
        """
        compute IOU on occlusion

        :param pred: occlusion prediction [N,H,W]
        :param occ_mask: ground truth occlusion mask [N,H,W]
        :param loss_dict: dictionary of losses
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        """
        # threshold
        pred_mask = pred > 0.5

        # iou for occluded region
        inter_occ = torch.logical_and(pred_mask, occ_mask).sum()
        union_occ = torch.logical_or(torch.logical_and(
            pred_mask, ~invalid_mask), occ_mask).sum()

        # iou for non-occluded region
        inter_noc = torch.logical_and(~pred_mask, ~invalid_mask).sum()
        union_noc = torch.logical_or(torch.logical_and(
            ~pred_mask, occ_mask), ~invalid_mask).sum()

        # aggregate
        loss_dict['iou'] = (inter_occ + inter_noc).float() / \
            (union_occ + union_noc)

        return

    def criterion_uncertainty(self, u, la, alpha, beta, y, mask):
        # our loss function
        om = 2 * beta * (1 + la)
        # len(u): size
        loss = torch.sum(
            (0.5 * torch.log(np.pi / la) - alpha * torch.log(om) +
             (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om) +
             torch.lgamma(alpha) - torch.lgamma(alpha+0.5))[mask]
        ) / torch.sum(mask == True)

        lossr = self.weight_reg * (torch.sum((torch.abs(u - y) * (2 * la + alpha))
                                             [mask])) / torch.sum(mask == True)
        loss = loss + lossr
        return loss

    def compute_uncertainty_loss(self, pred: Tensor, la: Tensor, alpha: Tensor, beta: Tensor, inputs: NestedTensor, invalid_mask: Tensor, fullres: bool = True):
        """
        compute smooth l1 loss

        :param pred: disparity prediction [N,H,W]
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :param fullres: Boolean indicating if prediction is full resolution
        :return: smooth l1 loss
        """
        disp = inputs.disp
        if not fullres:
            if inputs.sampled_cols is not None:
                if invalid_mask is not None:
                    invalid_mask = batched_index_select(
                        invalid_mask, 2, inputs.sampled_cols)
                disp = batched_index_select(disp, 2, inputs.sampled_cols)
            if inputs.sampled_rows is not None:
                if invalid_mask is not None:
                    invalid_mask = batched_index_select(
                        invalid_mask, 1, inputs.sampled_rows)
                disp = batched_index_select(disp, 1, inputs.sampled_rows)

        # return self.l1_criterion(pred[~invalid_mask], disp[~invalid_mask])
        return self.criterion_uncertainty(pred, la, alpha, beta, disp, ~invalid_mask)

    def compute_rr_loss(self, outputs: dict, inputs: NestedTensor, invalid_mask: Tensor):
        """
        compute rr loss

        :param outputs: dictionary, outputs from the network
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :return: rr loss
        """""
        if invalid_mask is not None:
            if inputs.sampled_cols is not None:
                invalid_mask = batched_index_select(
                    invalid_mask, 2, inputs.sampled_cols)
            if inputs.sampled_rows is not None:
                invalid_mask = batched_index_select(
                    invalid_mask, 1, inputs.sampled_rows)

        # compute rr loss in non-occluded region
        gt_response = outputs['gt_response']
        # print(outputs['gt_response'].shape)
        eps = 1e-6
        rr_loss = - torch.log(gt_response + eps)

        if invalid_mask is not None:
            rr_loss = rr_loss[~invalid_mask]

        # if there is occlusion
        try:
            rr_loss_occ_left = - \
                torch.log(outputs['gt_response_occ_left'] + eps)
            # print(rr_loss_occ_left.shape)
            rr_loss = torch.cat([rr_loss, rr_loss_occ_left])
        except KeyError:
            pass
        try:
            rr_loss_occ_right = - \
                torch.log(outputs['gt_response_occ_right'] + eps)
            # print(rr_loss_occ_right.shape)
            rr_loss = torch.cat([rr_loss, rr_loss_occ_right])
        except KeyError:
            pass

        return rr_loss.mean()

    def compute_l1_loss(self, pred: Tensor, inputs: NestedTensor, invalid_mask: Tensor, fullres: bool = True):
        """
        compute smooth l1 loss

        :param pred: disparity prediction [N,H,W]
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :param fullres: Boolean indicating if prediction is full resolution
        :return: smooth l1 loss
        """
        disp = inputs.disp
        if not fullres:
            if inputs.sampled_cols is not None:
                if invalid_mask is not None:
                    invalid_mask = batched_index_select(
                        invalid_mask, 2, inputs.sampled_cols)
                disp = batched_index_select(disp, 2, inputs.sampled_cols)
            if inputs.sampled_rows is not None:
                if invalid_mask is not None:
                    invalid_mask = batched_index_select(
                        invalid_mask, 1, inputs.sampled_rows)
                disp = batched_index_select(disp, 1, inputs.sampled_rows)

        return self.l1_criterion(pred[~invalid_mask], disp[~invalid_mask])

    def compute_entropy_loss(self, occ_pred: Tensor, inputs: NestedTensor, invalid_mask: Tensor):
        """
        compute binary entropy loss on occlusion mask

        :param occ_pred: occlusion prediction, [N,H,W]
        :param inputs: input data
        :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
        :return: binary entropy loss
        """
        eps = 1e-6

        occ_mask = inputs.occ_mask

        entropy_loss_occ = -torch.log(occ_pred[occ_mask] + eps)
        entropy_loss_noc = - torch.log(
            1.0 - occ_pred[~invalid_mask] + eps)  # invalid mask includes both occ and invalid points

        entropy_loss = torch.cat([entropy_loss_occ, entropy_loss_noc])

        return entropy_loss.mean()

    def aggregate_loss(self, loss_dict: dict):
        """
        compute weighted sum of loss

        :param loss_dict: dictionary of losses
        """
        loss = 0.0
        for key in loss_dict:
            loss += loss_dict[key] * self.weights[key]

        loss_dict['aggregated'] = loss
        return

    def forward(self, inputs: NestedTensor, outputs: dict):
        """
        :param inputs: input data
        :param outputs: output from the network, dictionary
        :return: loss dictionary
        """
        loss = {}

        if self.validation_max_disp == -1:
            invalid_mask = inputs.disp <= 0.0
        else:
            invalid_mask = torch.logical_or(
                inputs.disp <= 0.0, inputs.disp >= self.validation_max_disp)

        if torch.all(invalid_mask):
            return None

        loss['rr'] = self.compute_rr_loss(outputs, inputs, invalid_mask)
        loss['l1_raw'] = self.compute_l1_loss(
            outputs['disp_pred_low_res'], inputs, invalid_mask, fullres=False)
        # loss['l1'] = self.compute_l1_loss(outputs['disp_pred'], inputs, invalid_mask)
        loss['l1'] = self.compute_uncertainty_loss(
            outputs['disp_pred'], outputs['la'], outputs['alpha'], outputs['beta'], inputs, invalid_mask)
        loss['l1_pcw'] = self.compute_uncertainty_loss(
            outputs['disp_pcw'], outputs['la_pcw'], outputs['alpha_pcw'], outputs['beta_pcw'], inputs, invalid_mask)
        loss['l1_combine'] = self.compute_uncertainty_loss(
            outputs['disp_combine'], outputs['la_combine'], outputs['alpha_combine'], outputs['beta_combine'], inputs, invalid_mask)
        loss['occ_be'] = self.compute_entropy_loss(
            outputs['occ_pred'], inputs, invalid_mask)

        self.aggregate_loss(loss)

        # for benchmarking
        self.calc_px_error(outputs['disp_pred'],
                           inputs.disp, loss, invalid_mask)
        self.compute_epe(outputs['disp_pred'], inputs.disp, loss, invalid_mask)
        self.compute_D1(outputs['disp_pred'], inputs.disp, loss, invalid_mask)
        self.compute_iou(outputs['occ_pred'],
                         inputs.occ_mask, loss, invalid_mask)

        self.calc_px_error(outputs['disp_pcw'],
                           inputs.disp, loss, invalid_mask, name='_pcw')
        self.compute_epe(outputs['disp_pcw'], inputs.disp,
                         loss, invalid_mask, name='_pcw')
        self.compute_D1(outputs['disp_pcw'], inputs.disp,
                        loss, invalid_mask, name='_pcw')

        self.calc_px_error(
            outputs['disp_combine'], inputs.disp, loss, invalid_mask, name='_combine')
        self.compute_epe(outputs['disp_combine'], inputs.disp,
                         loss, invalid_mask, name='_combine')
        self.compute_D1(outputs['disp_combine'], inputs.disp,
                        loss, invalid_mask, name='_combine')

        return OrderedDict(loss)


def build_criterion(args):
    loss_weight = {}
    for weight in args.loss_weight.split(','):
        k, v = weight.split(':')
        k = k.strip()
        v = float(v)
        loss_weight[k] = v

    return Criterion(args.px_error_threshold, args.validation_max_disp, loss_weight, args.weight_reg)
