import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from module.feat_extractor_backbone import build_backbone
from module.feat_extractor_tokenizer import build_tokenizer
from module.pos_encoder import build_position_encoding
from module.regression_head import build_regression_head
from module.transformer import build_transformer
from utilities.misc import batched_index_select, NestedTensor


class STTR(nn.Module):
    """
    STTR: it consists of
        - backbone: contracting path of feature descriptor
        - tokenizer: expanding path of feature descriptor
        - pos_encoder: generates relative sine pos encoding
        - transformer: computes self and cross attention
        - regression_head: regresses disparity and occlusion, including optimal transport
    """

    def __init__(self, args):
        super(STTR, self).__init__()
        layer_channel = [64, 128, 128]

        self.backbone = build_backbone(args)
        self.tokenizer = build_tokenizer(args, layer_channel)
        self.pos_encoder = build_position_encoding(args)
        self.transformer = build_transformer(args)
        self.regression_head = build_regression_head(args)

        # uncertainty
        feature_dim = 32

        self.input_dim = 256
        self.uncertainty_head_1 = nn.Sequential(
            weight_norm(nn.Conv2d(self.input_dim, feature_dim,
                        kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim,
                        kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(feature_dim, feature_dim,
                        kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim,
                        kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1),
        )

        self.uncertainty_head_2 = nn.Sequential(
            weight_norm(nn.Conv2d(self.input_dim, feature_dim,
                        kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim,
                        kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(feature_dim, feature_dim,
                        kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim,
                        kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1),
        )

        self.uncertainty_head_3 = nn.Sequential(
            weight_norm(nn.Conv2d(self.input_dim, feature_dim,
                        kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim,
                        kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(feature_dim, feature_dim,
                        kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim,
                        kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1),
        )

        self._reset_parameters()
        self._disable_batchnorm_tracking()
        self._relu_inplace()

    def _reset_parameters(self):
        """
        xavier initialize all params
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def _disable_batchnorm_tracking(self):
        """
        disable Batchnorm tracking stats to reduce dependency on dataset (this acts as InstanceNorm with affine when batch size is 1)
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def _relu_inplace(self):
        """
        make all ReLU inplace
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.inplace = True

    def evidence(self, x):
        # return tf.exp(x)
        return F.softplus(x)

    def get_uncertainty(self, logv, logalpha, logbeta):
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return v, alpha, beta

    def forward(self, x: NestedTensor):
        """
        :param x: input data
        :return:
            a dictionary object with keys
            - "disp_pred" [N,H,W]: predicted disparity
            - "occ_pred" [N,H,W]: predicted occlusion mask
            - "disp_pred_low_res" [N,H//s,W//s]: predicted low res (raw) disparity
        """
        bs, _, h, w = x.left.size()

        # extract features
        feat = self.backbone(x)  # concatenate left and right along the dim=0
        tokens = self.tokenizer(feat).cuda()  # 2NxCxHxW
        pos_enc = self.pos_encoder(x).cuda()  # NxCxHx2W-1

        # separate left and right
        feat_left = tokens[:bs]
        feat_right = tokens[bs:]  # NxCxHxW

        # downsample
        if x.sampled_cols is not None:
            feat_left = batched_index_select(
                feat_left, 3, x.sampled_cols.cuda())
            feat_right = batched_index_select(
                feat_right, 3, x.sampled_cols.cuda())
        if x.sampled_rows is not None:
            feat_left = batched_index_select(
                feat_left, 2, x.sampled_rows.cuda())
            feat_right = batched_index_select(
                feat_right, 2, x.sampled_rows.cuda())

        # transformer
        attn_weight, feat = self.transformer(feat_left, feat_right, pos_enc)
        
        # regress disparity and occlusion
        output = self.regression_head(attn_weight, x)

        feat_left, feat_right = torch.chunk(feat, 2, dim=1)
        _, hn, c = feat_left.shape

        feat_left = feat_left.permute(2, 1, 0).reshape(bs, c, hn//bs, -1)
        feat_right = feat_right.permute(2, 1, 0).reshape(bs, c, hn//bs, -1)

        # uncertainty estimation
        feat_cat = torch.cat((feat_left, feat_right), dim=1)

        logla, logalpha, logbeta = self.uncertainty_head_1(
            feat_cat), self.uncertainty_head_2(feat_cat), self.uncertainty_head_3(feat_cat)

        logla = F.interpolate(logla, size=(h, w), mode='nearest').squeeze(1)
        logalpha = F.interpolate(logalpha, size=(
            h, w), mode='nearest').squeeze(1)
        logbeta = F.interpolate(logbeta, size=(
            h, w), mode='nearest').squeeze(1)
        # print("logla: ", logla.shape)
        la, alpha, beta = self.get_uncertainty(logla, logalpha, logbeta)

        output['la'] = la
        output['alpha'] = alpha
        output['beta'] = beta

        return output
