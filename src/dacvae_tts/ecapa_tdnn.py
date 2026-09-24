# ECAPA-TDNN speaker-verification head of the seed-tts-eval SIM model.
#
# Adapted from microsoft/UniSpeech, downstreams/speaker_verification/models/ecapa_tdnn.py
# (https://github.com/microsoft/UniSpeech), which states "part of the code is borrowed from
# https://github.com/lawlict/ECAPA-TDNN". UniSpeech is licensed under Creative Commons
# Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0); the full text is in third_party/unispeech/LICENSE.
# This file, as an adaptation of that work, is distributed under the same license (CC BY-SA 3.0),
# independently of the rest of this repository.
#
# Changes from the original (module and parameter names are kept so the published checkpoint
# `wavlm_large_finetune.pth` loads unchanged):
# - the self-supervised feature extractor is passed in (`feature_extract=`) instead of being created with
#   `torch.hub.load` inside the constructor; see dacvae_tts.sim_o for the WavLM-Large backbones;
# - the fbank/MFCC front-ends, `update_extract` fine-tuning and the fp32-attention switches were removed
#   (evaluation only: the extractor is always frozen);
# - formatting and comments.
"""ECAPA-TDNN over softmax-weighted self-supervised hidden states (UniSpeech `ECAPA_TDNN_SMALL`)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Res2Conv1dReluBn(nn.Module):
    """Res2Net-style multi-scale 1-D convolution (conv -> relu -> bn per split); in == out channels."""

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList(
            [nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
             for _ in range(self.nums)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    """Squeeze-and-excitation over time-averaged channels."""

    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut else x
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    """Attention-weighted mean and standard deviation over time."""

    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att
        # Conv1d with kernel 1 instead of Linear: no transposes needed.
        self.linear1 = nn.Conv1d(in_dim * 3 if global_context_att else in_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x):
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x
        alpha = torch.tanh(self.linear1(x_in))  # upstream: ReLU here fails to converge
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN on a frozen self-supervised extractor.

    `feature_extract` follows the s3prl upstream interface: called with a list of 1-D 16 kHz waveforms it returns a
    dict whose `feature_selection` entry is the list of hidden states (25 for WavLM-Large: the transformer input and
    the 24 layer outputs). They are mixed with softmax(`feature_weight`), instance-normalized and fed to the head.
    """

    def __init__(self, feature_extract, feat_dim=1024, channels=512, emb_dim=192, global_context_att=False,
                 sr=16000, feature_selection="hidden_states"):
        super().__init__()
        self.feature_selection = feature_selection
        self.sr = sr
        self.feature_extract = feature_extract
        for param in self.feature_extract.parameters():
            param.requires_grad = False
        self.feat_num = self.get_feat_num()
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))
        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.channels = [channels] * 4 + [1536]
        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(self.channels[0], self.channels[1], kernel_size=3, stride=1, padding=2,
                                   dilation=2, scale=8, se_bottleneck_dim=128)
        self.layer3 = SE_Res2Block(self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=3,
                                   dilation=3, scale=8, se_bottleneck_dim=128)
        self.layer4 = SE_Res2Block(self.channels[2], self.channels[3], kernel_size=3, stride=1, padding=4,
                                   dilation=4, scale=8, se_bottleneck_dim=128)
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(self.channels[-1], attention_channels=128,
                                          global_context_att=global_context_att)
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def get_feat_num(self):
        """Number of hidden states the extractor returns, found by running it once on 1 s of noise."""
        self.feature_extract.eval()
        parameter = next(self.feature_extract.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        with torch.no_grad():
            features = self.feature_extract([torch.randn(self.sr, device=device)])
        selected = features[self.feature_selection]
        return len(selected) if isinstance(selected, (list, tuple)) else 1

    def get_feat(self, x):
        with torch.no_grad():
            x = self.feature_extract([sample for sample in x])
        x = x[self.feature_selection]
        x = torch.stack(x, dim=0) if isinstance(x, (list, tuple)) else x.unsqueeze(0)
        norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = (norm_weights * x).sum(dim=0)
        x = torch.transpose(x, 1, 2) + 1e-6
        return self.instance_norm(x)

    def forward(self, x):
        x = self.get_feat(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        return self.linear(out)


def ECAPA_TDNN_SMALL(feature_extract, feat_dim, emb_dim=256, sr=16000, feature_selection="hidden_states"):
    return ECAPA_TDNN(feature_extract, feat_dim=feat_dim, channels=512, emb_dim=emb_dim, sr=sr,
                      feature_selection=feature_selection)
