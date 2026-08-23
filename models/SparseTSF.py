"""SparseTSF (Lin et al., ICML 2024 Oral) ported to the TSLib interface.

Cross-Period Sparse Forecasting: downsample the sequence by a fixed period,
forecast each phase with a shared linear layer, then upsample. Under the
unified lookback-96 protocol we fix period_len=24, which divides both the
96-step lookback and all horizons {96,192,336,720}.

Official implementation: https://github.com/lss-1138/SparseTSF
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = getattr(configs, "period_len", 24)
        assert self.seq_len % self.period_len == 0, "seq_len % period_len != 0"
        assert self.pred_len % self.period_len == 0, "pred_len % period_len != 0"
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False,
        )
        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

    def forecast(self, x):
        batch_size = x.shape[0]
        # instance-mean normalization, then b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)
        # 1D convolutional aggregation with residual
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(
            -1, self.enc_in, self.seq_len) + x
        # downsample: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        # cross-period sparse forecasting
        y = self.linear(x)  # bc,w,m
        # upsample: bc,w,m -> b,c,pred -> b,pred,c and de-normalize
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        return y.permute(0, 2, 1) + seq_mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc)
        raise NotImplementedError(f"SparseTSF: task {self.task_name} not supported")
