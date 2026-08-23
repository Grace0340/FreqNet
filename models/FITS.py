import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = bool(getattr(configs, 'individual', False))
        self.channels = configs.enc_in

        in_bins = self.seq_len // 2 + 1
        self.dominance_freq = min(int(getattr(configs, 'cut_freq', self.seq_len // 4 + 1)), in_bins)
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len
        self.out_dim = int(self.dominance_freq * self.length_ratio)

        if self.individual:
            self.freq_upsampler = nn.ModuleList(
                [nn.Linear(self.dominance_freq, self.out_dim, dtype=torch.cfloat)
                 for _ in range(self.channels)])
        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, self.out_dim, dtype=torch.cfloat)

    def forecast(self, x):
        x_mean = x.mean(1, keepdim=True)
        x = x - x_mean
        x_var = x.var(1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        spec = torch.fft.rfft(x, dim=1)
        spec = spec[:, :self.dominance_freq, :]

        if self.individual:
            out = torch.zeros([x.size(0), self.out_dim, self.channels],
                              dtype=spec.dtype, device=x.device)
            for i in range(self.channels):
                out[:, :, i] = self.freq_upsampler[i](spec[:, :, i])
        else:
            out = self.freq_upsampler(spec.permute(0, 2, 1)).permute(0, 2, 1)

        full_len = (self.seq_len + self.pred_len) // 2 + 1
        full = torch.zeros([x.size(0), full_len, self.channels],
                           dtype=out.dtype, device=x.device)
        L = min(self.out_dim, full_len)
        full[:, :L, :] = out[:, :L, :]

        y = torch.fft.irfft(full, n=self.seq_len + self.pred_len, dim=1)
        y = y * self.length_ratio
        y = y * torch.sqrt(x_var) + x_mean
        return y[:, -self.pred_len:, :]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc)
        return None
