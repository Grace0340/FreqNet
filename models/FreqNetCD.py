import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        n_freqs     = getattr(configs, 'freqnet_k', 30)
        hidden      = getattr(configs, 'freqnet_hidden', 64)
        gate_hidden = getattr(configs, 'freqnet_gate_hidden', 32)
        d_attn      = getattr(configs, 'freqnet_dattn', 64)
        heads       = getattr(configs, 'freqnet_heads', 4)
        self.use_cd     = bool(getattr(configs, 'freqnet_cd', True))
        self.reg_lambda = float(getattr(configs, 'freqnet_cd_l1', 5e-3))
        self.use_revin       = bool(getattr(configs, 'freqnet_use_revin', True))
        self.adaptive_fusion = bool(getattr(configs, 'freqnet_adaptive_fusion', True))
        self.branch          = str(getattr(configs, 'freqnet_branch', 'both')).lower()
        assert self.branch in ('both', 'linear', 'freq')

        in_bins  = self.seq_len // 2 + 1
        out_bins = self.pred_len // 2 + 1
        self.k_in  = min(n_freqs, in_bins)
        self.k_out = max(1, min(out_bins, round(self.k_in * out_bins / in_bins)))

        self.bn = nn.BatchNorm1d(2 * self.k_in)
        self.mlp_ci = nn.Sequential(
            nn.Linear(2 * self.k_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * self.k_out))
        self.linear_proj = nn.Linear(self.seq_len, self.pred_len)
        self.gate_base = nn.Sequential(
            nn.Linear(self.k_in + 2, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, 2))
        nn.init.zeros_(self.gate_base[-1].weight)
        nn.init.zeros_(self.gate_base[-1].bias)

        cd_active = self.use_cd and (self.branch == 'both')
        if cd_active:
            self.proj_in = nn.Linear(2 * self.k_in, d_attn)
            self.attn = nn.MultiheadAttention(d_attn, heads, batch_first=True)
            self.ln = nn.LayerNorm(d_attn)
            self.mlp_cd = nn.Sequential(
                nn.Linear(d_attn, hidden), nn.ReLU(),
                nn.Linear(hidden, 2 * self.k_out))
            self.gate_cd = nn.Sequential(
                nn.Linear(self.k_in + 2, gate_hidden), nn.ReLU(),
                nn.Linear(gate_hidden, 1))
            self.gamma = nn.Parameter(torch.zeros(1))
        self._cd_active = cd_active
        self.aux_loss = torch.tensor(0.0)

    def _coeffs_to_time(self, pf):
        shape = pf.shape[:-1]
        pc = torch.complex(pf[..., :self.k_out], pf[..., self.k_out:])
        full = torch.zeros(*shape, self.pred_len // 2 + 1,
                           dtype=torch.complex64, device=pf.device)
        full[..., :self.k_out] = pc
        return torch.fft.irfft(full, n=self.pred_len, dim=-1)

    def forecast(self, x_enc):
        B, L, C = x_enc.shape
        x = x_enc.permute(0, 2, 1)
        if self.use_revin:
            m = x.mean(-1, keepdim=True).detach()
            s = torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5).detach()
        else:
            m = torch.zeros(B, C, 1, device=x.device)
            s = torch.ones(B, C, 1, device=x.device)
        xn = (x - m) / s

        spec = torch.fft.rfft(xn, dim=-1)
        lc = spec[..., :self.k_in]
        feat = torch.cat([lc.real, lc.imag], dim=-1)

        lin = self.linear_proj(xn)
        feat_bn = self.bn(feat.reshape(B * C, -1)).reshape(B, C, -1)
        freq_ci = self._coeffs_to_time(self.mlp_ci(feat_bn))

        e_full = (spec.abs() ** 2).sum(-1) + 1e-8
        mag = lc.abs()
        low_ratio = ((mag ** 2).sum(-1) / e_full).unsqueeze(-1)
        log_e = torch.log(e_full).unsqueeze(-1)
        mag_norm = mag / (mag.sum(-1, keepdim=True) + 1e-8)
        gfeat = torch.cat([mag_norm, low_ratio, log_e], dim=-1)

        if self.branch == 'linear':
            out = lin
        elif self.branch == 'freq':
            out = freq_ci
        else:
            if self.adaptive_fusion:
                wb = torch.softmax(self.gate_base(gfeat), dim=-1)
            else:
                wb = torch.full((B, C, 2), 0.5, device=x.device)
            out = wb[..., 0:1] * lin + wb[..., 1:2] * freq_ci

        if self._cd_active:
            h = self.proj_in(feat)
            h2, _ = self.attn(h, h, h)
            h = self.ln(h + h2)
            freq_cd = self._coeffs_to_time(self.mlp_cd(h))
            g_cd = torch.sigmoid(self.gate_cd(gfeat))
            coef = self.gamma * g_cd
            out = out + coef * freq_cd
            self.aux_loss = self.reg_lambda * coef.abs().mean()
        else:
            self.aux_loss = torch.tensor(0.0, device=x_enc.device)

        out = out * s + m
        return out.permute(0, 2, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        return None
