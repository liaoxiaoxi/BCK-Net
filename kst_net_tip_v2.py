import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBNReLU2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBNReLU2d(ch, ch, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class FrameEncoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, out_ch=96):
        super().__init__()
        self.stem = ConvBNReLU2d(in_ch, base_ch, 3, 1, 1)
        self.layer1 = nn.Sequential(
            ConvBNReLU2d(base_ch, base_ch, 3, 2, 1),
            ResBlock2d(base_ch),
        )
        self.layer2 = nn.Sequential(
            ConvBNReLU2d(base_ch, out_ch, 3, 2, 1),
            ResBlock2d(out_ch),
        )

    def forward(self, x):
        return self.layer2(self.layer1(self.stem(x)))


class TemporalRefiner3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B,T,C,H,W]
        x_in = x.permute(0, 2, 1, 3, 4).contiguous()
        x_out = self.block(x_in) + x_in
        return x_out.permute(0, 2, 1, 3, 4).contiguous()

class ObservableHead(nn.Module):
    def __init__(self, in_ch, latent_dim):
        super().__init__()
        hidden = max(in_ch, latent_dim * 3)
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, latent_dim, 1, 1, 0),
        )
        self.skip = nn.Conv2d(in_ch, latent_dim, 1, 1, 0, bias=False)
        self.out_norm = nn.GroupNorm(1, latent_dim)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        z = self.main(x) + self.skip(x)
        z = self.out_norm(z)
        z = torch.tanh(z)
        return z.reshape(b, t, z.shape[1], h, w)


class LocalKoopmanMixer(nn.Module):
    def __init__(self, latent_dim, num_basis):
        super().__init__()
        hidden = max(latent_dim * 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim * 3, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_basis, 1, 1, 0),
        )

    def forward(self, z):
        z_mean = z.mean(dim=1)
        z_var = z.var(dim=1, unbiased=False)
        if z.shape[1] > 1:
            z_delta = torch.abs(z[:, 1:] - z[:, :-1]).mean(dim=1)
        else:
            z_delta = torch.zeros_like(z_mean)
        stats = torch.cat([z_mean, z_var, z_delta], dim=1)
        return torch.softmax(self.net(stats), dim=1)

class KoopmanDynamics(nn.Module):
    """
    Stable local Koopman operator field.

    K is used for bidirectional dynamic prediction. The modal projector below
    also projects the same local K into a POD coordinate system, but dynamic_seq
    is NOT used as detector evidence in Version C.
    """
    def __init__(self, latent_dim, num_basis, pred_window_size=16):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_basis = int(num_basis)
        self.pred_window_size = int(pred_window_size)

        self.base_core = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.02)
        self.base_damping = nn.Parameter(torch.zeros(latent_dim))
        self.residual_cores = nn.Parameter(torch.randn(num_basis, latent_dim, latent_dim) * 0.02)
        self.residual_scale = nn.Parameter(torch.zeros(num_basis))
        self.dt_logit = nn.Parameter(torch.tensor(0.0))

    def get_dt(self):
        return 0.2 + 0.8 * torch.sigmoid(self.dt_logit)

    def _stable_base_generator(self):
        skew = 0.5 * (self.base_core - self.base_core.transpose(-1, -2))
        damp = torch.diag(F.softplus(self.base_damping) + 1e-4)
        return torch.nan_to_num(skew - damp, nan=0.0, posinf=1e4, neginf=-1e4)

    def effective_generators(self):
        a0 = self._stable_base_generator()
        r = 0.05 * torch.tanh(self.residual_cores)
        alpha = torch.tanh(self.residual_scale).view(self.num_basis, 1, 1)
        a_all = a0.unsqueeze(0) + alpha * r
        return torch.nan_to_num(a_all, nan=0.0, posinf=1e4, neginf=-1e4)

    def effective_basis(self):
        a_all = self.effective_generators()
        dt = self.get_dt()
        k_all = torch.matrix_exp(dt * a_all)
        return torch.nan_to_num(k_all, nan=0.0, posinf=1e4, neginf=-1e4)

    def _pad_hw(self, x, ws):
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h == 0 and pad_w == 0:
            return x, h, w, h, w
        x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2], x.shape[-1]
        return x, h, w, hp, wp

    def _partition_feature(self, z_t, ws):
        z_t, h, w, hp, wp = self._pad_hw(z_t, ws)
        b, d, _, _ = z_t.shape
        n_h, n_w = hp // ws, wp // ws
        z_w = z_t.view(b, d, n_h, ws, n_w, ws).permute(0, 2, 4, 1, 3, 5).contiguous()
        return z_w, h, w, hp, wp, n_h, n_w

    def _unpartition_feature(self, z_w, h, w, hp, wp):
        b, n_h, n_w, d, ws, _ = z_w.shape
        z = z_w.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d, hp, wp)
        return z[..., :h, :w]

    def local_operator_field(self, omega, window_size, direction=1.0):
        b, m, _, _ = omega.shape
        ws = int(window_size)
        omega_pad, _, _, hp, wp = self._pad_hw(omega, ws)
        n_h, n_w = hp // ws, wp // ws
        o_w = omega_pad.view(b, m, n_h, ws, n_w, ws).permute(0, 2, 4, 1, 3, 5).contiguous()
        omega_local = o_w.mean(dim=(-1, -2))
        a_all = self.effective_generators().to(dtype=omega.dtype)
        dt = self.get_dt().to(dtype=omega.dtype)
        a_local = torch.einsum("bhwm,mij->bhwij", omega_local, a_all)
        signed_dt = float(direction) * dt
        k_local = torch.matrix_exp(signed_dt * a_local.reshape(b * n_h * n_w, self.latent_dim, self.latent_dim))
        return torch.nan_to_num(k_local, nan=0.0, posinf=1e4, neginf=-1e4), omega_local, (hp, wp, n_h, n_w, ws)

    def _apply_operator_from_field(self, z_t, k_local, ws):
        z_w, h, w, hp, wp, n_h, n_w = self._partition_feature(z_t, ws)
        z_flat = z_w.reshape(z_t.shape[0] * n_h * n_w, self.latent_dim, ws * ws)
        pred_flat = torch.einsum("bij,bjn->bin", k_local, z_flat)
        pred_w = pred_flat.reshape(z_t.shape[0], n_h, n_w, self.latent_dim, ws, ws)
        pred = self._unpartition_feature(pred_w, h, w, hp, wp)
        return torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

    def bidirectional_teacher_forcing_predict(self, z, omega):
        ws = int(self.pred_window_size)
        k_fw, _, _ = self.local_operator_field(omega, ws, direction=1.0)
        k_bw, _, _ = self.local_operator_field(omega, ws, direction=-1.0)
        pred_fw = [self._apply_operator_from_field(z[:, tt], k_fw, ws) for tt in range(z.shape[1] - 1)]
        pred_bw = [self._apply_operator_from_field(z[:, tt + 1], k_bw, ws) for tt in range(z.shape[1] - 1)]
        return torch.stack(pred_fw, dim=1), torch.stack(pred_bw, dim=1)

    def teacher_forcing_predict(self, z, omega):
        pred_fw, _ = self.bidirectional_teacher_forcing_predict(z, omega)
        return pred_fw


class LocalWindowModalKoopmanProjector(nn.Module):
    """
    Local POD/eigen decomposition + Koopman modal consistency projector.

    This is the TIP-style branch: it separates background-consistent modal
    components z_bg and non-background modal residual z_spec_res. The modal
    basis is built only from local observable samples; no auxiliary mask is
    used to define the POD covariance or background modes.
    """
    def __init__(
        self,
        latent_dim,
        window_size=16,
        max_bg_modes=8,
        energy_keep=0.90,
        select_temp=0.06,
        modal_consistency_beta=8.0,
        temporal_sigma_scale=0.35,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.window_size = int(window_size)
        self.max_bg_modes = int(max_bg_modes)
        self.energy_keep = float(energy_keep)
        self.select_temp = float(select_temp)
        self.modal_consistency_beta = float(modal_consistency_beta)
        self.temporal_sigma_scale = float(temporal_sigma_scale)

    def _pad_hw(self, x):
        ws = self.window_size
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h == 0 and pad_w == 0:
            return x, h, w, h, w
        x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2], x.shape[-1]
        return x, h, w, hp, wp

    def _partition_z(self, z):
        z, h, w, hp, wp = self._pad_hw(z)
        b, t, d, _, _ = z.shape
        ws = self.window_size
        n_h, n_w = hp // ws, wp // ws
        z_w = z.view(b, t, d, n_h, ws, n_w, ws).permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        return z_w, h, w, hp, wp, n_h, n_w

    def _partition_omega(self, omega, hp, wp):
        b, m, h, w = omega.shape
        ws = self.window_size
        pad_h = hp - h
        pad_w = wp - w
        if pad_h > 0 or pad_w > 0:
            omega = F.pad(omega, (0, pad_w, 0, pad_h))
        n_h, n_w = hp // ws, wp // ws
        return omega.view(b, m, n_h, ws, n_w, ws).permute(0, 2, 4, 1, 3, 5).contiguous()

    def _unpartition_z(self, z_w, h, w, hp, wp):
        b, n_h, n_w, t, d, ws, _ = z_w.shape
        z = z_w.permute(0, 3, 4, 1, 5, 2, 6).contiguous().view(b, t, d, hp, wp)
        return z[..., :h, :w]

    def _window_map_to_full(self, x_w, h, w, ws):
        if x_w.dim() == 4:
            x_w = x_w[..., 0]
        b, n_h, n_w = x_w.shape
        x = x_w[:, :, :, None, None].expand(b, n_h, n_w, ws, ws)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, n_h * ws, n_w * ws)
        return x[:, :h, :w].unsqueeze(1)

    def _window_seq_to_full(self, x_w_seq, h, w, ws):
        b, n_h, n_w, tm = x_w_seq.shape
        x = x_w_seq[:, :, :, :, None, None].expand(b, n_h, n_w, tm, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, tm, n_h * ws, n_w * ws)
        return x[:, :, :h, :w].unsqueeze(2)

    def _safe_eigh(self, A):
        bnw, d, _ = A.shape
        A = torch.nan_to_num(A, nan=0.0, posinf=1e4, neginf=-1e4)
        A = 0.5 * (A + A.transpose(-1, -2))
        scale = A.norm(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        A = A / scale
        eye = torch.eye(d, device=A.device, dtype=A.dtype).unsqueeze(0)
        for eps in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            try:
                return torch.linalg.eigh(A + eps * eye)
            except RuntimeError:
                pass
        evals = torch.diagonal(A, dim1=-2, dim2=-1)
        evecs = eye.expand(bnw, -1, -1)
        return evals, evecs

    def _temporal_sample_weights(self, t, ws, device, dtype):
        idx = torch.arange(t, device=device, dtype=dtype)
        center = 0.5 * (t - 1)
        sigma = max(float(t) * self.temporal_sigma_scale, 1.0)
        time_w = torch.exp(-0.5 * ((idx - center) / sigma) ** 2)
        time_w = time_w / (time_w.sum() + 1e-6)
        return time_w[:, None].expand(t, ws * ws).reshape(1, 1, t * ws * ws)

    def _build_sample_weights(self, t, ws, z_w_flat):
        # Pure temporal weighting. KMCP remains determined only by local modal
        # energy dominance and Koopman modal predictability; no auxiliary mask
        # or dynamic-error-derived prior is used to build the POD basis.
        return self._temporal_sample_weights(t, ws, z_w_flat.device, z_w_flat.dtype)

    def forward(self, generators, dt, omega, z):
        ws = self.window_size
        b, t, d, h, w = z.shape

        z_w, h0, w0, hp, wp, n_h, n_w = self._partition_z(z)
        o_w = self._partition_omega(omega, hp, wp)

        omega_local = o_w.mean(dim=(-1, -2)).detach().float()
        a_local = torch.einsum("bhwm,mij->bhwij", omega_local, generators.detach().float())
        k_local = torch.matrix_exp(dt.detach().float() * a_local.reshape(b * n_h * n_w, d, d))
        k_local = torch.nan_to_num(k_local, nan=0.0, posinf=1e4, neginf=-1e4)

        bnw = b * n_h * n_w
        z_w_flat = z_w.reshape(b, n_h, n_w, t, d, ws * ws).reshape(bnw, t, d, ws * ws)
        x_raw = z_w_flat.permute(0, 2, 1, 3).reshape(bnw, d, t * ws * ws)

        sample_w = self._build_sample_weights(t, ws, z_w_flat)
        mu = (x_raw * sample_w).sum(dim=-1, keepdim=True) / (sample_w.sum(dim=-1, keepdim=True) + 1e-6)
        x_centered = (x_raw - mu) * sample_w.sqrt()
        denom = sample_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        c_local = torch.matmul(x_centered, x_centered.transpose(1, 2)) / denom
        c_local = 0.5 * (c_local + c_local.transpose(1, 2))

        pod_vals, pod_vecs = self._safe_eigh(c_local)
        idx = torch.argsort(pod_vals, dim=1, descending=True)
        pod_vals = torch.gather(pod_vals, 1, idx)
        idx_expand = idx.unsqueeze(1).expand(-1, d, -1)
        pod_vecs = torch.gather(pod_vecs, 2, idx_expand)

        if self.max_bg_modes < d:
            valid = (torch.arange(d, device=z.device) < self.max_bg_modes).unsqueeze(0)
            pod_vals = pod_vals.masked_fill(~valid, 0.0)
        else:
            valid = None

        energy = pod_vals.clamp_min(0.0)
        energy = energy / (energy.sum(dim=1, keepdim=True) + 1e-6)
        cum_energy = energy.cumsum(dim=1)
        soft_keep = torch.sigmoid((self.energy_keep - cum_energy) / self.select_temp)
        if valid is not None:
            soft_keep = soft_keep * valid.to(dtype=soft_keep.dtype)

        coeff = torch.einsum("bji,btjn->btin", pod_vecs, z_w_flat)
        a_t = coeff.mean(dim=-1)

        k_modal = torch.einsum("bji,bjk,bkl->bil", pod_vecs, k_local, pod_vecs)
        a_pred = torch.einsum("bij,btj->bti", k_modal, a_t[:, :-1])

        modal_mean = a_t[:, 1:].abs().mean(dim=1).clamp_min(1e-6)
        modal_err = torch.abs(a_t[:, 1:] - a_pred).mean(dim=1)
        rel_err = modal_err / modal_mean
        consistency = torch.exp(-self.modal_consistency_beta * rel_err).clamp(0.0, 1.0)
        if valid is not None:
            consistency = consistency * valid.to(dtype=consistency.dtype)

        # IMPORTANT FIX: background modes require both dominant energy and
        # Koopman modal consistency. This makes the eigenvalue/POD story match
        # the code: high-energy + predictable modes are treated as background.
        mode_weights = torch.nan_to_num(
            soft_keep * consistency,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        expected_count = mode_weights.sum(dim=1)
        p_bg = pod_vecs @ torch.diag_embed(mode_weights) @ pod_vecs.transpose(1, 2)
        p_bg = p_bg.detach()

        z_bg_flat = torch.einsum("bij,btjn->btin", p_bg, z_w_flat)
        z_bg_w = z_bg_flat.reshape(b, n_h, n_w, t, d, ws, ws)
        z_spec_w = z_w - z_bg_w

        z_bg = self._unpartition_z(z_bg_w, h0, w0, hp, wp)
        z_spec_res = self._unpartition_z(z_spec_w, h0, w0, hp, wp)
        z_bg = torch.nan_to_num(z_bg, nan=0.0, posinf=1e4, neginf=-1e4)
        z_spec_res = torch.nan_to_num(z_spec_res, nan=0.0, posinf=1e4, neginf=-1e4)

        mode_count_map = self._window_map_to_full(expected_count.reshape(b, n_h, n_w), h0, w0, ws)
        consistency_scalar = (mode_weights * consistency).sum(dim=1) / (mode_weights.sum(dim=1) + 1e-6)
        consistency_map = self._window_map_to_full(consistency_scalar.reshape(b, n_h, n_w), h0, w0, ws)
        modal_err_scalar = (mode_weights * rel_err).sum(dim=1) / (mode_weights.sum(dim=1) + 1e-6)
        modal_err_map = self._window_map_to_full(modal_err_scalar.reshape(b, n_h, n_w), h0, w0, ws)

        denom_modes = mode_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        err_modes = torch.abs(a_t[:, 1:] - a_pred)
        err_seq = (err_modes * mode_weights.unsqueeze(1)).sum(dim=-1) / denom_modes
        err_seq = torch.nan_to_num(err_seq, nan=0.0, posinf=1e4, neginf=0.0)
        err_w = err_seq.reshape(b, n_h, n_w, t - 1)
        one_step_violation_seq = self._window_seq_to_full(err_w, h0, w0, ws)

        return {
            "z_bg": z_bg,
            "z_spec_res": z_spec_res,
            "mode_count_map": mode_count_map,
            "consistency_map": consistency_map,
            "modal_error_map": modal_err_map,
            "one_step_violation_seq": one_step_violation_seq.detach(),
            "projector_violation_profile": one_step_violation_seq.detach(),
        }


class MultiScaleModalKoopmanProjector(nn.Module):
    def __init__(self, latent_dim, window_sizes=(4, 8), max_bg_modes=8):
        super().__init__()
        self.window_sizes = tuple(int(ws) for ws in window_sizes)
        self.projectors = nn.ModuleList([
            LocalWindowModalKoopmanProjector(
                latent_dim=latent_dim,
                window_size=ws,
                max_bg_modes=max_bg_modes,
            )
            for ws in self.window_sizes
        ])
        self.scale_fuser = nn.Sequential(
            nn.Conv2d(len(self.window_sizes), 24, 3, 1, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, len(self.window_sizes), 1, 1, 0),
        )

    def forward(self, generators, dt, omega, z):
        outputs = []
        spec_maps = []
        for proj in self.projectors:
            out = proj(generators, dt, omega, z)
            outputs.append(out)
            spec_maps.append(torch.norm(out["z_spec_res"].mean(dim=1), p=2, dim=1, keepdim=True))

        scale_logits = self.scale_fuser(torch.cat(spec_maps, dim=1))
        scale_w = torch.softmax(scale_logits, dim=1)

        z_bg = 0.0
        z_spec_res = 0.0
        mode_count_map = 0.0
        consistency_map = 0.0
        modal_error_map = 0.0
        one_step_violation_seq = 0.0
        for i, out in enumerate(outputs):
            w = scale_w[:, i:i + 1]
            z_bg = z_bg + out["z_bg"] * w.unsqueeze(1)
            z_spec_res = z_spec_res + out["z_spec_res"] * w.unsqueeze(1)
            mode_count_map = mode_count_map + out["mode_count_map"] * w
            consistency_map = consistency_map + out["consistency_map"] * w
            modal_error_map = modal_error_map + out["modal_error_map"] * w
            one_step_violation_seq = one_step_violation_seq + out["one_step_violation_seq"] * w.unsqueeze(1)

        return {
            "z_bg": torch.nan_to_num(z_bg, nan=0.0, posinf=1e4, neginf=-1e4),
            "z_spec_res": torch.nan_to_num(z_spec_res, nan=0.0, posinf=1e4, neginf=-1e4),
            "mode_count_map": torch.nan_to_num(mode_count_map, nan=0.0, posinf=1e4, neginf=-1e4),
            "consistency_map": torch.nan_to_num(consistency_map, nan=0.0, posinf=1.0, neginf=0.0),
            "modal_error_map": torch.nan_to_num(modal_error_map, nan=0.0, posinf=1e4, neginf=0.0),
            "one_step_violation_seq": torch.nan_to_num(one_step_violation_seq, nan=0.0, posinf=1e4, neginf=0.0),
            "projector_violation_profile": torch.nan_to_num(one_step_violation_seq, nan=0.0, posinf=1e4, neginf=0.0),
            "scale_weights": scale_w,
            "per_scale": outputs,
        }

class KoopmanViolationTPro(nn.Module):
    """
    Multi-Probe Koopman Residual Temporal Attention (MP-KTRA) version.

    Drop-in replacement for the previous KoopmanViolationTPro.

    Main changes:
      1) Channel-grouped temporal probes:
         The embedded KMCP residual feature is split into G channel groups.
         Each group has an independent KTRA probe with its own temporal query,
         input-time key embedding, content key/value projections, and relative
         temporal bias.

      2) Position-adaptive + channel-diverse temporal modeling:
         Each probe keeps the residual-content-dependent key term, so attention
         remains spatially adaptive. Different channel groups learn different
         temporal violation patterns.

      3) Probe gate:
         A lightweight softmax gate adaptively fuses different probes at each
         spatial position and output time.

    Input:
        x: [B, T, C_p, H, W]
           non-negative channel-aware KMCP residual profile.

    Output:
        feat_out:     [B, T, C_out, H, W]
        score_out:    [B, T, 1, H, W]
        affinity_vis: [1, T, T]
        aux:          dict for visualization/analysis

    Notes:
      - The constructor keeps the old argument name `num_scorm` for compatibility
        with your existing KSTTipV2Net. In this implementation, `num_scorm` means
        the number of KTRA probes / channel groups, not a static SCorM bank.
      - No motion alignment, no tube offset, no hard-clutter pseudo-label branch.
    """

    def __init__(
        self,
        max_time=64,
        profile_ch=32,
        embed_ch=32,
        num_scorm=4,        # compatibility: now used as num_probes
        out_ch=4,
        tube_radius=0,      # compatibility only; no tube is used
        signature_ch=16,
        attn_ch=None,
    ):
        super().__init__()

        self.max_time = int(max_time)
        self.profile_ch = int(profile_ch)
        self.embed_ch = int(embed_ch)
        self.out_ch = int(out_ch)
        self.signature_ch = int(signature_ch)
        self.eps = 1e-6

        # Number of channel-grouped KTRA probes.
        self.num_probes = max(1, int(num_scorm))
        self.num_probes = min(self.num_probes, self.embed_ch)

        if attn_ch is None:
            attn_ch = max(embed_ch // 2, 8)
        self.attn_ch = int(attn_ch)

        # Split embed_ch into almost equal channel groups.
        base = self.embed_ch // self.num_probes
        rem = self.embed_ch % self.num_probes
        self.group_sizes = [base + (1 if i < rem else 0) for i in range(self.num_probes)]
        assert sum(self.group_sizes) == self.embed_ch

        # Each probe has its own lower-dimensional attention space.
        self.probe_attn_ch = max(self.attn_ch // self.num_probes, 4)
        self.total_probe_attn_ch = self.probe_attn_ch * self.num_probes

        # Four descriptors for each KMCP residual channel:
        # log magnitude, temporal z-score, local temporal contrast, temporal difference.
        in_ch = 4 * self.profile_ch
        self.profile_embed = nn.Sequential(
            nn.Conv3d(in_ch, embed_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_ch, embed_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_ch),
            nn.ReLU(inplace=True),
        )

        hidden_gate = max(embed_ch // 2, 8)

        # Shared residual reliability gate before temporal probing.
        self.residual_gate = nn.Sequential(
            nn.Conv3d(
                embed_ch,
                embed_ch,
                kernel_size=(5, 1, 1),
                padding=(2, 0, 0),
                groups=embed_ch,
                bias=False,
            ),
            nn.BatchNorm3d(embed_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_ch, hidden_gate, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_gate),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_gate, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Probe gate: adaptively select different temporal probes per location/time.
        self.probe_gate = nn.Sequential(
            nn.Conv3d(embed_ch, hidden_gate, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_gate),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_gate, self.num_probes, kernel_size=1),
        )

        # Probe-specific temporal queries, temporal key embeddings, projections, and biases.
        self.temporal_queries = nn.ParameterList()
        self.temporal_key_embed = nn.ParameterList()
        self.query_norms = nn.ModuleList()
        self.key_time_norms = nn.ModuleList()
        self.k_projs = nn.ModuleList()
        self.v_projs = nn.ModuleList()
        self.rel_biases = nn.ParameterList()
        self.rel_prior_scales = nn.ParameterList()

        for g, g_ch in enumerate(self.group_sizes):
            self.temporal_queries.append(
                nn.Parameter(torch.randn(self.max_time, self.probe_attn_ch) * 0.02)
            )
            self.temporal_key_embed.append(
                nn.Parameter(torch.randn(self.max_time, self.probe_attn_ch) * 0.02)
            )
            self.query_norms.append(nn.LayerNorm(self.probe_attn_ch))
            self.key_time_norms.append(nn.LayerNorm(self.probe_attn_ch))
            self.k_projs.append(nn.Conv3d(g_ch, self.probe_attn_ch, kernel_size=1, bias=False))
            self.v_projs.append(nn.Conv3d(g_ch, self.probe_attn_ch, kernel_size=1, bias=False))
            self.rel_biases.append(nn.Parameter(torch.zeros(2 * self.max_time - 1)))
            self.rel_prior_scales.append(nn.Parameter(torch.tensor(0.35)))

        self.register_buffer(
            "rel_prior",
            self._build_relative_prior(self.max_time),
            persistent=False,
        )

        self.attn_out_proj = nn.Sequential(
            nn.Conv3d(self.total_probe_attn_ch, self.total_probe_attn_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.total_probe_attn_ch),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(self.total_probe_attn_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.score_head = nn.Conv3d(out_ch, 1, kernel_size=1)

        # Evidence head only for visualization / auxiliary map generation.
        # It is not a residual-signature classifier.
        factor_in = out_ch + 3  # feature + amp + attention focus + residual gate
        self.evidence_factor_head = nn.Sequential(
            nn.Conv2d(factor_in, max(signature_ch, 8), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(signature_ch, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(signature_ch, 8), 1, kernel_size=1),
        )

        # Set True during validation/analysis if point-wise full affinity is needed.
        self.return_full_affinity = False

    def _build_relative_prior(self, max_time):
        idx = torch.arange(max_time).float()
        dist = (idx[:, None] - idx[None, :]).abs()
        scale = max(2.0, max_time / 4.0)
        return -dist / float(scale)

    def _get_relative_bias(self, tm, probe_idx, device, dtype):
        max_t = self.max_time
        idx = torch.arange(tm, device=device)
        rel_index = idx[:, None] - idx[None, :] + (max_t - 1)
        learned = self.rel_biases[probe_idx][rel_index].to(device=device, dtype=dtype)
        prior = self.rel_prior[:tm, :tm].to(device=device, dtype=dtype)
        scale = self.rel_prior_scales[probe_idx].to(device=device, dtype=dtype)
        return learned + scale * prior

    def _spatial_norm(self, x):
        denom = x.detach().mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        return torch.nan_to_num(x / denom, nan=0.0, posinf=1e4, neginf=0.0)

    def _make_profile_features(self, x):
        """
        x: [B, C_p, T, H, W]
        """
        x = torch.clamp(x, min=0.0)
        x_log = torch.log1p(x)

        mean = x_log.mean(dim=2, keepdim=True)
        std = x_log.std(dim=2, keepdim=True, unbiased=False).clamp_min(self.eps)
        x_norm = (x_log - mean) / std

        t = x.shape[2]
        k = 5 if t >= 5 else (3 if t >= 3 else 1)
        avg = F.avg_pool3d(
            x_log,
            kernel_size=(k, 1, 1),
            stride=1,
            padding=(k // 2, 0, 0),
        )
        contrast = F.relu(x_log - avg)

        diff = torch.zeros_like(x_log)
        if t > 1:
            diff[:, :, 1:] = torch.abs(x_log[:, :, 1:] - x_log[:, :, :-1])

        return torch.cat([x_log, x_norm, contrast, diff], dim=1), x_log

    def _temporal_attention(self, feat):
        b, c, tm, h, w = feat.shape
        # [B,G,T,H,W], softmax over probe dimension.
        probe_weight = torch.softmax(self.probe_gate(feat), dim=1)
        probe_weight = torch.nan_to_num(probe_weight, nan=0.0, posinf=0.0, neginf=0.0)
        feat_groups = torch.split(feat, self.group_sizes, dim=1)
        y_list = []
        attn_mix = None
        prior_acc = None
        probe_affinity_list = []

        for g, feat_g in enumerate(feat_groups):
            q_time = self.query_norms[g](self.temporal_queries[g][:tm]).to(
                device=feat.device,
                dtype=feat.dtype,
            )  # [T,Dg]

            k_time = self.key_time_norms[g](self.temporal_key_embed[g][:tm]).to(
                device=feat.device,
                dtype=feat.dtype,
            )  # [T,Dg]

            k_content = self.k_projs[g](feat_g)  # [B,Dg,T,H,W]
            v = self.v_projs[g](feat_g)          # [B,Dg,T,H,W]

            k = k_content + k_time.transpose(0, 1)[None, :, :, None, None]

            logits = torch.einsum(
                "td,bdlhw->btlhw",
                q_time,
                k,
            ) / math.sqrt(float(self.probe_attn_ch))

            rel_bias = self._get_relative_bias(tm, g, feat.device, feat.dtype)
            logits = logits + rel_bias[None, :, :, None, None]


            attn = torch.softmax(logits, dim=2)
            attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

            y_g = torch.einsum(
                "btlhw,bdlhw->bdthw",
                attn,
                v,
            )

            # Dynamic probe selection. Each probe contributes differently at each
            # output time and spatial location.
            w_g = probe_weight[:, g:g + 1]  # [B,1,T,H,W]
            y_g = y_g * w_g
            y_list.append(y_g)

            # Mixed attention used for entropy/focus visualization.
            attn_weight = w_g.squeeze(1).unsqueeze(2)  # [B,T,1,H,W]
            attn_weighted = attn * attn_weight          # [B,T,T,H,W]
            if attn_mix is None:
                attn_mix = attn_weighted
            else:
                attn_mix = attn_mix + attn_weighted

            prior_logits = torch.einsum("td,ld->tl", q_time, k_time) / math.sqrt(float(self.probe_attn_ch))
            prior_logits = prior_logits + rel_bias
            prior_attn_g = torch.softmax(prior_logits, dim=1)
            prior_attn_g = torch.nan_to_num(prior_attn_g, nan=0.0, posinf=0.0, neginf=0.0)

            if prior_acc is None:
                prior_acc = prior_attn_g
            else:
                prior_acc = prior_acc + prior_attn_g

            probe_affinity_list.append(attn.mean(dim=(0, 3, 4)).detach())  # [T,T]

        y = torch.cat(y_list, dim=1)  # [B,D_total,T,H,W]
        y = self.attn_out_proj(y)

        # Probe weights sum to one over G, so attn_mix remains normalized over input time.
        attn_mix = torch.nan_to_num(attn_mix, nan=0.0, posinf=0.0, neginf=0.0)
        prior_attn = prior_acc / float(self.num_probes)
        probe_affinity = torch.stack(probe_affinity_list, dim=0)  # [G,T,T]

        return y, attn_mix, prior_attn, probe_affinity, probe_weight

    def forward(self, x):
        """
        x: [B, T, C_p, H, W]
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=0.0)
        b, tm, c, h, w = x.shape

        if c != self.profile_ch:
            raise ValueError(
                f"KoopmanViolationTPro expects profile_ch={self.profile_ch}, "
                f"but got input channel={c}."
            )

        tm_use = min(tm, self.max_time)

        # [B,T,C_p,H,W] -> [B,C_p,T,H,W]
        x = x[:, :tm_use].permute(0, 2, 1, 3, 4).contiguous()

        profile, _ = self._make_profile_features(x)
        feat0 = self.profile_embed(profile)

        residual_gate = self.residual_gate(feat0)  # [B,1,T,H,W]
        feat0_mod = feat0 * (1.0 + residual_gate)

        y, attn, prior_attn, probe_affinity, probe_weight = self._temporal_attention(feat0_mod)

        feat = self.fuse(y)
        score_logit = self.score_head(feat)
        score = F.softplus(score_logit)

        # Visualization / interpretation maps.
        fixed_mag = torch.norm(x, p=2, dim=1, keepdim=True)  # [B,1,T,H,W]
        amp_map = self._spatial_norm(fixed_mag.mean(dim=2))  # [B,1,H,W]

        # Gated amplitude map, useful for before/after residual-gate visualization.
        gated_mag = fixed_mag * (1.0 + residual_gate)
        gated_amp_map = self._spatial_norm(gated_mag.mean(dim=2))

        log_tm = math.log(max(tm_use, 2))
        attn_entropy = -(attn * attn.clamp_min(self.eps).log()).sum(dim=2) / log_tm
        attn_entropy_map = attn_entropy.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        attn_focus_map = (1.0 - attn_entropy_map).clamp(0.0, 1.0)

        gate_map = residual_gate.mean(dim=2).clamp(0.0, 1.0)

        feat_pool = feat.mean(dim=2)
        factor_feat = torch.cat(
            [
                feat_pool,
                amp_map.clamp(0.0, 8.0),
                attn_focus_map,
                gate_map,
            ],
            dim=1,
        )
        evidence_logits = self.evidence_factor_head(factor_feat)
        evidence_map = torch.sigmoid(evidence_logits)

        feat_out = feat.permute(0, 2, 1, 3, 4).contiguous()
        score_out = score.permute(0, 2, 1, 3, 4).contiguous()

        # Compact average temporal affinity for paper-style visualization.
        affinity_vis = attn.mean(dim=(0, 3, 4), keepdim=False).unsqueeze(0).detach()  # [1,T,T]

        # Probe-wise compact affinity, useful for multi-probe visualization.
        probe_affinity_vis = probe_affinity.unsqueeze(0).detach()  # [1,G,T,T]
        probe_weight_map = probe_weight.mean(dim=2).detach()        # [B,G,H,W]
        probe_weight_mean = probe_weight.mean(dim=(2, 3, 4)).detach()  # [B,G]

        aux = {
            "ktvm_fixed_amp_map": amp_map,
            "ktvm_amp_map": gated_amp_map,
            "ktvm_residual_gate_map": gate_map,
            "ktvm_attention_entropy_map": attn_entropy_map,
            "ktvm_attention_focus_map": attn_focus_map,
            "ktvm_evidence_logits": evidence_logits,
            "ktvm_evidence_map": evidence_map,
            "ktvm_temporal_affinity": affinity_vis,
            "ktvm_learned_temporal_prior": prior_attn.unsqueeze(0).detach(),  # [1,T,T]
            # New MP-KTRA visualization keys.
            "ktvm_temporal_affinity_probes": probe_affinity_vis,              # [1,G,T,T]
            "ktvm_probe_weight_map": probe_weight_map,                        # [B,G,H,W]
            "ktvm_probe_weight_mean": probe_weight_mean,                      # [B,G]
            "ktvm_num_probes": self.num_probes,
        }

        if getattr(self, "return_full_affinity", False):
            aux["ktvm_temporal_affinity_full"] = attn.detach()  # [B,T,T,H,W]

        return feat_out, score_out, affinity_vis, aux

class KoopmanModalCorrelationAggregator(nn.Module):
    def __init__(self, in_ch=4, out_ch=32, hidden_ch=32, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.temporal_body = nn.Sequential(
            nn.Conv3d(in_ch, hidden_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_ch, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.score_head = nn.Sequential(
            nn.Conv3d(out_ch, max(out_ch // 2, 8), 1, bias=False),
            nn.BatchNorm3d(max(out_ch // 2, 8)),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(out_ch // 2, 8), 1, 1),
        )
    def _normalize_channelwise(self, x):
        denom = x.detach().mean(dim=(2, 3, 4), keepdim=True).clamp_min(self.eps)
        x = x / denom
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=0.0)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self._normalize_channelwise(torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=0.0))
        feat_seq = self.temporal_body(x)
        score_seq = F.softplus(self.score_head(feat_seq))
        weight = score_seq / score_seq.mean(dim=2, keepdim=True).clamp_min(self.eps)
        feat = (feat_seq * weight).mean(dim=2)
        fused_seq = score_seq.permute(0, 2, 1, 3, 4).contiguous()
        gate_map = x.mean(dim=2)
        return feat, fused_seq, gate_map, score_seq.squeeze(1)


class SegDecoder(nn.Module):
    def __init__(self, in_ch, mid_ch=64):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU2d(in_ch, mid_ch, 3, 1, 1),
            ResBlock2d(mid_ch),
            ConvBNReLU2d(mid_ch, mid_ch, 3, 1, 1),
        )
        self.pred = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, x, out_size):
        return F.interpolate(self.pred(self.block(x)), size=out_size, mode="bilinear", align_corners=False)


class InnovationGuidedFusion(nn.Module):
    def __init__(self, app_ch, innov_ch, out_ch=96):
        super().__init__()
        self.app_proj = nn.Sequential(nn.Conv2d(app_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.innov_proj = nn.Sequential(nn.Conv2d(innov_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.innov_gate = nn.Sequential(
            nn.Conv2d(innov_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResBlock2d(out_ch),
        )

    def forward(self, app_feat, innov_feat):
        app = self.app_proj(app_feat)
        innov = self.innov_proj(innov_feat)
        gate = self.innov_gate(innov_feat)
        return self.refine(app * (1.0 + gate) + innov), gate


class KSTTipV2Net(nn.Module):
    """
    Recommended clean BCK-Net version:

      KMCP:  pure energy-dominance x Koopman-predictability modal projection.
             No auxiliary weighting is used in the modal basis.

      TVM:   Fixed-and-Motion-aligned Residual Temporal Violation Modeling
             (FM-MRTVM). It preserves fixed residual-profile temporal pulses,
             adds local motion-aligned residual tubes, and learns tube-level
             violation signatures from the KMCP residual space only.
    """
    def __init__(
        self,
        in_ch=1,
        feat_ch=96,
        latent_dim=32,
        num_basis=4,
        window_sizes=(4, 8),
        max_bg_modes=8,
        res_feat_ch=32,
        t_clip=40,
        evidence_hidden_ch=32,
        vtpro_embed_ch=32,
        vtpro_num_scorm=4,
        vtpro_out_ch=4,
        ktvm_tube_radius=1,
        kmec_hidden_ch=None,
    ):
        super().__init__()
        if kmec_hidden_ch is not None:
            evidence_hidden_ch = kmec_hidden_ch
        self.t_clip = int(t_clip)

        self.encoder = FrameEncoder(in_ch=in_ch, base_ch=32, out_ch=feat_ch)
        self.temporal_refiner = TemporalRefiner3D(ch=feat_ch)
        self.observable = ObservableHead(in_ch=feat_ch, latent_dim=latent_dim)
        self.omega_head = LocalKoopmanMixer(latent_dim=latent_dim, num_basis=num_basis)
        self.koopman = KoopmanDynamics(latent_dim=latent_dim, num_basis=num_basis, pred_window_size=max(window_sizes))
        self.projector = MultiScaleModalKoopmanProjector(
            latent_dim=latent_dim,
            window_sizes=window_sizes,
            max_bg_modes=max_bg_modes,
        )
        self.violation_tpro = KoopmanViolationTPro(
            max_time=max(self.t_clip - 1, 8),
            profile_ch=latent_dim,
            embed_ch=vtpro_embed_ch,
            num_scorm=vtpro_num_scorm,
            out_ch=vtpro_out_ch,
            tube_radius=ktvm_tube_radius,
        )
        self.evidence_aggregator = KoopmanModalCorrelationAggregator(
            in_ch=vtpro_out_ch,
            out_ch=res_feat_ch,
            hidden_ch=evidence_hidden_ch,
        )
        self.innovation_fusion = InnovationGuidedFusion(app_ch=feat_ch, innov_ch=res_feat_ch, out_ch=feat_ch)
        self.decoder = SegDecoder(in_ch=feat_ch, mid_ch=64)

    @staticmethod
    def _normalize_evidence(x, eps=1e-6):
        dims = tuple(i for i in range(1, x.dim()) if i != 2)
        denom = x.detach().mean(dim=dims, keepdim=True).clamp_min(eps)
        return torch.nan_to_num(x / denom, nan=0.0, posinf=1e4, neginf=0.0)

    def forward(self, clip):
        b, t, c, h, w = clip.shape
        feats = torch.stack([self.encoder(clip[:, i]) for i in range(t)], dim=1)
        feats = self.temporal_refiner(feats)
        z = self.observable(feats)
        omega = self.omega_head(z)
        basis_eff = self.koopman.effective_basis()
        generators = self.koopman.effective_generators()
        dt = self.koopman.get_dt()

        pred_fw, pred_bw = self.koopman.bidirectional_teacher_forcing_predict(z, omega)
        fw_seq_raw = torch.norm(z[:, 1:] - pred_fw, p=2, dim=2, keepdim=True)
        bw_seq_raw = torch.norm(z[:, :-1] - pred_bw, p=2, dim=2, keepdim=True)
        dynamic_seq_raw = 0.5 * (fw_seq_raw + bw_seq_raw)
        dynamic_seq = self._normalize_evidence(dynamic_seq_raw)

        # KMCP is intentionally pure: no auxiliary dynamic-error map is built or
        # injected. Dynamic error is kept only for diagnostics/background loss.
        proj = self.projector(generators, dt, omega, z)
        z_bg = proj["z_bg"]
        z_spec_res = proj["z_spec_res"]

        # Channel-aware KMCP residual profile. The residual is the only input to
        # MRTVM, keeping the TVM mechanism self-contained and interpretable.
        violation_profile_raw = torch.abs(z_spec_res[:, 1:])
        violation_profile = self._normalize_evidence(violation_profile_raw)
        violation_feat, violation_tpro_score_seq, violation_scorm, ktvm_aux = self.violation_tpro(violation_profile)

        evidence_feat, fused_evidence_seq, evidence_gate_map, evidence_score_seq = self.evidence_aggregator(violation_feat)
        evidence_feat = evidence_feat * (1.0 + ktvm_aux["ktvm_evidence_map"])
        center_feat = feats[:, t // 2]
        fusion, innovation_gate = self.innovation_fusion(center_feat, evidence_feat)
        logits = self.decoder(fusion, out_size=(h, w))

        dynamic_map = dynamic_seq.mean(dim=1)
        modal_map = torch.norm(z_spec_res, p=2, dim=2).mean(dim=1, keepdim=True)
        violation_profile_map = torch.norm(violation_profile, p=2, dim=2).mean(dim=1, keepdim=True)
        violation_tpro_map = violation_tpro_score_seq.mean(dim=1)
        evidence_map = evidence_score_seq.unsqueeze(2).mean(dim=1)
        fused_evidence_map = 0.5 * (fused_evidence_seq.mean(dim=1) + ktvm_aux["ktvm_evidence_map"])
        evidence_feature_map = torch.norm(evidence_feat, p=2, dim=1, keepdim=True)

        return {
            "logits": logits,
            "dynamic_map": dynamic_map,
            "modal_map": modal_map,
            "violation_profile_map": violation_profile_map,
            "violation_tpro_map": violation_tpro_map,
            "evidence_map": evidence_map,
            "fused_evidence_map": fused_evidence_map,
            "evidence_feature_map": evidence_feature_map,
            "mode_count_map": proj["mode_count_map"],
            "consistency_map": proj["consistency_map"],
            "modal_error_map": proj["modal_error_map"],
            "scale_weights": proj["scale_weights"],
            "per_scale": proj["per_scale"],
            "z": z,
            "z_bg": z_bg,
            "z_spec_res": z_spec_res,
            "pred_fw": pred_fw,
            "pred_bw": pred_bw,
            "fw_seq_raw": fw_seq_raw,
            "bw_seq_raw": bw_seq_raw,
            "dynamic_seq_raw": dynamic_seq_raw,
            "one_step_violation_seq": proj["one_step_violation_seq"].squeeze(2),
            "projector_violation_profile": proj["projector_violation_profile"],
            "violation_profile": violation_profile,
            "violation_profile_raw": violation_profile_raw,
            "violation_tpro_feat": violation_feat,
            "violation_tpro_score_seq": violation_tpro_score_seq.squeeze(2),
            "violation_scorm": violation_scorm,
            "evidence_score_seq": evidence_score_seq,
            # KRTA-LRQ auxiliary maps.
            # Explicit motion-alignment / tube / hard-clutter-signature keys were removed.
            "ktvm_fixed_amp_map": ktvm_aux["ktvm_fixed_amp_map"],
            "ktvm_amp_map": ktvm_aux["ktvm_amp_map"],
            "ktvm_evidence_logits": ktvm_aux["ktvm_evidence_logits"],
            "ktvm_evidence_map": ktvm_aux["ktvm_evidence_map"],
            "ktvm_residual_gate_map": ktvm_aux["ktvm_residual_gate_map"],
            "ktvm_attention_entropy_map": ktvm_aux.get("ktvm_attention_entropy_map", None),
            "ktvm_attention_focus_map": ktvm_aux.get("ktvm_attention_focus_map", None),
            "ktvm_query_focus_map": ktvm_aux.get("ktvm_query_focus_map", None),
            "ktvm_temporal_affinity": ktvm_aux.get("ktvm_temporal_affinity", violation_scorm),
            "ktvm_query_time_attention": ktvm_aux.get("ktvm_query_time_attention", None),
            "ktvm_query_mixture": ktvm_aux.get("ktvm_query_mixture", None),
            "omega": omega,
            "basis_eff": basis_eff,
            "generators": generators,
            "dt": dt,
            "innovation_gate": innovation_gate,
            "evidence_gate_map": evidence_gate_map,
        }
