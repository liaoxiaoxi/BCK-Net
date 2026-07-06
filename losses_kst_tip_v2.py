import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits, targets, eps=1e-6):
    """
    Dice loss for target-map prediction.

    logits:  [B, 1, H, W]
    targets: [B, 1, H, W]
    """
    logits = torch.clamp(logits, -20.0, 20.0)
    targets = targets.float()
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    loss = 1.0 - num / den
    return torch.nan_to_num(loss.mean(), nan=0.0, posinf=10.0, neginf=10.0)


def bce_dice_loss(logits, targets):
    """BCE + Dice for segmentation logits."""
    logits = torch.clamp(logits, -20.0, 20.0)
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return torch.nan_to_num(bce + dice, nan=0.0, posinf=10.0, neginf=10.0)


def downsample_mask(mask, size):
    return F.interpolate(mask.float(), size=size, mode="nearest")


class KSTTipV2Loss(nn.Module):
    """
    KRTA-version loss for BCK-Net.

    Total loss:
        L = lambda_det  * L_det
          + lambda_bg   * L_pairwise_background_Koopman_consistency
          + lambda_stab * L_generator_stability

    Important:
      - Final detection is supervised by the center-frame mask.
      - Koopman background consistency uses per-frame masks when a mask sequence
        [B,T,1,H,W] or [B,T,H,W] is provided.
      - Residual signature / hard-clutter pseudo-label loss is removed for KRTA.
        KRTA is supervised through the final detection objective.

    Backward compatibility:
      - lambda_vtpro, lambda_sig, target_margin and hard_power are accepted but ignored.
      - log_dict still contains zero-valued loss_vtpro/loss_sig keys to avoid
        breaking old training scripts or CSV readers.
    """
    def __init__(
        self,
        lambda_det=1.0,
        lambda_bg=0.03,
        lambda_vtpro=0.0,      # deprecated / ignored
        target_margin=0.12,    # deprecated / ignored
        dilate_ks=9,
        eps=1e-6,
        lambda_sig=None,       # deprecated / ignored
        lambda_stab=1e-4,
        robust_eps=1e-3,
        hard_power=1.5,        # deprecated / ignored
    ):
        super().__init__()
        self.lambda_det = float(lambda_det)
        self.lambda_bg = float(lambda_bg)
        self.lambda_sig = 0.0
        self.lambda_vtpro = 0.0
        self.lambda_stab = float(lambda_stab)
        self.target_margin = float(target_margin)
        self.dilate_ks = int(dilate_ks)
        self.eps = float(eps)
        self.robust_eps = float(robust_eps)
        self.hard_power = float(hard_power)

    def _clean(self, x, pos=1e4, neg=-1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=pos, neginf=neg)

    def _dilate_mask_4d(self, mask):
        """mask: [B,1,H,W] -> dilated binary mask [B,1,H,W]."""
        if self.dilate_ks <= 1:
            return (mask > 0.5).float()
        pad = self.dilate_ks // 2
        dilated = F.max_pool2d(mask.float(), kernel_size=self.dilate_ks, stride=1, padding=pad)
        return (dilated > 0.5).float()

    def _charbonnier(self, x):
        return torch.sqrt(x * x + self.robust_eps * self.robust_eps)

    def _weighted_mean(self, value, weight):
        weight = weight.to(dtype=value.dtype, device=value.device)
        denom = weight.sum().clamp_min(self.eps)
        return (value * weight).sum() / denom

    def _mask_to_5d(self, mask):
        """
        Normalize masks to [B,Tm,1,H,W]. Supported inputs:
          [B,H,W], [B,1,H,W], [B,T,H,W], [B,T,1,H,W], [B,1,T,H,W].
        """
        m = mask.float()
        if m.dim() == 3:
            return m.unsqueeze(1).unsqueeze(2)
        if m.dim() == 4:
            if m.shape[1] == 1:
                return m.unsqueeze(1)
            return m.unsqueeze(2)
        if m.dim() == 5:
            if m.shape[2] == 1:
                return m
            if m.shape[1] == 1:
                return m.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(
            "Unsupported mask shape. Expected [B,H,W], [B,1,H,W], [B,T,H,W], "
            f"[B,T,1,H,W], or [B,1,T,H,W], got {tuple(mask.shape)}."
        )

    def _center_mask(self, mask, target_size=None, clip_t=None):
        m5 = self._mask_to_5d(mask)
        tm = m5.shape[1]
        if tm > 1:
            if clip_t is not None and tm == clip_t:
                center = int(clip_t) // 2
            else:
                center = tm // 2
            y = m5[:, center]
        else:
            y = m5[:, 0]
        if target_size is not None and y.shape[-2:] != tuple(target_size):
            y = downsample_mask(y, size=target_size)
        return y.clamp(0.0, 1.0)

    def _get_pair_errors(self, outputs):
        if ("fw_seq_raw" in outputs) and ("bw_seq_raw" in outputs):
            err_fw = self._clean(outputs["fw_seq_raw"], pos=1e4, neg=0.0)
            err_bw = self._clean(outputs["bw_seq_raw"], pos=1e4, neg=0.0)
            return err_fw, err_bw

        required = ["z", "pred_fw", "pred_bw"]
        missing = [k for k in required if k not in outputs]
        if missing:
            raise KeyError(
                "Koopman background loss requires outputs['fw_seq_raw']/outputs['bw_seq_raw'] "
                "or outputs['z']/outputs['pred_fw']/outputs['pred_bw']. "
                f"Missing keys: {missing}"
            )
        z = self._clean(outputs["z"])
        pred_fw = self._clean(outputs["pred_fw"])
        pred_bw = self._clean(outputs["pred_bw"])
        err_fw = torch.abs(z[:, 1:] - pred_fw).mean(dim=2, keepdim=True)
        err_bw = torch.abs(z[:, :-1] - pred_bw).mean(dim=2, keepdim=True)
        return self._clean(err_fw, pos=1e4, neg=0.0), self._clean(err_bw, pos=1e4, neg=0.0)

    def _pair_background_mask(self, mask, size, pair_count, clip_t=None):
        """
        Build reliable background masks for temporal pairs.

        If per-frame masks are available, pair mask for (t,t+1) is:
          (1 - Dilate(Y_t)) * (1 - Dilate(Y_{t+1})).
        If only a center-frame mask is provided, repeat the dilated
        center-background mask over all temporal pairs.
        """
        m5 = self._mask_to_5d(mask).clamp(0.0, 1.0)  # [B,Tm,1,H,W]
        b, tm, _, h, w = m5.shape

        if tm >= pair_count + 1:
            if clip_t is not None and tm == clip_t and pair_count + 1 <= tm:
                start = 0
            else:
                start = max(0, (tm - (pair_count + 1)) // 2)
            m_pair = m5[:, start:start + pair_count + 1]
            mt = m_pair.reshape(b * (pair_count + 1), 1, h, w)
            if mt.shape[-2:] != tuple(size):
                mt = downsample_mask(mt, size=size)
            mt = self._dilate_mask_4d(mt).reshape(b, pair_count + 1, 1, size[0], size[1])
            bg_pair = (1.0 - mt[:, :-1]) * (1.0 - mt[:, 1:])
        else:
            yc = self._center_mask(mask, target_size=size, clip_t=clip_t)
            bg = (1.0 - self._dilate_mask_4d(yc)).clamp(0.0, 1.0)
            bg_pair = bg.unsqueeze(1).expand(b, pair_count, 1, size[0], size[1])
        return bg_pair.clamp(0.0, 1.0)

    def _koopman_background_consistency_loss(self, outputs, targets):
        """Pair-wise background Koopman consistency loss."""
        err_fw, err_bw = self._get_pair_errors(outputs)
        dyn_err = self._clean(0.5 * (err_fw + err_bw), pos=1e4, neg=0.0)
        b, tm, _, hf, wf = dyn_err.shape
        clip_t = outputs.get("z", None).shape[1] if torch.is_tensor(outputs.get("z", None)) else (tm + 1)

        bg_pair = self._pair_background_mask(targets, size=(hf, wf), pair_count=tm, clip_t=clip_t)
        loss_kbg_raw = self._weighted_mean(self._charbonnier(dyn_err), bg_pair)
        dyn_bg = self._weighted_mean(dyn_err.detach(), bg_pair).detach()
        return torch.nan_to_num(loss_kbg_raw, nan=0.0, posinf=10.0, neginf=10.0), dyn_bg

    def _stability_loss(self, outputs):
        if "generators" not in outputs:
            return None
        A = outputs["generators"]
        if not torch.is_tensor(A):
            return None
        A = self._clean(A, pos=1e4, neg=-1e4)
        eigvals = torch.linalg.eigvals(A.float())
        loss = torch.relu(eigvals.real).pow(2).mean().to(dtype=A.dtype, device=A.device)
        return torch.nan_to_num(loss, nan=0.0, posinf=10.0, neginf=10.0)

    def forward(self, outputs, mask):
        if "logits" not in outputs:
            raise KeyError("KSTTipV2Loss expects outputs['logits'], but it was not found.")

        logits = self._clean(outputs["logits"])
        clip_t = outputs.get("z", None).shape[1] if torch.is_tensor(outputs.get("z", None)) else None
        targets = self._center_mask(mask, target_size=logits.shape[-2:], clip_t=clip_t).clamp(0.0, 1.0)

        zero = logits.new_tensor(0.0)
        det_loss = bce_dice_loss(logits, targets)
        loss_det = self.lambda_det * det_loss

        if self.lambda_bg > 0:
            kbg_raw, dyn_bg = self._koopman_background_consistency_loss(outputs, mask)
            loss_bg = self.lambda_bg * kbg_raw
        else:
            kbg_raw, dyn_bg, loss_bg = zero, zero, zero

        stab_raw = self._stability_loss(outputs)
        if stab_raw is not None and self.lambda_stab > 0:
            loss_stab = self.lambda_stab * stab_raw
        else:
            stab_raw, loss_stab = zero, zero

        # Removed for KRTA: residual signature / hard-clutter pseudo-label loss.
        loss_sig = zero
        sig_raw = sig_tgt = sig_bg = sig_hard = zero

        total = torch.nan_to_num(
            loss_det + loss_bg + loss_stab,
            nan=0.0,
            posinf=10.0,
            neginf=10.0,
        )

        loss_parts = {
            "loss_det": loss_det,
            "loss_bg": loss_bg,
            "loss_stab": loss_stab,
            # compatibility aliases
            "loss_vtpro": loss_sig,
            "loss_sig": loss_sig,
        }

        log_dict = {
            "loss_total": total.detach(),
            "loss_det": loss_det.detach(),
            "loss_bg": loss_bg.detach(),
            "loss_kbg_raw": kbg_raw.detach(),
            "loss_stab": loss_stab.detach(),
            "loss_stab_raw": stab_raw.detach(),
            "dyn_bg": dyn_bg.detach(),
            # compatibility zero logs
            "loss_vtpro": loss_sig.detach(),
            "loss_sig": loss_sig.detach(),
            "loss_sig_raw": sig_raw.detach(),
            "loss_sig_tgt": sig_tgt.detach(),
            "loss_sig_bg": sig_bg.detach(),
            "loss_sig_hard": sig_hard.detach(),
            "vtpro_margin": zero.detach(),
            "vtpro_tgt": zero.detach(),
            "vtpro_bg": zero.detach(),
        }
        log_dict = {k: torch.nan_to_num(v, nan=0.0, posinf=10.0, neginf=10.0) for k, v in log_dict.items()}
        return total, loss_parts, log_dict
