import argparse
import csv
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from koopman_dataset_factory import build_dataset
from kst_net_tip_v2 import KSTTipV2Net
from losses_kst_tip_v2 import KSTTipV2Loss


def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def collate_fn(batch):
    clips, masks = zip(*batch)
    return torch.stack(clips, dim=0), torch.stack(masks, dim=0)


def get_center_mask(masks, clip_t=None):
    """Return center-frame masks as [B,1,H,W]. Supports mask sequences."""
    if masks.dim() == 3:
        return masks.unsqueeze(1)
    if masks.dim() == 4:
        if masks.shape[1] == 1:
            return masks
        # [B,T,H,W]
        center = (clip_t // 2) if (clip_t is not None and masks.shape[1] == clip_t) else (masks.shape[1] // 2)
        return masks[:, center:center + 1]
    if masks.dim() == 5:
        if masks.shape[2] == 1:
            # [B,T,1,H,W]
            center = (clip_t // 2) if (clip_t is not None and masks.shape[1] == clip_t) else (masks.shape[1] // 2)
            return masks[:, center]
        if masks.shape[1] == 1:
            # [B,1,T,H,W]
            center = (clip_t // 2) if (clip_t is not None and masks.shape[2] == clip_t) else (masks.shape[2] // 2)
            return masks[:, :, center]
    raise ValueError(f"Unsupported mask shape for center-frame extraction: {tuple(masks.shape)}")


def to_uint8_gray(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def mask_to_uint8(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    return (np.nan_to_num(x, nan=0.0) > 0.5).astype(np.uint8) * 255


def prob_to_uint8(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def gray_to_rgb(gray_u8):
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)


def overlay_mask_on_gray(gray_u8, mask_u8, color="red"):
    rgb = gray_to_rgb(gray_u8).copy()
    m = mask_u8 > 127
    if color == "red":
        rgb[m, 0] = 255
        rgb[m, 1] = (rgb[m, 1] * 0.3).astype(np.uint8)
        rgb[m, 2] = (rgb[m, 2] * 0.3).astype(np.uint8)
    else:
        rgb[m, 1] = 255
        rgb[m, 0] = (rgb[m, 0] * 0.3).astype(np.uint8)
        rgb[m, 2] = (rgb[m, 2] * 0.3).astype(np.uint8)
    return rgb


def make_panel(images, pad=6):
    proc = []
    for im in images:
        if im.ndim == 2:
            im = gray_to_rgb(im)
        proc.append(im)
    h = max(x.shape[0] for x in proc)
    proc2 = []
    for im in proc:
        if im.shape[0] != h:
            im = np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)), mode="constant")
        proc2.append(im)
    canvas = []
    for i, im in enumerate(proc2):
        canvas.append(im)
        if i != len(proc2) - 1:
            canvas.append(np.ones((h, pad, 3), dtype=np.uint8) * 255)
    return np.concatenate(canvas, axis=1)


def interp_map(x, out_size, mode="bilinear"):
    if x is None:
        return None
    return torch.nn.functional.interpolate(x, size=out_size, mode=mode, align_corners=False if mode == "bilinear" else None)


def _safe_float(x):
    if torch.is_tensor(x):
        x = x.detach().float().cpu()
        if x.numel() == 0:
            return 0.0
        x = x.mean().item()
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v


def append_csv_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {k: _safe_float(v) if isinstance(v, (int, float, np.number)) or torch.is_tensor(v) else v for k, v in row.items()}
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = {}
    for k, v in obj.items():
        if torch.is_tensor(v) or isinstance(v, (int, float, np.number)):
            clean[k] = _safe_float(v)
        else:
            clean[k] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)


def _sample_pixel_metrics(prob, mask, thresh=0.5):
    pred = (prob > thresh).astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)
    tp = float((pred * mask).sum())
    fp = float((pred * (1.0 - mask)).sum())
    fn = float(((1.0 - pred) * mask).sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _connected_components(binary):
    """Small dependency-free connected components for validation metrics."""
    binary = np.asarray(binary).astype(bool)
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    ys, xs = np.where(binary)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for dy, dx in neigh:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and binary[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        arr = np.asarray(pix, dtype=np.int32)
        y1, x1 = arr.min(axis=0)
        y2, x2 = arr.max(axis=0) + 1
        comps.append({"bbox": (int(x1), int(y1), int(x2), int(y2)), "area": int(len(pix)), "pixels": arr})
    return comps

def _bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)

def _object_metrics_single(prob, mask, thresh=0.5, iou_thr=0.5, ap_min_score=0.05):
    """Object-level precision/recall/F1 and AP50 from mask components."""
    prob = np.asarray(prob, dtype=np.float32)
    mask = (np.asarray(mask) > 0.5)
    gt = _connected_components(mask)

    pred_bin = prob > float(thresh)
    pred = _connected_components(pred_bin)
    for comp in pred:
        pix = comp["pixels"]
        comp["score"] = float(prob[pix[:, 0], pix[:, 1]].max()) if pix.size else 0.0

    matched = set()
    tp = 0
    fp = 0
    for comp in sorted(pred, key=lambda c: c.get("score", 0.0), reverse=True):
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt):
            if j in matched:
                continue
            iou = _bbox_iou(comp["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp += 1
            matched.add(best_j)
        else:
            fp += 1
    fn = max(0, len(gt) - tp)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)

    # AP50: rank all predicted components above a low threshold by component score.
    ap_pred = _connected_components(prob > float(ap_min_score))
    for comp in ap_pred:
        pix = comp["pixels"]
        comp["score"] = float(prob[pix[:, 0], pix[:, 1]].max()) if pix.size else 0.0
    ap_pred = sorted(ap_pred, key=lambda c: c.get("score", 0.0), reverse=True)
    if len(gt) == 0:
        ap50 = np.nan
    else:
        matched_ap = set()
        tps, fps = [], []
        for comp in ap_pred:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt):
                if j in matched_ap:
                    continue
                iou = _bbox_iou(comp["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                matched_ap.add(best_j)
                tps.append(1.0)
                fps.append(0.0)
            else:
                tps.append(0.0)
                fps.append(1.0)
        if len(tps) == 0:
            ap50 = 0.0
        else:
            tp_c = np.cumsum(np.asarray(tps, dtype=np.float32))
            fp_c = np.cumsum(np.asarray(fps, dtype=np.float32))
            rec = tp_c / (len(gt) + 1e-6)
            prec = tp_c / (tp_c + fp_c + 1e-6)
            mrec = np.concatenate([[0.0], rec, [1.0]])
            mpre = np.concatenate([[0.0], prec, [0.0]])
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap50 = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {
        "obj_tp": float(tp), "obj_fp": float(fp), "obj_fn": float(fn),
        "obj_precision": float(precision), "obj_recall": float(recall), "obj_f1": float(f1),
        "ap50": float(ap50) if np.isfinite(ap50) else np.nan,
    }


def save_temporal_affinity_heatmaps(outputs, save_path):
    """Save KRTA temporal affinity heatmaps.

    Supports either:
      - outputs['ktvm_temporal_affinity']: [1,T,T] or [N,T,T]
      - outputs['violation_scorm']: legacy third return from RTVM, now used as affinity_vis
    """
    affinity = outputs.get("ktvm_temporal_affinity", None)
    if affinity is None:
        affinity = outputs.get("violation_scorm", None)
    if affinity is None or not torch.is_tensor(affinity):
        return

    affinity = affinity.detach().cpu().float()
    if affinity.dim() == 2:
        affinity = affinity.unsqueeze(0)
    if affinity.dim() != 3:
        return

    n = min(4, affinity.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), dpi=180)
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        im = ax.imshow(affinity[i].numpy(), aspect="auto", cmap="turbo")
        ax.set_title(f"KRTA Affinity {i + 1}")
        ax.set_xlabel("input time")
        ax.set_ylabel("output time")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


@torch.no_grad()
def save_val_visuals(clips, masks, outputs, save_dir, epoch, max_items=2):
    epoch_dir = os.path.join(save_dir, "val_vis", f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)
    probs = torch.sigmoid(outputs["logits"])
    preds = (probs > 0.5).float()
    center_masks = get_center_mask(masks, clip_t=clips.shape[1])
    out_size = center_masks.shape[-2:]

    dyn = interp_map(outputs.get("dynamic_map"), out_size)
    vprof = interp_map(outputs.get("violation_profile_map"), out_size)
    vtpro = interp_map(outputs.get("violation_tpro_map"), out_size)
    evidence = interp_map(outputs.get("evidence_map"), out_size)
    fused = interp_map(outputs.get("fused_evidence_map"), out_size)
    ktvm_evi = interp_map(outputs.get("ktvm_evidence_map"), out_size)
    ktvm_fixed_amp = interp_map(outputs.get("ktvm_fixed_amp_map"), out_size)
    ktvm_amp = interp_map(outputs.get("ktvm_amp_map"), out_size)
    ktvm_gate = interp_map(outputs.get("ktvm_residual_gate_map"), out_size)
    ktvm_focus = interp_map(outputs.get("ktvm_attention_focus_map"), out_size)
    ktvm_entropy = interp_map(outputs.get("ktvm_attention_entropy_map"), out_size)

    b, t = clips.shape[0], clips.shape[1]
    center = t // 2
    for i in range(min(b, max_items)):
        base = os.path.join(epoch_dir, f"sample_{i:02d}")
        center_u8 = to_uint8_gray(clips[i, center, 0])
        gt_u8 = mask_to_uint8(center_masks[i, 0])
        pred_u8 = mask_to_uint8(preds[i, 0])
        prob_u8 = prob_to_uint8(probs[i, 0])
        panel_list = [
            center_u8,
            overlay_mask_on_gray(center_u8, gt_u8, color="green"),
            overlay_mask_on_gray(center_u8, pred_u8, color="red"),
            prob_u8,
        ]
        for arr in [dyn, vprof, vtpro, evidence, fused, ktvm_evi, ktvm_fixed_amp, ktvm_amp, ktvm_gate, ktvm_focus, ktvm_entropy]:
            if arr is not None:
                panel_list.append(to_uint8_gray(arr[i, 0]))
        panel = make_panel(panel_list)
        iio.imwrite(base + "_panel.png", panel)
    save_temporal_affinity_heatmaps(outputs, os.path.join(epoch_dir, "krta_temporal_affinity.png"))


def save_val_numeric_data(clips, masks, outputs, save_dir, epoch, max_items=2):
    epoch_dir = os.path.join(save_dir, "val_data", f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)
    probs = torch.sigmoid(outputs["logits"]).detach().cpu().float()
    preds = (probs > 0.5).float()
    clips_cpu = clips.detach().cpu().float()
    center_masks = get_center_mask(masks, clip_t=clips.shape[1])
    masks_cpu = center_masks.detach().cpu().float()

    map_keys = [
        "dynamic_map", "modal_map", "violation_profile_map", "violation_tpro_map", "evidence_map",
        "fused_evidence_map", "evidence_feature_map",
        "mode_count_map", "consistency_map", "modal_error_map",
        "ktvm_fixed_amp_map", "ktvm_amp_map", "ktvm_attention_entropy_map", "ktvm_attention_focus_map",
        "ktvm_evidence_map", "ktvm_residual_gate_map", "ktvm_evidence_logits",
    ]
    seq_keys = [
        "violation_profile", "violation_tpro_score_seq", "evidence_score_seq",
        "dynamic_seq_raw", "fw_seq_raw", "bw_seq_raw",
    ]

    stats_csv = os.path.join(epoch_dir, "sample_metrics.csv")
    for i in range(min(clips_cpu.shape[0], max_items)):
        prob_i = probs[i, 0].numpy()
        mask_i = masks_cpu[i, 0].numpy()
        pred_i = preds[i, 0].numpy()
        center_idx = clips_cpu.shape[1] // 2
        pack = {
            "center_frame": clips_cpu[i, center_idx, 0].numpy(),
            "mask": mask_i,
            "prob": prob_i,
            "pred": pred_i,
            "logits": outputs["logits"][i, 0].detach().cpu().float().numpy(),
        }
        for key in map_keys:
            val = outputs.get(key, None)
            if torch.is_tensor(val):
                pack[key] = val[i].detach().cpu().float().numpy()
        for key in seq_keys:
            val = outputs.get(key, None)
            if torch.is_tensor(val):
                pack[key] = val[i].detach().cpu().float().numpy()
        if torch.is_tensor(outputs.get("omega", None)):
            pack["omega"] = outputs["omega"][i].detach().cpu().float().numpy()
        if torch.is_tensor(outputs.get("violation_scorm", None)):
            pack["violation_scorm"] = outputs["violation_scorm"].detach().cpu().float().numpy()
        if torch.is_tensor(outputs.get("ktvm_temporal_affinity", None)):
            pack["ktvm_temporal_affinity"] = outputs["ktvm_temporal_affinity"].detach().cpu().float().numpy()
        np.savez_compressed(os.path.join(epoch_dir, f"sample_{i:02d}_data.npz"), **pack)
        row = {
            "epoch": epoch,
            "sample_idx": i,
            "prob_mean": float(np.nanmean(prob_i)),
            "prob_max": float(np.nanmax(prob_i)),
            "mask_pixels": float(mask_i.sum()),
            "pred_pixels": float(pred_i.sum()),
            **_sample_pixel_metrics(prob_i, mask_i),
        }
        append_csv_row(stats_csv, row)


def find_nonfinite_grad(model):
    for name, p in model.named_parameters():
        if p.grad is not None and (not torch.isfinite(p.grad).all()):
            return name
    return None


def save_checkpoint(path, epoch, model, optimizer, scaler, args, val_metrics, best_f1):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "val_metrics": val_metrics,
        "best_f1": best_f1,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)


def resume_from_checkpoint(resume_path, model, optimizer, scaler, device, override_lr=None):
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if "optimizer" in ckpt and ckpt["optimizer"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            print(f"[WARN] Failed to load optimizer state: {exc}")
    if scaler is not None and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as exc:
            print(f"[WARN] Failed to load AMP scaler state: {exc}")
    if override_lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = override_lr
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_f1 = float(ckpt.get("best_f1", ckpt.get("val_metrics", {}).get("f1", -1.0)))
    return start_epoch, best_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch=0, epochs=0, save_dir=None, save_vis=False, max_vis=2, save_data=False, max_data=2, save_details=True):
    model.eval()
    total_tp = total_fp = total_fn = 0.0
    obj_tp = obj_fp = obj_fn = 0.0
    obj_ap_sum = 0.0
    obj_ap_count = 0
    num_images = 0
    total_loss = 0.0
    log_sums = {}
    log_count = 0
    details_path = None
    if save_details and save_dir is not None:
        details_dir = os.path.join(save_dir, "val_details")
        os.makedirs(details_dir, exist_ok=True)
        details_path = os.path.join(details_dir, f"epoch_{epoch:03d}_batches.csv")
        if os.path.exists(details_path):
            os.remove(details_path)

    pbar = tqdm(loader, desc=f"Val   [{epoch:03d}/{epochs:03d}]", ncols=120, leave=False)
    vis_saved = False
    data_saved = False
    for step, (clips, masks) in enumerate(pbar, start=1):
        clips = clips.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        masks_center = get_center_mask(masks, clip_t=clips.shape[1])
        outputs = model(clips)
        loss, _, log_dict = criterion(outputs, masks)
        loss_value = float(loss.item()) if torch.isfinite(loss) else 0.0
        total_loss += loss_value
        for k, v in log_dict.items():
            log_sums[k] = log_sums.get(k, 0.0) + _safe_float(v)
        log_count += 1

        probs = torch.sigmoid(outputs["logits"])
        preds = (probs > 0.5).float()
        batch_tp = (preds * masks_center).sum().item()
        batch_fp = (preds * (1 - masks_center)).sum().item()
        batch_fn = ((1 - preds) * masks_center).sum().item()
        total_tp += batch_tp
        total_fp += batch_fp
        total_fn += batch_fn

        probs_np = probs.detach().cpu().float().numpy()
        masks_np = masks_center.detach().cpu().float().numpy()
        batch_obj_tp = batch_obj_fp = batch_obj_fn = 0.0
        batch_ap_sum = 0.0
        batch_ap_count = 0
        for bi in range(probs_np.shape[0]):
            obj_m = _object_metrics_single(probs_np[bi, 0], masks_np[bi, 0], thresh=0.5, iou_thr=0.5)
            batch_obj_tp += obj_m["obj_tp"]
            batch_obj_fp += obj_m["obj_fp"]
            batch_obj_fn += obj_m["obj_fn"]
            if np.isfinite(obj_m["ap50"]):
                batch_ap_sum += obj_m["ap50"]
                batch_ap_count += 1
        obj_tp += batch_obj_tp
        obj_fp += batch_obj_fp
        obj_fn += batch_obj_fn
        obj_ap_sum += batch_ap_sum
        obj_ap_count += batch_ap_count
        num_images += probs_np.shape[0]

        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        obj_precision = obj_tp / (obj_tp + obj_fp + 1e-6)
        obj_recall = obj_tp / (obj_tp + obj_fn + 1e-6)
        obj_f1 = 2 * obj_precision * obj_recall / (obj_precision + obj_recall + 1e-6)
        ap50 = obj_ap_sum / max(obj_ap_count, 1)
        fa_per_img = obj_fp / max(num_images, 1)

        if details_path is not None:
            b_precision = batch_tp / (batch_tp + batch_fp + 1e-6)
            b_recall = batch_tp / (batch_tp + batch_fn + 1e-6)
            b_f1 = 2 * b_precision * b_recall / (b_precision + b_recall + 1e-6)
            b_obj_precision = batch_obj_tp / (batch_obj_tp + batch_obj_fp + 1e-6)
            b_obj_recall = batch_obj_tp / (batch_obj_tp + batch_obj_fn + 1e-6)
            b_obj_f1 = 2 * b_obj_precision * b_obj_recall / (b_obj_precision + b_obj_recall + 1e-6)
            append_csv_row(details_path, {"epoch": epoch, "batch": step, "loss": loss_value, "tp": batch_tp, "fp": batch_fp, "fn": batch_fn, "precision": b_precision, "recall": b_recall, "f1": b_f1, "obj_tp": batch_obj_tp, "obj_fp": batch_obj_fp, "obj_fn": batch_obj_fn, "obj_precision": b_obj_precision, "obj_recall": b_obj_recall, "obj_f1": b_obj_f1, "ap50": batch_ap_sum / max(batch_ap_count, 1)})

        pbar.set_postfix({"loss": f"{total_loss / step:.4f}", "P": f"{precision:.4f}", "R": f"{recall:.4f}", "F1": f"{f1:.4f}", "ObjF1": f"{obj_f1:.4f}", "AP50": f"{ap50:.4f}"})
        if save_vis and not vis_saved and save_dir is not None:
            save_val_visuals(clips, masks, outputs, save_dir, epoch, max_items=max_vis)
            vis_saved = True
        if save_data and not data_saved and save_dir is not None:
            save_val_numeric_data(clips, masks, outputs, save_dir, epoch, max_items=max_data)
            data_saved = True
    pbar.close()

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    obj_precision = obj_tp / (obj_tp + obj_fp + 1e-6)
    obj_recall = obj_tp / (obj_tp + obj_fn + 1e-6)
    obj_f1 = 2 * obj_precision * obj_recall / (obj_precision + obj_recall + 1e-6)
    ap50 = obj_ap_sum / max(obj_ap_count, 1)
    fa_per_img = obj_fp / max(num_images, 1)
    metrics = {"val_loss": total_loss / max(len(loader), 1), "precision": precision, "recall": recall, "f1": f1, "obj_precision": obj_precision, "obj_recall": obj_recall, "obj_f1": obj_f1, "ap50": ap50, "fa_per_img": fa_per_img}
    if log_count > 0:
        for k, v in log_sums.items():
            metrics[f"val_{k}"] = v / log_count
    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, epoch=0, epochs=0, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    skipped = 0
    log_sums = {}
    log_count = 0
    pbar = tqdm(loader, desc=f"Train [{epoch:03d}/{epochs:03d}]", ncols=120, leave=False)
    for step, (clips, masks) in enumerate(pbar, start=1):
        clips = clips.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        try:
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(clips)
                    loss, _, log_dict = criterion(outputs, masks)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                bad_grad = find_nonfinite_grad(model)
                if bad_grad is not None:
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(clips)
                loss, _, log_dict = criterion(outputs, masks)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                bad_grad = find_nonfinite_grad(model)
                if bad_grad is not None:
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
        except RuntimeError as e:
            msg = str(e).lower()
            if "linalg" in msg or "svd" in msg or "eigh" in msg or "eig" in msg:
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                continue
            raise

        total_loss += float(loss.item())
        for k, v in log_dict.items():
            log_sums[k] = log_sums.get(k, 0.0) + _safe_float(v)
        log_count += 1
        avg_loss = total_loss / max(step - skipped, 1)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg": f"{avg_loss:.4f}",
            "det": f"{_safe_float(log_dict.get('loss_det', 0.0)):.4f}",
            "bg": f"{_safe_float(log_dict.get('loss_bg', 0.0)):.4f}",
            "stab": f"{_safe_float(log_dict.get('loss_stab', 0.0)):.5f}",
            "skip": skipped,
        })
    pbar.close()
    avg_loss = total_loss / max(len(loader) - skipped, 1)
    avg_logs = {f"train_{k}": v / max(log_count, 1) for k, v in log_sums.items()}
    avg_logs["train_skipped"] = skipped
    return avg_loss, avg_logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--root", type=str, default="DAUB")
    parser.add_argument("--train_meta", type=str, default="DAUB/train.txt")
    parser.add_argument("--val_meta", type=str, default="DAUB/test.txt")
    parser.add_argument("--T_clip", type=int)
    parser.add_argument("--window_sizes", type=int, nargs="+")
    parser.add_argument("--max_bg_modes", type=int)  # compatibility only
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_bck_ktvm")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--feat_ch", type=int)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--num_basis", type=int)
    parser.add_argument("--grad_clip", type=float)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--override_lr", type=float, default=None)
    parser.add_argument("--reset_best", action="store_true")
    parser.add_argument("--lambda_bg", type=float)
    parser.add_argument("--lambda_vtpro", type=float)  # deprecated; KRTA uses no signature loss
    parser.add_argument("--lambda_sig", type=float)  # deprecated; ignored by KRTA loss
    parser.add_argument("--lambda_stab", type=float)
    parser.add_argument("--target_margin", type=float)
    parser.add_argument("--evidence_hidden_ch", type=int)
    parser.add_argument("--vtpro_embed_ch", type=int)
    parser.add_argument("--vtpro_num_scorm", type=int)
    parser.add_argument("--vtpro_out_ch", type=int)
    parser.add_argument("--ktvm_tube_radius", type=int)  # compatibility only; KRTA uses no motion tube
    parser.add_argument("--cpu_num_threads", type=int)
    parser.add_argument("--save_data_every", type=int)
    parser.add_argument("--save_data_max_items", type=int)
    parser.add_argument("--save_vis_every", type=int)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(3407)
    if args.cpu_num_threads and args.cpu_num_threads > 0:
        torch.set_num_threads(args.cpu_num_threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("==> Building datasets...")
    train_set, val_set = build_dataset(
        dataset_name=args.dataset_name,
        root=args.root,
        train_meta=args.train_meta,
        val_meta=args.val_meta,
        T_clip=args.T_clip,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn, worker_init_fn=worker_init_fn)

    print("==> Building model...")
    model = KSTTipV2Net(
        in_ch=1,
        feat_ch=args.feat_ch,
        latent_dim=args.latent_dim,
        num_basis=args.num_basis,
        window_sizes=tuple(args.window_sizes),
        max_bg_modes=args.max_bg_modes,
        res_feat_ch=32,
        t_clip=args.T_clip,
        evidence_hidden_ch=args.evidence_hidden_ch,
        vtpro_embed_ch=args.vtpro_embed_ch,
        vtpro_num_scorm=args.vtpro_num_scorm,
        vtpro_out_ch=args.vtpro_out_ch,
        ktvm_tube_radius=args.ktvm_tube_radius,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = KSTTipV2Loss(
        lambda_bg=args.lambda_bg,
        lambda_vtpro=args.lambda_vtpro,
        lambda_sig=args.lambda_sig,
        lambda_stab=args.lambda_stab,
        target_margin=args.target_margin,
    )
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device == "cuda") else None

    start_epoch, best_f1 = 1, -1.0
    if args.resume:
        start_epoch, best_f1 = resume_from_checkpoint(args.resume, model, optimizer, scaler, device, override_lr=args.override_lr)
        if args.reset_best:
            best_f1 = -1.0

    print("==> Start training")
    print("Version: BCK-Net + KMCP + KRTA-no-motion + pair-wise mask Koopman loss")
    print(f"T_clip={args.T_clip}, window_sizes={args.window_sizes}")
    print(f"lambda_bg={args.lambda_bg}, lambda_stab={args.lambda_stab}  (signature loss disabled)")

    metrics_csv = os.path.join(args.save_dir, "metrics_history.csv")
    metrics_dir = os.path.join(args.save_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_logs = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler, epoch=epoch, epochs=args.epochs, grad_clip=args.grad_clip)
        save_vis = args.save_vis_every > 0 and (epoch % args.save_vis_every == 0)
        save_data = args.save_data_every > 0 and (epoch % args.save_data_every == 0)
        val_metrics = evaluate(model, val_loader, criterion, device, epoch=epoch, epochs=args.epochs, save_dir=args.save_dir, save_vis=save_vis, max_vis=2, save_data=save_data, max_data=args.save_data_max_items, save_details=True)
        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        is_best = val_metrics["f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["f1"]

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "time_sec": dt,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
            "best_f1": best_f1,
            "is_best": int(is_best),
            **train_logs,
            **val_metrics,
        }
        append_csv_row(metrics_csv, epoch_record)
        save_json(os.path.join(metrics_dir, f"epoch_{epoch:03d}.json"), epoch_record)
        save_json(os.path.join(metrics_dir, "latest.json"), epoch_record)

        tqdm.write(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | val_loss={val_metrics['val_loss']:.4f} | "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} F1={val_metrics['f1']:.4f} | "
            f"best={best_f1:.4f} | time={dt:.1f}s"
        )
        if save_data:
            tqdm.write(f"    saved val data: {os.path.join(args.save_dir, 'val_data', f'epoch_{epoch:03d}')}")

        save_checkpoint(os.path.join(args.save_dir, "latest.pth"), epoch, model, optimizer, scaler, args, val_metrics, best_f1)
        if is_best:
            save_checkpoint(os.path.join(args.save_dir, "best_f1.pth"), epoch, model, optimizer, scaler, args, val_metrics, best_f1)
            tqdm.write(f"==> New best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()

