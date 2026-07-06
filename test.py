# test.py

import argparse
import csv
import json
import os
import random

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
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
    """
    Return center-frame masks as [B,1,H,W].
    Supports:
      [B,H,W]
      [B,1,H,W]
      [B,T,H,W]
      [B,T,1,H,W]
      [B,1,T,H,W]
    """
    if masks.dim() == 3:
        return masks.unsqueeze(1)

    if masks.dim() == 4:
        if masks.shape[1] == 1:
            return masks
        center = (clip_t // 2) if (clip_t is not None and masks.shape[1] == clip_t) else (masks.shape[1] // 2)
        return masks[:, center:center + 1]

    if masks.dim() == 5:
        if masks.shape[2] == 1:
            center = (clip_t // 2) if (clip_t is not None and masks.shape[1] == clip_t) else (masks.shape[1] // 2)
            return masks[:, center]
        if masks.shape[1] == 1:
            center = (clip_t // 2) if (clip_t is not None and masks.shape[2] == clip_t) else (masks.shape[2] // 2)
            return masks[:, :, center]

    raise ValueError(f"Unsupported mask shape: {tuple(masks.shape)}")


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


def append_csv_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    clean_row = {}
    for k, v in row.items():
        if torch.is_tensor(v):
            v = v.detach().float().cpu().mean().item()
        if isinstance(v, (np.number,)):
            v = float(v)
        clean_row[k] = v

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(clean_row)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = {}
    for k, v in obj.items():
        if torch.is_tensor(v):
            v = v.detach().float().cpu().mean().item()
        if isinstance(v, np.number):
            v = float(v)
        clean[k] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)


def connected_components(binary):
    binary = np.asarray(binary).astype(bool)
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    neigh = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]

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
        comps.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "area": int(len(pix)),
            "pixels": arr,
        })

    return comps


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    return inter / (area_a + area_b - inter + 1e-6)


def object_metrics_single(prob, mask, thresh=0.5, iou_thr=0.5, ap_min_score=0.05):
    prob = np.asarray(prob, dtype=np.float32)
    mask = np.asarray(mask) > 0.5

    gt = connected_components(mask)

    pred_bin = prob > float(thresh)
    pred = connected_components(pred_bin)
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
            iou = bbox_iou(comp["bbox"], g["bbox"])
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

    # AP50
    ap_pred = connected_components(prob > float(ap_min_score))
    for comp in ap_pred:
        pix = comp["pixels"]
        comp["score"] = float(prob[pix[:, 0], pix[:, 1]].max()) if pix.size else 0.0

    ap_pred = sorted(ap_pred, key=lambda c: c.get("score", 0.0), reverse=True)

    if len(gt) == 0:
        ap50 = np.nan
    elif len(ap_pred) == 0:
        ap50 = 0.0
    else:
        matched_ap = set()
        tps, fps = [], []

        for comp in ap_pred:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt):
                if j in matched_ap:
                    continue
                iou = bbox_iou(comp["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_thr and best_j >= 0:
                matched_ap.add(best_j)
                tps.append(1.0)
                fps.append(0.0)
            else:
                tps.append(0.0)
                fps.append(1.0)

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
        "obj_tp": float(tp),
        "obj_fp": float(fp),
        "obj_fn": float(fn),
        "obj_precision": float(precision),
        "obj_recall": float(recall),
        "obj_f1": float(f1),
        "ap50": float(ap50) if np.isfinite(ap50) else np.nan,
    }


def build_model(args, device):
    model = KSTTipV2Net(
        in_ch=1,
        feat_ch=args.feat_ch,
        latent_dim=args.latent_dim,
        num_basis=args.num_basis,
        window_sizes=tuple(args.window_sizes),
        max_bg_modes=args.max_bg_modes,
        res_feat_ch=args.res_feat_ch,
        t_clip=args.T_clip,
        evidence_hidden_ch=args.evidence_hidden_ch,
        vtpro_embed_ch=args.vtpro_embed_ch,
        vtpro_num_scorm=args.vtpro_num_scorm,
        vtpro_out_ch=args.vtpro_out_ch,
        ktvm_tube_radius=args.ktvm_tube_radius,
    ).to(device)
    return model


def load_checkpoint(model, checkpoint_path, device, strict=False):
    if not checkpoint_path:
        raise ValueError("Please provide --checkpoint path.")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=strict)

    print(f"==> Loaded checkpoint: {checkpoint_path}")
    if isinstance(ckpt, dict):
        if "epoch" in ckpt:
            print(f"    checkpoint epoch: {ckpt['epoch']}")
        if "best_f1" in ckpt:
            print(f"    checkpoint best_f1: {ckpt['best_f1']}")

    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
        print(missing[:20])
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")
        print(unexpected[:20])


def save_batch_visuals(clips, masks_center, probs, preds, outputs, save_dir, start_idx, max_save):
    os.makedirs(save_dir, exist_ok=True)

    b, t = clips.shape[0], clips.shape[1]
    center = t // 2
    out_size = masks_center.shape[-2:]

    map_keys = [
        "dynamic_map",
        "modal_map",
        "violation_profile_map",
        "violation_tpro_map",
        "evidence_map",
        "fused_evidence_map",
        "ktvm_evidence_map",
        "ktvm_fixed_amp_map",
        "ktvm_amp_map",
        "ktvm_residual_gate_map",
        "ktvm_attention_focus_map",
        "ktvm_attention_entropy_map",
    ]

    saved = 0

    for i in range(b):
        if saved >= max_save:
            break

        global_idx = start_idx + i

        center_u8 = to_uint8_gray(clips[i, center, 0])
        gt_u8 = mask_to_uint8(masks_center[i, 0])
        pred_u8 = mask_to_uint8(preds[i, 0])
        prob_u8 = prob_to_uint8(probs[i, 0])

        panel_list = [
            center_u8,
            overlay_mask_on_gray(center_u8, gt_u8, color="green"),
            overlay_mask_on_gray(center_u8, pred_u8, color="red"),
            prob_u8,
        ]

        for key in map_keys:
            val = outputs.get(key, None)
            if not torch.is_tensor(val):
                continue
            val = val.detach()
            if val.dim() == 4:
                if val.shape[-2:] != out_size:
                    val = F.interpolate(val, size=out_size, mode="bilinear", align_corners=False)
                panel_list.append(to_uint8_gray(val[i, 0]))

        panel = make_panel(panel_list)
        iio.imwrite(os.path.join(save_dir, f"sample_{global_idx:05d}_panel.png"), panel)

        saved += 1

    return saved


def save_npz_batch(clips, masks_center, probs, preds, outputs, save_dir, start_idx, max_save):
    os.makedirs(save_dir, exist_ok=True)

    b, t = clips.shape[0], clips.shape[1]
    center = t // 2
    saved = 0

    for i in range(b):
        if saved >= max_save:
            break

        global_idx = start_idx + i

        pack = {
            "center_frame": clips[i, center, 0].detach().cpu().float().numpy(),
            "mask": masks_center[i, 0].detach().cpu().float().numpy(),
            "prob": probs[i, 0].detach().cpu().float().numpy(),
            "pred": preds[i, 0].detach().cpu().float().numpy(),
            "logits": outputs["logits"][i, 0].detach().cpu().float().numpy(),
        }

        extra_keys = [
            "dynamic_map",
            "modal_map",
            "violation_profile_map",
            "violation_tpro_map",
            "evidence_map",
            "fused_evidence_map",
            "ktvm_evidence_map",
            "ktvm_fixed_amp_map",
            "ktvm_amp_map",
            "ktvm_residual_gate_map",
            "ktvm_attention_focus_map",
            "ktvm_attention_entropy_map",
            "omega",
            "scale_weights",
        ]

        for key in extra_keys:
            val = outputs.get(key, None)
            if torch.is_tensor(val):
                try:
                    pack[key] = val[i].detach().cpu().float().numpy()
                except Exception:
                    pack[key] = val.detach().cpu().float().numpy()

        np.savez_compressed(os.path.join(save_dir, f"sample_{global_idx:05d}.npz"), **pack)
        saved += 1

    return saved


@torch.no_grad()
def test(model, loader, criterion, device, args):
    model.eval()

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    obj_tp = 0.0
    obj_fp = 0.0
    obj_fn = 0.0
    obj_ap_sum = 0.0
    obj_ap_count = 0

    total_loss = 0.0
    loss_count = 0
    num_images = 0

    details_csv = os.path.join(args.save_dir, "test_details.csv")
    metrics_json = os.path.join(args.save_dir, "test_metrics.json")
    metrics_csv = os.path.join(args.save_dir, "test_metrics.csv")
    vis_dir = os.path.join(args.save_dir, "vis")
    npz_dir = os.path.join(args.save_dir, "npz")

    os.makedirs(args.save_dir, exist_ok=True)

    saved_vis = 0
    saved_npz = 0

    pbar = tqdm(loader, desc="Test", ncols=120)

    for step, (clips, masks) in enumerate(pbar, start=1):
        clips = clips.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        masks_center = get_center_mask(masks, clip_t=clips.shape[1]).float()

        if args.amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(clips)
        else:
            outputs = model(clips)

        logits = outputs["logits"]
        probs = torch.sigmoid(logits)

        if masks_center.shape[-2:] != probs.shape[-2:]:
            masks_center = F.interpolate(masks_center, size=probs.shape[-2:], mode="nearest")

        preds = (probs > args.threshold).float()

        if args.compute_loss and criterion is not None:
            loss, _, _ = criterion(outputs, masks)
            if torch.isfinite(loss):
                total_loss += float(loss.item())
                loss_count += 1

        batch_tp = (preds * masks_center).sum().item()
        batch_fp = (preds * (1.0 - masks_center)).sum().item()
        batch_fn = ((1.0 - preds) * masks_center).sum().item()

        total_tp += batch_tp
        total_fp += batch_fp
        total_fn += batch_fn

        probs_np = probs.detach().cpu().float().numpy()
        masks_np = masks_center.detach().cpu().float().numpy()

        batch_obj_tp = 0.0
        batch_obj_fp = 0.0
        batch_obj_fn = 0.0
        batch_ap_sum = 0.0
        batch_ap_count = 0

        for bi in range(probs_np.shape[0]):
            obj_m = object_metrics_single(
                probs_np[bi, 0],
                masks_np[bi, 0],
                thresh=args.threshold,
                iou_thr=args.iou_thr,
                ap_min_score=args.ap_min_score,
            )

            batch_obj_tp += obj_m["obj_tp"]
            batch_obj_fp += obj_m["obj_fp"]
            batch_obj_fn += obj_m["obj_fn"]

            if np.isfinite(obj_m["ap50"]):
                batch_ap_sum += obj_m["ap50"]
                batch_ap_count += 1

            append_csv_row(details_csv, {
                "sample_idx": num_images + bi,
                "obj_tp": obj_m["obj_tp"],
                "obj_fp": obj_m["obj_fp"],
                "obj_fn": obj_m["obj_fn"],
                "obj_precision": obj_m["obj_precision"],
                "obj_recall": obj_m["obj_recall"],
                "obj_f1": obj_m["obj_f1"],
                "ap50": obj_m["ap50"] if np.isfinite(obj_m["ap50"]) else "",
                "prob_max": float(probs_np[bi, 0].max()),
                "prob_mean": float(probs_np[bi, 0].mean()),
                "mask_pixels": float(masks_np[bi, 0].sum()),
                "pred_pixels": float((probs_np[bi, 0] > args.threshold).sum()),
            })

        obj_tp += batch_obj_tp
        obj_fp += batch_obj_fp
        obj_fn += batch_obj_fn
        obj_ap_sum += batch_ap_sum
        obj_ap_count += batch_ap_count

        bsz = clips.shape[0]

        if args.save_vis and saved_vis < args.max_vis:
            saved_vis += save_batch_visuals(
                clips=clips,
                masks_center=masks_center,
                probs=probs,
                preds=preds,
                outputs=outputs,
                save_dir=vis_dir,
                start_idx=num_images,
                max_save=args.max_vis - saved_vis,
            )

        if args.save_npz and saved_npz < args.max_npz:
            saved_npz += save_npz_batch(
                clips=clips,
                masks_center=masks_center,
                probs=probs,
                preds=preds,
                outputs=outputs,
                save_dir=npz_dir,
                start_idx=num_images,
                max_save=args.max_npz - saved_npz,
            )

        num_images += bsz

        pixel_precision = total_tp / (total_tp + total_fp + 1e-6)
        pixel_recall = total_tp / (total_tp + total_fn + 1e-6)
        pixel_f1 = 2.0 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall + 1e-6)

        obj_precision = obj_tp / (obj_tp + obj_fp + 1e-6)
        obj_recall = obj_tp / (obj_tp + obj_fn + 1e-6)
        obj_f1 = 2.0 * obj_precision * obj_recall / (obj_precision + obj_recall + 1e-6)

        pbar.set_postfix({
            "PixF1": f"{pixel_f1:.4f}",
            "ObjF1": f"{obj_f1:.4f}",
            "ObjR": f"{obj_recall:.4f}",
            "FA/img": f"{obj_fp / max(num_images, 1):.3f}",
        })

    pbar.close()

    pixel_precision = total_tp / (total_tp + total_fp + 1e-6)
    pixel_recall = total_tp / (total_tp + total_fn + 1e-6)
    pixel_f1 = 2.0 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall + 1e-6)

    obj_precision = obj_tp / (obj_tp + obj_fp + 1e-6)
    obj_recall = obj_tp / (obj_tp + obj_fn + 1e-6)
    obj_f1 = 2.0 * obj_precision * obj_recall / (obj_precision + obj_recall + 1e-6)

    ap50 = obj_ap_sum / max(obj_ap_count, 1)
    fa_per_img = obj_fp / max(num_images, 1)

    metrics = {
        "num_images": int(num_images),
        "threshold": float(args.threshold),
        "pixel_precision": float(pixel_precision),
        "pixel_recall": float(pixel_recall),
        "pixel_f1": float(pixel_f1),
        "obj_precision": float(obj_precision),
        "obj_recall": float(obj_recall),
        "obj_f1": float(obj_f1),
        "ap50": float(ap50),
        "fa_per_img": float(fa_per_img),
        "obj_tp": float(obj_tp),
        "obj_fp": float(obj_fp),
        "obj_fn": float(obj_fn),
    }

    if args.compute_loss:
        metrics["test_loss"] = total_loss / max(loss_count, 1)

    save_json(metrics_json, metrics)
    append_csv_row(metrics_csv, metrics)

    print("\n========== Test Results ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("==================================")
    print(f"Saved metrics: {metrics_json}")
    print(f"Saved details: {details_csv}")
    if args.save_vis:
        print(f"Saved visuals: {vis_dir}")
    if args.save_npz:
        print(f"Saved npz: {npz_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--root", type=str, default="DAUB")
    parser.add_argument("--train_meta", type=str, default="")
    parser.add_argument("--val_meta", type=str, default="DAUB/test.txt")
    parser.add_argument("--T_clip", type=int)

    # Model args, must match training
    parser.add_argument("--window_sizes", type=int, nargs="+")
    parser.add_argument("--max_bg_modes", type=int)
    parser.add_argument("--feat_ch", type=int)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--num_basis", type=int)
    parser.add_argument("--res_feat_ch", type=int)
    parser.add_argument("--evidence_hidden_ch", type=int )
    parser.add_argument("--vtpro_embed_ch", type=int)
    parser.add_argument("--vtpro_num_scorm", type=int)
    parser.add_argument("--vtpro_out_ch", type=int)
    parser.add_argument("--ktvm_tube_radius", type=int)

    # Checkpoint / loader
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_bck_ktvm/best_f1.pth")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--cpu_num_threads", type=int)
    parser.add_argument("--seed", type=int)

    # Test settings
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--iou_thr", type=float)
    parser.add_argument("--ap_min_score", type=float)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compute_loss", action="store_true")

    # Save
    parser.add_argument("--save_dir", type=str, default="./test_results_bck")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--max_vis", type=int)
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--max_npz", type=int)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    if args.cpu_num_threads and args.cpu_num_threads > 0:
        torch.set_num_threads(args.cpu_num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> Device: {device}")

    # build_dataset 必须传 train_meta 和 val_meta；
    # 测试时只使用 val_set，所以 train_meta 为空时直接复用 val_meta。
    train_meta = args.train_meta if args.train_meta else args.val_meta

    print("==> Building test dataset...")
    _, test_set = build_dataset(
        dataset_name=args.dataset_name,
        root=args.root,
        train_meta=train_meta,
        val_meta=args.val_meta,
        T_clip=args.T_clip,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    print(f"==> Test samples: {len(test_set)}")

    print("==> Building model...")
    model = build_model(args, device)

    load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        strict=args.strict,
    )

    criterion = None
    if args.compute_loss:
        criterion = KSTTipV2Loss(
            lambda_bg=args.lambda_bg,
            lambda_vtpro=args.lambda_vtpro,
            lambda_sig=args.lambda_sig,
            lambda_stab=args.lambda_stab,
            target_margin=args.target_margin,
        )

    test(model, test_loader, criterion, device, args)


if __name__ == "__main__":
    main()