#!/usr/bin/env python3
"""
Diagnostic and trainer for nc2000-bot's antisymmetric EvalResidualWeights (32 -> 16 -> 1)
and rank-4 bilinear RolloutResidualWeights (12 x 6 -> 4).
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvalResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_w1_syn = nn.Parameter(torch.rand(8, 16) * 0.20)
        self.raw_w1_vul = nn.Parameter(torch.rand(8, 16) * 0.20)
        self.raw_b1 = nn.Parameter(torch.zeros(16))
        self.raw_w2_syn = nn.Parameter(torch.full((8,), 0.05))
        self.raw_w2_vul = nn.Parameter(torch.full((8,), 0.05))

    def effective_params(self):
        w1 = torch.zeros((16, 32), dtype=self.raw_w1_syn.dtype, device=self.raw_w1_syn.device)
        w1[:8, :16] = torch.clamp(self.raw_w1_syn, min=0.0)
        w1[8:, 16:] = torch.clamp(self.raw_w1_vul, min=0.0)
        w1[:, 29:31] = 0.0  # x[29..31] reserved zero
        b1 = -torch.clamp(-self.raw_b1, min=0.0)  # b1 <= 0 so f(0) == 0 identically
        w2 = torch.cat(
            [
                torch.clamp(self.raw_w2_syn, min=0.0),
                -torch.clamp(self.raw_w2_vul, min=0.0),
            ],
            dim=0,
        )
        return w1, b1, w2

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        w1, b1, w2 = self.effective_params()
        h = F.relu(F.linear(x, w1, b1))
        return torch.mv(h, w2)

    def forward_sym(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.forward_single(x0) - self.forward_single(x1)

    def export_dict(self, active: bool = True):
        with torch.no_grad():
            w1, b1, w2 = self.effective_params()
        return {
            "active": active,
            "w1": w1.detach().cpu().numpy().astype(np.float32).tolist(),
            "b1": b1.detach().cpu().numpy().astype(np.float32).tolist(),
            "w2": w2.detach().cpu().numpy().astype(np.float32).tolist(),
        }


class RolloutResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_proj = nn.Parameter(torch.randn(4, 12) * 0.05)
        self.v_proj = nn.Parameter(torch.randn(4, 6) * 0.05)
        self.act_bias = nn.Parameter(torch.zeros(6))

    def forward_scores(
        self,
        phi: torch.Tensor,         # [B, 12]
        act_feats: torch.Tensor,   # [B, 7, 6]
        act_mask: torch.Tensor,    # [B, 7] bool (has non-zero act_feats)
        base_scores: torch.Tensor, # [B, 7]
    ) -> torch.Tensor:
        # u = phi @ u_proj^T -> [B, 4]
        u = F.linear(phi, self.u_proj)
        # w_eff = act_bias + u @ v_proj -> [B, 6]
        w_eff = self.act_bias.unsqueeze(0) + torch.matmul(u, self.v_proj)
        # dq = sum_a w_eff[:, None, a] * act_feats[:, :, a] -> [B, 7]
        dq = (w_eff.unsqueeze(1) * act_feats).sum(dim=-1) * act_mask.float()
        return base_scores + dq

    def export_dict(self, active: bool = True):
        return {
            "active": active,
            "u_proj": self.u_proj.detach().cpu().numpy().astype(np.float32).tolist(),
            "v_proj": self.v_proj.detach().cpu().numpy().astype(np.float32).tolist(),
            "act_bias": self.act_bias.detach().cpu().numpy().astype(np.float32).tolist(),
        }


def zero_rollout_dict():
    return {
        "active": False,
        "u_proj": [[0.0] * 12 for _ in range(4)],
        "v_proj": [[0.0] * 6 for _ in range(4)],
        "act_bias": [0.0] * 6,
    }


def eval_metrics(probs: np.ndarray, targets: np.ndarray):
    diff = probs - targets
    mse = float(np.mean(diff ** 2))
    rms = math.sqrt(mse)
    bias = float(np.mean(diff))
    if len(probs) > 2 and np.std(probs) > 1e-8 and np.std(targets) > 1e-8:
        r = float(np.corrcoef(probs, targets)[0, 1])
    else:
        r = 0.0
    return rms, mse, bias, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="tmp/residual-dataset.jsonl")
    parser.add_argument("--td-alpha", type=float, default=0.50, help="Blend weight on v_td vs v_mcts")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--wd", type=float, default=2e-3)
    parser.add_argument("--l2-out", type=float, default=0.04, help="Regularization on residual logit magnitude")
    parser.add_argument("--out-eval", default="tmp/residual-weights-eval-only.json")
    parser.add_argument("--out-full", default="tmp/residual-weights-full.json")
    args = parser.parse_args()

    rows = []
    with open(args.data, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} labeled states from {args.data}")

    # Deterministic train/val split by (src, game) hash so no game crosses train/val
    train_idx, val_idx = [], []
    for i, r in enumerate(rows):
        h = ((r["src"] * 1000003) ^ (r["game"] * 0x9E3779B9)) & 0xFFFFFFFF
        if (h % 5) == 0:
            val_idx.append(i)
        else:
            train_idx.append(i)

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    print(f"Split: {len(train_idx)} train, {len(val_idx)} val (grouped by game)")

    x0_np = np.array([r["x0"] for r in rows], dtype=np.float32)
    x1_np = np.array([r["x1"] for r in rows], dtype=np.float32)
    base_logit_np = np.array([r["base_logit"] for r in rows], dtype=np.float32)
    base_eval_np = 1.0 / (1.0 + np.exp(-base_logit_np))
    v_mcts_np = np.array([r["v_mcts"] for r in rows], dtype=np.float32)
    v_td_np = np.array([r["v_td"] for r in rows], dtype=np.float32)
    target_np = (1.0 - args.td_alpha) * v_mcts_np + args.td_alpha * v_td_np

    is_1v1_unboosted = np.array(
        [("1v1" in r["tags"] and "unboosted" in r["tags"]) for r in rows],
        dtype=np.float32,
    )

    # ---- Step 1: Signal-to-Noise & Slice Diagnostic on Held-Out Validation Set
    print("\n=== Step 1: Baseline Residual Diagnostic (Validation Set) ===")
    val_base_rms, val_base_mse, val_base_bias, val_base_r = eval_metrics(
        base_eval_np[val_idx], target_np[val_idx]
    )
    print(
        f"ALL VAL (n={len(val_idx):4d}) | RMS: {val_base_rms:.4f}  MSE: {val_base_mse:.5f}  "
        f"Bias: {val_base_bias:+.4f}  Pearson r: {val_base_r:.4f}"
    )

    tag_Order = [
        "unboosted",
        "boosted",
        "has_substitute",
        "has_self_switch",
        "boosted_self_switch",
        "has_spikes",
        "perish_or_trap",
        "alive_diff",
        "1v1",
    ]
    for tag in tag_Order:
        sub = [i for i in val_idx if tag in rows[i]["tags"]]
        if sub:
            sub = np.array(sub, dtype=np.int64)
            rms, mse, bias, r = eval_metrics(base_eval_np[sub], target_np[sub])
            print(f"  slice {tag:20s} (n={len(sub):4d}) | base RMS: {rms:.4f}  r: {r:.4f}")

    # ---- Step 2: Train Antisymmetric EvalResidualNet (32 -> 16 -> 1)
    torch.manual_seed(20261029)
    net = EvalResidualNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)

    x0_tr = torch.from_numpy(x0_np[train_idx])
    x1_tr = torch.from_numpy(x1_np[train_idx])
    bl_tr = torch.from_numpy(base_logit_np[train_idx])
    y_tr = torch.from_numpy(target_np[train_idx])
    u1v1_tr = torch.from_numpy(is_1v1_unboosted[train_idx])

    x0_va = torch.from_numpy(x0_np[val_idx])
    x1_va = torch.from_numpy(x1_np[val_idx])
    bl_va = torch.from_numpy(base_logit_np[val_idx])
    y_va = torch.from_numpy(target_np[val_idx])

    best_val_mse = val_base_mse
    best_state = {k: v.clone() for k, v in net.state_dict().items()}

    for epoch in range(1, args.epochs + 1):
        net.train()
        optimizer.zero_grad()
        delta = net.forward_sym(x0_tr, x1_tr)
        logits = bl_tr + delta
        preds = torch.sigmoid(logits)

        mse_loss = F.mse_loss(preds, y_tr)
        bce_loss = F.binary_cross_entropy_with_logits(logits, y_tr)
        reg_out = args.l2_out * torch.mean(delta ** 2 * (1.0 + 2.0 * u1v1_tr))
        loss = mse_loss + 0.25 * bce_loss + reg_out
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            net.eval()
            with torch.no_grad():
                val_delta = net.forward_sym(x0_va, x1_va)
                val_preds = torch.sigmoid(bl_va + val_delta).numpy()
                rms, mse, bias, r = eval_metrics(val_preds, target_np[val_idx])
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        all_delta = net.forward_sym(torch.from_numpy(x0_np), torch.from_numpy(x1_np)).numpy()
        all_preds = 1.0 / (1.0 + np.exp(-(base_logit_np + all_delta)))

    print("\n=== Step 2: Trained EvalResidualNet Validation Results ===")
    val_rms, val_mse, val_bias, val_r = eval_metrics(all_preds[val_idx], target_np[val_idx])
    print(
        f"ALL VAL (n={len(val_idx):4d}) | RMS: {val_base_rms:.4f} -> {val_rms:.4f} "
        f"(dRMS={val_rms - val_base_rms:+.4f}) | MSE: {val_base_mse:.5f} -> {val_mse:.5f} | "
        f"r: {val_base_r:.4f} -> {val_r:.4f}"
    )
    for tag in tag_Order:
        sub = [i for i in val_idx if tag in rows[i]["tags"]]
        if sub:
            sub = np.array(sub, dtype=np.int64)
            b_rms, _, _, b_r = eval_metrics(base_eval_np[sub], target_np[sub])
            n_rms, _, _, n_r = eval_metrics(all_preds[sub], target_np[sub])
            print(
                f"  slice {tag:20s} (n={len(sub):4d}) | RMS: {b_rms:.4f} -> {n_rms:.4f} "
                f"({n_rms - b_rms:+.4f}) | r: {b_r:.4f} -> {n_r:.4f}"
            )

    # Verify antisymmetry and zero-baseline numerically
    with torch.no_grad():
        d_fwd = net.forward_sym(x0_va, x1_va)
        d_rev = net.forward_sym(x1_va, x0_va)
        max_asym = float(torch.max(torch.abs(d_fwd + d_rev)).item())
        zero_out = float(net.forward_single(torch.zeros(1, 32)).item())
        print(f"Max antisymmetry error = {max_asym:.2e} | Zero-input output f(0) = {zero_out:.2e}")
        w1_e, b1_e, w2_e = net.effective_params()
        eff_grad = (w2_e @ w1_e).cpu().numpy()
        print("  Synergy slopes (0..16, all >= 0):", [round(float(x), 3) for x in eff_grad[:16]])
        print("  Vuln    slopes (16..32, all <= 0):", [round(float(x), 3) for x in eff_grad[16:]])

    # ---- Step 3: Train RolloutResidualNet (Rank-4 Bilinear 12x6)
    act_train, act_val = [], []
    for i in train_idx:
        for a_rec in rows[i]["actions"]:
            act_train.append(a_rec)
    for i in val_idx:
        for a_rec in rows[i]["actions"]:
            act_val.append(a_rec)

    def pack_act_dataset(records):
        n = len(records)
        phi = np.zeros((n, 12), dtype=np.float32)
        act_feats = np.zeros((n, 7, 6), dtype=np.float32)
        act_mask = np.zeros((n, 7), dtype=np.bool_)
        valid_mask = np.zeros((n, 7), dtype=np.bool_)
        base_scores = np.full((n, 7), -1e9, dtype=np.float32)
        target_probs = np.zeros((n, 7), dtype=np.float32)
        best_idx = np.zeros(n, dtype=np.int64)

        for idx, rec in enumerate(records):
            nl = rec["n_legal"]
            phi[idx] = rec["phi"]
            vis = np.array(rec["visits"], dtype=np.float32)
            q = np.array(rec["q_values"], dtype=np.float32)
            # Combine visit fraction and Q-advantage into a soft target distribution
            v_frac = vis / max(float(np.sum(vis)), 1.0)
            for c in range(nl):
                af = rec["act_feats"][c]
                act_feats[idx, c] = af
                if any(abs(v) > 1e-6 for v in af):
                    act_mask[idx, c] = True
                valid_mask[idx, c] = True
                base_scores[idx, c] = rec["base_scores"][c]
                target_probs[idx, c] = v_frac[c]
            # Best action by MCTS visits (tie-broken by Q)
            best_idx[idx] = int(np.argmax(vis + 0.01 * q))
        return (
            torch.from_numpy(phi),
            torch.from_numpy(act_feats),
            torch.from_numpy(act_mask),
            torch.from_numpy(valid_mask),
            torch.from_numpy(base_scores),
            torch.from_numpy(target_probs),
            torch.from_numpy(best_idx),
        )

    rnet = RolloutResidualNet()
    if len(act_train) > 50 and len(act_val) > 10:
        phi_tr, af_tr, am_tr, vm_tr, bs_tr, tp_tr, bi_tr = pack_act_dataset(act_train)
        phi_va, af_va, am_va, vm_va, bs_va, tp_va, bi_va = pack_act_dataset(act_val)

        with torch.no_grad():
            base_top1_va = float((torch.argmax(bs_va, dim=-1) == bi_va).float().mean().item())

        r_opt = torch.optim.AdamW(rnet.parameters(), lr=5e-3, weight_decay=5e-3)
        best_r_acc = base_top1_va
        best_r_state = None
        temp = 0.20

        for epoch in range(1, 201):
            rnet.train()
            r_opt.zero_grad()
            scores = rnet.forward_scores(phi_tr, af_tr, am_tr, bs_tr)
            log_p = F.log_softmax(scores / temp, dim=-1)
            kl_loss = -torch.sum(tp_tr * log_p, dim=-1).mean()
            # Regularize residual adjustment magnitude so base hit_fraction stays anchor unless needed
            res_mag = ((scores - bs_tr) * vm_tr.float()) ** 2
            loss = kl_loss + 0.15 * res_mag.mean()
            loss.backward()
            r_opt.step()

            if epoch % 10 == 0 or epoch == 200:
                rnet.eval()
                with torch.no_grad():
                    v_scores = rnet.forward_scores(phi_va, af_va, am_va, bs_va)
                    acc = float((torch.argmax(v_scores, dim=-1) == bi_va).float().mean().item())
                    if acc > best_r_acc:
                        best_r_acc = acc
                        best_r_state = {k: v.clone() for k, v in rnet.state_dict().items()}

        if best_r_state is not None:
            rnet.load_state_dict(best_r_state)
        rnet.eval()
        with torch.no_grad():
            v_scores = rnet.forward_scores(phi_va, af_va, am_va, bs_va)
            final_top1_va = float((torch.argmax(v_scores, dim=-1) == bi_va).float().mean().item())
        print(
            f"\n=== Step 3: RolloutResidualNet Validation Top-1 MCTS Agreement ===\n"
            f"  records: {len(act_train)} train, {len(act_val)} val | "
            f"Top-1 agreement: {base_top1_va:.4f} -> {final_top1_va:.4f} ({final_top1_va - base_top1_va:+.4f})"
        )
    else:
        print("\n=== Step 3: RolloutResidualNet skipped (insufficient action records) ===")

    # Save Gate 1 (eval-only) and Gate 2 (eval + rollout) JSON bundles
    eval_dict = net.export_dict()
    bundle_eval_only = {
        "eval": eval_dict,
        "rollout": zero_rollout_dict(),
    }
    bundle_full = {
        "eval": eval_dict,
        "rollout": rnet.export_dict(),
    }

    Path(args.out_eval).write_text(json.dumps(bundle_eval_only, indent=2))
    Path(args.out_full).write_text(json.dumps(bundle_full, indent=2))
    print(f"\nExported Gate 1 bundle to {args.out_eval}")
    print(f"Exported Gate 2 bundle to {args.out_full}")


if __name__ == "__main__":
    main()
