#!/usr/bin/env python3
"""S1 fitted value iteration: train the value network on fvi_gen records.

The forward pass mirrors `ValueNet` in crates/bot/src/value.rs exactly;
`fvi_gen --parity MODEL.json` checks the exported model against the logits
stored here. Targets are λ-returns over each game's backup values y_t and its
outcome: G_T = outcome, G_t = (1-λ)·y_t + λ·G_{t+1}.

  tmp/venv/bin/python tools/fvi-train.py --codec tmp/fvi-codec.json \
      --data tmp/fvi/iter0 [--data DIR ...] --out tmp/fvi/model-1.json \
      [--init PREV.json] [--lam 0.5] [--epochs 8]
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EXTRA = 4


class Net(nn.Module):
    def __init__(self, c, ds=16, di=8, dm=16, hm=32, h1=96, h2=64, hh1=192, hh2=96):
        super().__init__()
        d = c["dims"]
        self.c = c
        self.o = c["offsets"]
        self.M, self.K, self.I = d["mons"], d["moves"], d["ids"]
        self.f_mon, self.f_mdyn, self.f_mstat = d["mon"], d["mdyn"], d["mstat"]
        self.f_vol, self.f_side, self.f_pair, self.f_global = d["vol"], d["side"], d["pair"], d["global"]
        self.species = nn.Embedding(c["n_species"] + 1, ds)
        self.items = nn.Embedding(c["n_items"] + 1, di)
        self.moves = nn.Embedding(c["n_moves"] + 1, dm)
        self.register_buffer("mstat", torch.tensor(c["move_static"], dtype=torch.float32))
        self.move_layer = nn.Linear(dm + self.f_mstat + self.f_mdyn, hm)
        self.mon1 = nn.Linear(ds + di + hm + self.f_mon, h1)
        self.mon2 = nn.Linear(h1, h2)
        side = 2 * h2 + self.f_side + self.f_vol
        self.head1 = nn.Linear(2 * side + self.f_pair + self.f_global, hh1)
        self.head2 = nn.Linear(hh1, hh2)
        self.head3 = nn.Linear(hh2, 1)

    def block(self, r, name, shape):
        off = self.o[name]
        n = int(np.prod(shape))
        return r[:, off:off + n].reshape(r.shape[0], *shape)

    def side_vectors(self, r):
        B, M, K = r.shape[0], self.M, self.K
        ids = self.block(r, "ids", (2, M, self.I)).long()
        mon = self.block(r, "mon", (2, M, self.f_mon))
        mdyn = self.block(r, "mdyn", (2, M, K, self.f_mdyn))
        vol = self.block(r, "vol", (2, self.f_vol))
        side = self.block(r, "side", (2, self.f_side))
        mids = ids[..., 2:]
        mx = torch.cat([self.moves(mids), self.mstat[mids], mdyn], dim=-1)
        mv = F.relu(self.move_layer(mx)) * (mids > 0).unsqueeze(-1).float()
        mv = mv.sum(dim=3)
        x = torch.cat([self.species(ids[..., 0]), self.items(ids[..., 1]), mv, mon], dim=-1)
        h = F.relu(self.mon2(F.relu(self.mon1(x))))
        h = h * (mon[..., 0] != 0).unsqueeze(-1).float()
        active = h[:, :, 0]
        bench = h[:, :, 1:].sum(dim=2)
        return torch.cat([active, bench, side, vol], dim=-1)

    def head(self, sv, r, me):
        pair = self.block(r, "pair", (2, self.f_pair))[:, me]
        glob = r[:, self.o["global"]:self.o["global"] + self.f_global]
        x = torch.cat([sv[:, me], sv[:, 1 - me], pair, glob], dim=-1)
        return self.head3(F.relu(self.head2(F.relu(self.head1(x))))).squeeze(-1)

    def forward(self, r):
        sv = self.side_vectors(r)
        return self.head(sv, r, 0) - self.head(sv, r, 1)


def dense(lin):
    w = lin.weight.detach().cpu().float().numpy()
    return {"n_in": w.shape[1], "n_out": w.shape[0], "w": w.reshape(-1).tolist(),
            "b": lin.bias.detach().cpu().float().numpy().tolist()}


def emb(e):
    w = e.weight.detach().cpu().float().numpy()
    return {"rows": w.shape[0], "dim": w.shape[1], "w": w.reshape(-1).tolist()}


def export(net, path, parity_rows, meta):
    net.eval()
    with torch.no_grad():
        logits = net(parity_rows).cpu().numpy().tolist()
    doc = {
        "schema": net.c["schema"],
        "species": emb(net.species), "items": emb(net.items), "moves": emb(net.moves),
        "move_layer": dense(net.move_layer), "mon1": dense(net.mon1), "mon2": dense(net.mon2),
        "head1": dense(net.head1), "head2": dense(net.head2), "head3": dense(net.head3),
        "parity": [[row, lg] for row, lg in zip(parity_rows.cpu().numpy().tolist(), logits)],
        "meta": meta,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, path)


def load_init(net, path):
    doc = json.load(open(path))
    sd = {}
    for name in ["species", "items", "moves"]:
        e = doc[name]
        sd[f"{name}.weight"] = torch.tensor(e["w"]).reshape(e["rows"], e["dim"])
    for name in ["move_layer", "mon1", "mon2", "head1", "head2", "head3"]:
        d = doc[name]
        sd[f"{name}.weight"] = torch.tensor(d["w"]).reshape(d["n_out"], d["n_in"])
        sd[f"{name}.bias"] = torch.tensor(d["b"])
    missing = net.load_state_dict(sd, strict=False)
    assert set(missing.missing_keys) <= {"mstat"}, missing


def lambda_targets(rows, record, lam):
    y = rows[:, record]
    outcome = rows[:, record + 1]
    game = rows[:, record + 2]
    step = rows[:, record + 3]
    tgt = np.empty_like(y)
    n = len(rows)
    i = n - 1
    nxt = None
    while i >= 0:
        last = i == n - 1 or game[i + 1] != game[i] or step[i + 1] != step[i] + 1
        g_next = outcome[i] if last else nxt
        tgt[i] = (1 - lam) * y[i] + lam * g_next
        nxt = tgt[i]
        i -= 1
    return tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    c = json.load(open(args.codec))
    record = c["record"]

    feats, tgts, ys, outs, keys = [], [], [], [], []
    for di, d in enumerate(args.data):
        raw = np.fromfile(os.path.join(d, "records.f32"), dtype="<f4")
        rows = raw.reshape(-1, record + EXTRA)
        assert not np.isnan(rows[:, record + 1]).any(), f"{d}: unfinished game rows"
        tgts.append(lambda_targets(rows, record, args.lam))
        feats.append(rows[:, :record])
        ys.append(rows[:, record])
        outs.append(rows[:, record + 1])
        keys.append(di * 10_000_000 + rows[:, record + 2].astype(np.int64))
    X = torch.from_numpy(np.concatenate(feats))
    T = torch.from_numpy(np.concatenate(tgts))
    Y = np.concatenate(ys)
    O = np.concatenate(outs)
    G = np.concatenate(keys)
    rng = np.random.default_rng(args.seed)
    games = np.unique(G)
    val_games = set(rng.choice(games, size=max(1, int(len(games) * args.val)), replace=False).tolist())
    is_val = np.array([g in val_games for g in G])
    tr = np.flatnonzero(~is_val)
    va = np.flatnonzero(is_val)
    print(f"records {len(X)} (train {len(tr)}, val {len(va)}), games {len(games)}, lam {args.lam}", flush=True)

    net = Net(c)
    if args.init:
        load_init(net, args.init)
    dev = torch.device(args.device)
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(tr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    Xv, Tv = X[va].to(dev), T[va].to(dev)

    def evaluate():
        net.eval()
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(net(Xv[i:i + 8192])) for i in range(0, len(Xv), 8192)])
        net.train()
        p = p.cpu().numpy()
        return {
            "bce": float(F.binary_cross_entropy(torch.from_numpy(p).clamp(1e-6, 1 - 1e-6), Tv.cpu()).item()),
            "mse_target": float(np.mean((p - Tv.cpu().numpy()) ** 2)),
            "mse_y": float(np.mean((p - Y[va]) ** 2)),
            "brier_outcome": float(np.mean((p - O[va]) ** 2)),
            "brier_y_outcome": float(np.mean((Y[va] - O[va]) ** 2)),
        }

    t0 = time.time()
    best = None
    for ep in range(args.epochs):
        perm = torch.from_numpy(rng.permutation(tr))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb, tb = X[idx].to(dev), T[idx].to(dev)
            loss = F.binary_cross_entropy_with_logits(net(xb), tb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        m = evaluate()
        print(f"epoch {ep + 1}/{args.epochs} train_bce {tot / len(tr):.5f} "
              + " ".join(f"{k} {v:.5f}" for k, v in m.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
        best = m
    net.cpu()
    meta = {"data": args.data, "lam": args.lam, "epochs": args.epochs, "init": args.init,
            "records": int(len(X)), "val": best}
    export(net, args.out, X[va[:64]], meta)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
