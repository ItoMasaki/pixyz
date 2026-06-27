import argparse
import os
import sys
import time

import torch
from torch.nn import functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pixyz.distributions import Bernoulli, Deterministic, Normal
from pixyz.losses import IterativeLoss
from pixyz.utils import call_sample_batch, compile_if_available


class Decoder(Bernoulli):
    def __init__(self, z_dim, h_dim, x_dim):
        super().__init__(var=["x"], cond_var=["z", "h_prev"], name="p")
        self.fc = torch.nn.Linear(z_dim + h_dim, x_dim)

    def forward(self, z, h_prev):
        return {"logits": self.fc(torch.cat((z, h_prev), dim=-1))}


class Encoder(Normal):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__(var=["z"], cond_var=["x", "h_prev"], name="q")
        self.fc_loc = torch.nn.Linear(x_dim + h_dim, z_dim)
        self.fc_scale = torch.nn.Linear(x_dim + h_dim, z_dim)

    def forward(self, x, h_prev):
        xh = torch.cat((x, h_prev), dim=-1)
        return {"loc": self.fc_loc(xh), "scale": F.softplus(self.fc_scale(xh)) + 1e-5}


class Recurrence(Deterministic):
    def __init__(self, x_dim, z_dim, h_dim):
        super().__init__(var=["h"], cond_var=["x", "z", "h_prev"], name="f")
        self.rnncell = torch.nn.GRUCell(x_dim + z_dim, h_dim)

    def forward(self, x, z, h_prev):
        return {"h": call_sample_batch(self.rnncell, torch.cat((z, x), dim=-1), h_prev)}


def build_loss(x_dim, z_dim, h_dim, use_compile=False):
    p = Decoder(z_dim=z_dim, h_dim=h_dim, x_dim=x_dim)
    q = Encoder(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
    f = Recurrence(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim)

    step_loss_cls = p.log_prob().expectation(q * f).mean()
    loss_cls = IterativeLoss(step_loss=step_loss_cls, series_var=["x"], update_value={"h": "h_prev"})

    if use_compile:
        p = compile_if_available(p)
        q = compile_if_available(q)
        f = compile_if_available(f)
        loss_cls = compile_if_available(loss_cls)

    return loss_cls, (p, q, f)


def measure(loss_cls, modules, t_max, batch_size, x_dim, h_dim, loops, device):
    x_sample = torch.randn(t_max, batch_size, x_dim, device=device)
    h_init = torch.zeros(batch_size, h_dim, device=device)

    for _ in range(3):
        loss = loss_cls.eval({"x": x_sample, "h_prev": h_init})
        loss.backward()
        for module in modules:
            module.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(loops):
        loss = loss_cls.eval({"x": x_sample, "h_prev": h_init})
        loss.backward()
        for module in modules:
            module.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    elapsed = (time.perf_counter() - start) / loops
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark Pixyz sequential training performance.")
    parser.add_argument("--t-max", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--x-dim", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--h-dim", type=int, default=16)
    parser.add_argument("--loops", type=int, default=30)
    parser.add_argument("--compile", action="store_true", dest="use_compile")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    loss_cls, modules = build_loss(args.x_dim, args.z_dim, args.h_dim, use_compile=args.use_compile)
    for module in modules:
        module.to(device)

    elapsed = measure(loss_cls, modules, args.t_max, args.batch_size, args.x_dim, args.h_dim, args.loops, device)
    print({"train_step_sec": elapsed, "compile": args.use_compile, "device": str(device)})


if __name__ == "__main__":
    main()
