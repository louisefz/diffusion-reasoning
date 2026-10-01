"""Minimal ELF-style model. No claim of architectural equivalence to ELF."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from task import encode, vocabulary


@dataclass
class Config:
    symbols: int = 8
    functions: int = 4
    max_depth: int = 16
    width: int = 128
    layers: int = 4
    heads: int = 4
    self_condition: bool = False
    decode_history: bool = True  # Legacy checkpoint behavior; new training disables it.


class FlowModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        nv = len(vocabulary(c.symbols, c.functions))
        # Fixed one-hot symbols remove a pretrained contextual encoder confound.
        self.register_buffer("condition_codes", torch.eye(nv))
        self.register_buffer("answer_codes", math.sqrt(c.symbols) * torch.eye(c.symbols))
        self.condition_in = nn.Linear(nv, c.width)
        self.answer_in = nn.Linear(c.symbols, c.width)
        self.history_in = nn.Linear(c.symbols, c.width, bias=False)
        max_length = 5 + c.functions * (c.symbols + 2) + c.max_depth + 1
        self.position = nn.Embedding(max_length, c.width)
        self.mode = nn.Embedding(2, c.width)
        self.time = nn.Sequential(nn.Linear(3, c.width), nn.SiLU(), nn.Linear(c.width, c.width))
        layer = nn.TransformerEncoderLayer(c.width, c.heads, c.width * 4,
                    dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c.width)
        self.clean = nn.Linear(c.width, c.symbols)
        self.unembed = nn.Linear(c.symbols, c.symbols)
        # TransformerEncoder clones initial weights; initialize layers independently.
        for block in self.transformer.layers:
            for p in block.parameters():
                if p.ndim > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, condition, z, t, mode=0, previous=None):
        b, length = condition.shape
        context = self.condition_in(self.condition_codes[condition])
        answer = self.answer_in(z)
        if self.config.self_condition and previous is not None:
            answer = answer + self.history_in(previous)
        h = torch.cat([context, answer[:, None, :]], dim=1)
        pos = torch.arange(length + 1, device=z.device)
        features = torch.stack([t, torch.sin(math.pi*t), torch.cos(math.pi*t)], dim=-1)
        modes = torch.full((b,), mode, dtype=torch.long, device=z.device)
        h = h + self.position(pos)[None] + self.time(features)[:, None] + self.mode(modes)[:, None]
        padding = torch.cat([condition.eq(0), torch.zeros(b, 1, dtype=torch.bool, device=z.device)], dim=1)
        h = self.transformer(h, src_key_padding_mask=padding)
        xhat = self.clean(self.norm(h[:, -1]))
        return self.unembed(xhat) if mode == 1 else xhat


def batch(rows, config, device):
    inputs = [encode(r, config.symbols, config.functions) for r in rows]
    # Fixed length across batches ensures replay independent of batch padding.
    length = 5 + config.functions * (config.symbols + 2) + config.max_depth
    if max(map(len, inputs)) > length:
        raise ValueError("Program exceeds configured positional capacity")
    condition = torch.zeros(len(rows), length, dtype=torch.long, device=device)
    for i, ids in enumerate(inputs):
        condition[i, :len(ids)] = torch.tensor(ids, device=device)
    labels = torch.tensor([r["answer"] for r in rows], device=device)
    return condition, labels


@torch.no_grad()
def sample(model, condition, steps=32, z=None, previous=None, start_step=0, record=False):
    if steps < 1 or not 0 <= start_step <= steps:
        raise ValueError("Invalid sampling grid")
    if start_step and z is None:
        raise ValueError("Resume requires saved state")
    if start_step and model.config.self_condition and previous is None:
        raise ValueError("Self-conditioned replay requires previous prediction")
    device = condition.device
    b = len(condition)
    if z is None:
        z = torch.randn(b, model.config.symbols, device=device)
    else:
        z = z.clone()
    if previous is None:
        previous = torch.zeros_like(z)
    trace = []
    for i in range(start_step, steps + 1):
        if record:
            trace.append(dict(step=i, t=i/steps, z=z.cpu().clone(), previous=previous.cpu().clone()))
        if i == steps:
            break
        t = torch.full((b,), i/steps, device=device)
        xhat = model(condition, z, t, previous=previous)
        velocity = (xhat - z) / (1 - i/steps)
        if record:
            trace[-1].update(xhat=xhat.cpu().clone(), velocity=velocity.cpu().clone())
        z = z + velocity / steps
        previous = xhat
    logits = model(condition, z, torch.ones(b, device=device), mode=1,
                   previous=previous if model.config.decode_history else None)
    return logits, trace


def direct_logits(model, condition):
    # A constant answer query: neither label embeddings nor random noise enter.
    z = torch.zeros(len(condition), model.config.symbols, device=condition.device)
    return model(condition, z, torch.ones(len(condition), device=condition.device), mode=1)


def losses(model, condition, labels, objective="flow"):
    if objective == "direct":
        ce = nn.functional.cross_entropy(direct_logits(model, condition), labels)
        return ce, ce.detach().new_zeros(()), ce.detach()
    if objective != "flow":
        raise ValueError(objective)
    x = model.answer_codes[labels]
    # Truncate training time away from 1 to bound x-prediction weighting.
    t = torch.sigmoid(torch.randn(len(x), device=x.device)).clamp(0.02, 0.98)
    noise = torch.randn_like(x)
    z = t[:, None] * x + (1 - t[:, None]) * noise
    previous = torch.zeros_like(x)
    if model.config.self_condition:
        with torch.no_grad():
            prediction = model(condition, z, t, previous=previous)
        mask = torch.rand(len(x), 1, device=x.device) < 0.5
        previous = prediction * mask
    xhat = model(condition, z, t, previous=previous)
    fm = (((xhat-x)/(1-t[:, None]))**2).mean()
    # Decode mode is conditioned on time 1 but trained on corrupted clean codes.
    p = torch.sigmoid(0.8 + 0.8*torch.randn(len(x), 1, device=x.device))
    corrupted = p*x + (1-p)*torch.randn_like(x)
    logits = model(condition, corrupted, torch.ones_like(t), mode=1)
    ce = nn.functional.cross_entropy(logits, labels)
    return 0.8*fm + 0.2*ce, fm.detach(), ce.detach()
