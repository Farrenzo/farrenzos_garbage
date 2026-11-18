#!/usr/bin/env python3
"""Stand-in for the `dac` and `audiotools` packages, so fish-speech's codec
can be imported without installing descript-audio-codec.

Why this exists
---------------
`fish_speech/models/dac/modded_dac.py` and `rvq.py` import five symbols from
two descript packages:

    from audiotools import AudioSignal          # imported, never used
    from audiotools.ml import BaseModel         # base class only
    from dac.model.base import CodecMixin       # only get_delay() is called
    from dac.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d
    from dac.nn.quantize import ResidualVectorQuantize

Installing descript-audio-codec to get those five things pulls in
descript-audiotools, which hard-pins `protobuf<3.20` and drags in tensorboard,
matplotlib, librosa (and therefore numba, which pins numpy), ipython, scipy,
julius, torch-stoi and a dozen more. In a shared environment that is a
meaningful chance of a numpy or protobuf downgrade.

The five symbols themselves are small and depend on nothing but torch, numpy
and einops. This module reproduces them and registers fake `dac` and
`audiotools` packages in sys.modules, so the fish-speech imports resolve
without anything being installed.

The parameter shapes and names here match descript-audio-codec 1.0.0 exactly,
including the use of the legacy `torch.nn.utils.weight_norm` (which produces
`weight_g` / `weight_v` keys) rather than the newer parametrization API
(`parametrizations.weight.original0` / `original1`). Getting this wrong would
not raise, because fish-speech loads the codec with strict=False — it would
just produce noise. Call `install()` before importing anything from
fish_speech, and check the missing-key report when you load the checkpoint.

Usage:
    import descript_shim
    descript_shim.install()      # no-op if the real packages are present
"""

from __future__ import annotations

import math
import sys
import types
import warnings
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

with warnings.catch_warnings():
    # torch.nn.utils.weight_norm is deprecated in favour of the parametrization
    # API, but the checkpoint was written with the old one and the state dict
    # key names differ between them. Keep the old one.
    warnings.simplefilter("ignore")
    from torch.nn.utils import weight_norm


# --------------------------------------------------------------------------
# dac.nn.layers
# --------------------------------------------------------------------------


def WNConv1d(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def _snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


try:
    # upstream scripts this for speed; fall back to eager if the JIT is
    # unhappy on this backend
    snake = torch.jit.script(_snake)
except Exception:  # pragma: no cover
    snake = _snake


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


# --------------------------------------------------------------------------
# dac.nn.quantize
# --------------------------------------------------------------------------


class VectorQuantize(nn.Module):
    """Factorized, l2-normalized VQ (ViT-VQGAN style)."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """Residual VQ stack (SoundStream, arXiv:2107.03312)."""

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[0]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


# --------------------------------------------------------------------------
# dac.model.base / audiotools.ml
# --------------------------------------------------------------------------


class CodecMixin:
    """Only get_delay/get_output_length are reachable from the fish-speech
    inference path; DAC.__init__ calls get_delay() once."""

    @property
    def padding(self):
        if not hasattr(self, "_padding"):
            self._padding = True
        return self._padding

    @padding.setter
    def padding(self, value):
        self._padding = value

    def get_delay(self):
        l_out = self.get_output_length(0)
        L = l_out

        layers = []
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(layer)

        for layer in reversed(layers):
            d = layer.dilation[0]
            k = layer.kernel_size[0]
            s = layer.stride[0]

            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1

            L = math.ceil(L)

        return (L - l_out) // 2

    def get_output_length(self, input_length):
        L = input_length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]

                if isinstance(layer, nn.Conv1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d):
                    L = (L - 1) * s + d * (k - 1) + 1

                L = math.floor(L)
        return L


class BaseModel(nn.Module):
    """audiotools.ml.BaseModel minus the checkpoint helpers we never call."""

    @property
    def device(self):
        return next(self.parameters()).device


class AudioSignal:  # imported by modded_dac.py, never instantiated
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "descript_shim provides a placeholder AudioSignal only; install "
            "descript-audiotools if you need the real one."
        )


# --------------------------------------------------------------------------
# module registration
# --------------------------------------------------------------------------


def _make_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__shimmed__ = True
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def real_packages_available() -> bool:
    import importlib.util

    return all(
        importlib.util.find_spec(name) is not None
        for name in ("dac", "audiotools")
    )


def install(force: bool = False) -> bool:
    """Register the fake packages. Returns True if the shim was installed."""
    if not force and real_packages_available():
        return False
    if any(getattr(sys.modules.get(n), "__shimmed__", False) for n in ("dac", "audiotools")):
        return True

    audiotools = _make_module("audiotools", AudioSignal=AudioSignal)
    audiotools.__path__ = []
    audiotools_ml = _make_module("audiotools.ml", BaseModel=BaseModel)
    audiotools.ml = audiotools_ml

    dac = _make_module("dac")
    dac.__path__ = []
    dac_model = _make_module("dac.model")
    dac_model.__path__ = []
    dac_model_base = _make_module("dac.model.base", CodecMixin=CodecMixin)
    dac_model.base = dac_model_base
    dac.model = dac_model

    dac_nn = _make_module("dac.nn")
    dac_nn.__path__ = []
    dac_nn_layers = _make_module(
        "dac.nn.layers",
        Snake1d=Snake1d,
        WNConv1d=WNConv1d,
        WNConvTranspose1d=WNConvTranspose1d,
        snake=snake,
    )
    dac_nn_quantize = _make_module(
        "dac.nn.quantize",
        VectorQuantize=VectorQuantize,
        ResidualVectorQuantize=ResidualVectorQuantize,
    )
    dac_nn.layers = dac_nn_layers
    dac_nn.quantize = dac_nn_quantize
    dac.nn = dac_nn

    sys.modules.update(
        {
            "audiotools": audiotools,
            "audiotools.ml": audiotools_ml,
            "dac": dac,
            "dac.model": dac_model,
            "dac.model.base": dac_model_base,
            "dac.nn": dac_nn,
            "dac.nn.layers": dac_nn_layers,
            "dac.nn.quantize": dac_nn_quantize,
        }
    )
    return True
