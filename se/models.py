"""Model registry.

Every model maps a noisy waveform batch [B, T] to an enhanced one with
`enhance(noisy) -> [B, T]` (same length). Spectral-mask models also expose
`spectrum(wave)` and `forward(noisy) -> estimated complex spectrum`, which the
spectral losses use. Register a new model in MODELS; its constructor keyword
arguments are the config's "model_args". Models taken from open-source projects live
in se/arch/, one file per model, with their source and license at the top.
"""
from __future__ import annotations

import torch
from torch import nn

from se.arch.mpsenet import MPSENet


class MaskCNN(nn.Module):
    """The course notebook's example: 3 Conv1d layers over log|STFT| -> sigmoid mask.

    Frequency bins are the Conv1d channels, so each layer sees all bins of a few
    neighbouring frames. The mask multiplies the complex noisy spectrum (noisy
    phase kept). Parameter names match the notebook, so its weights load as is.
    """

    def __init__(self, n_fft: int = 512, hop: int = 128,
                 hidden_channels=(64, 64), kernel_sizes=(5, 3, 1)):
        super().__init__()
        if not (n_fft > 0 and 0 < hop <= n_fft // 2):
            raise ValueError("need n_fft > 0 and 0 < hop <= n_fft // 2")
        if len(hidden_channels) != 2 or len(kernel_sizes) != 3 or \
                any(k <= 0 or k % 2 == 0 for k in kernel_sizes):
            raise ValueError("need 2 hidden channel counts and 3 positive odd kernel sizes")
        self.n_fft, self.hop = n_fft, hop
        bins = n_fft // 2 + 1
        c1, c2 = hidden_channels
        k1, k2, k3 = kernel_sizes
        self.mask_net = nn.Sequential(
            nn.Conv1d(bins, c1, kernel_size=k1, padding=k1 // 2),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2),
            nn.ReLU(),
            nn.Conv1d(c2, bins, kernel_size=k3, padding=k3 // 2),
            nn.Sigmoid(),
        )
        self.register_buffer("window", torch.hann_window(n_fft))

    def spectrum(self, wave: torch.Tensor) -> torch.Tensor:
        return torch.stft(wave, n_fft=self.n_fft, hop_length=self.hop, window=self.window,
                          return_complex=True)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        spec = self.spectrum(noisy)                       # [B, bins, frames]
        mask = self.mask_net(torch.log1p(spec.abs()))
        return mask * spec

    def enhance(self, noisy: torch.Tensor) -> torch.Tensor:
        return torch.istft(self(noisy), n_fft=self.n_fft, hop_length=self.hop,
                           window=self.window, length=noisy.shape[-1])


class Ensemble(nn.Module):
    """Several models in one weight file; the output is the weighted mean of their outputs.

    members: [{"model": name, "model_args": {...}, "weight": w}, ...]. The weights are
    fixed numbers stored in the checkpoint's model_args (the same for every input file) and
    are used as given divided by their sum. Each member runs on the input on its own,
    exactly as it would alone.

    combine "wave" (default): weighted mean of the output waveforms.
    combine "mag": weighted mean of the outputs' STFT magnitudes (400/100, Hann) with the phase
    of member `phase_member`, one inverse STFT. On four validation sets this is 0.01 to 0.02
    PESQ above the waveform mean with the same weights (results/mix_study/report.md): where
    the members disagree in phase, a waveform mean cancels part of the speech, a magnitude
    mean does not. `pcs` > 0 then applies Perceptual Contrast Stretching of that strength to
    the averaged magnitude before the inverse STFT (se/pcs.py). An input too short for the
    STFT falls back to the waveform mean.
    """

    N_FFT, HOP = 400, 100

    def __init__(self, members: list[dict], combine: str = "wave", phase_member: int = 0,
                 pcs: float = 0.0):
        super().__init__()
        if combine not in ("wave", "mag") or not 0 <= int(phase_member) < len(members) or pcs < 0:
            raise ValueError("combine is 'wave' or 'mag', phase_member a member index, pcs >= 0")
        if combine == "wave" and pcs:
            raise ValueError("pcs needs combine='mag'")
        self.combine, self.phase_member, self.pcs = combine, int(phase_member), float(pcs)
        if len(members) < 2 or any(m["model"] == "ensemble" or float(m["weight"]) <= 0
                                   for m in members):
            raise ValueError("an ensemble needs two or more members that are not ensembles, "
                             "each with a weight > 0")
        self.members = nn.ModuleList(build_model(m["model"], m.get("model_args"))
                                     for m in members)
        total = sum(float(m["weight"]) for m in members)
        self.weights = [float(m["weight"]) / total for m in members]

    def enhance(self, noisy: torch.Tensor) -> torch.Tensor:
        outs = [member.enhance(noisy) for member in self.members]
        if self.combine == "wave" or noisy.shape[-1] <= self.N_FFT:
            return sum(w * y for w, y in zip(self.weights, outs))
        from se.pcs import pcs_stretch
        window = torch.hann_window(self.N_FFT, dtype=torch.float64, device=noisy.device)
        spec = [torch.stft(y.double(), self.N_FFT, self.HOP, window=window, return_complex=True)
                for y in outs]
        mag = sum(w * s.abs() for w, s in zip(self.weights, spec))
        phase = torch.angle(spec[self.phase_member])
        y = torch.istft(torch.polar(mag, phase), self.N_FFT, self.HOP, window=window,
                        length=noisy.shape[-1])
        if self.pcs:
            y = pcs_stretch(mag, phase, y, self.pcs, window, self.HOP)
        return y.to(noisy.dtype)


def _xlstm_senet(**model_args) -> nn.Module:
    from se.arch.xlstm_senet import XLSTMSENet      # imported on use: needs the einops package
    return XLSTMSENet(**model_args)


def _tflocoformer(**model_args) -> nn.Module:
    from se.arch.tflocoformer import LocoDenoiser   # imported on use: needs rotary-embedding-torch
    return LocoDenoiser(**model_args)


def _mossformer_gan_se(**model_args) -> nn.Module:
    from se.arch.mossformer_gan_se import MossFormerGANSE16K   # imported on use: needs einops
    return MossFormerGANSE16K(**model_args)


def _semamba(**model_args) -> nn.Module:
    from se.arch.semamba import SEMambaDenoiser     # imported on use: needs einops
    return SEMambaDenoiser(**model_args)


MODELS: dict[str, type[nn.Module]] = {
    "mask_cnn": MaskCNN,
    "mpsenet": MPSENet,          # se/arch/mpsenet.py (MP-SENet, MIT License)
    "xlstm_senet": _xlstm_senet,  # se/arch/xlstm_senet/ (xLSTM-SENet, MIT License)
    "tflocoformer": _tflocoformer,  # se/arch/tflocoformer/ (TF-Locoformer, Apache-2.0)
    "mossformer_gan_se": _mossformer_gan_se,  # se/arch/mossformer_gan_se/ (ClearerVoice, Apache-2.0)
    "semamba": _semamba,        # se/arch/semamba/ (SEMamba, MIT License; Mamba in plain PyTorch)
    "ensemble": Ensemble,       # weighted mean of member models (tools/make_ensemble.py)
}


def build_model(name: str, model_args: dict | None = None) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name](**(model_args or {}))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
