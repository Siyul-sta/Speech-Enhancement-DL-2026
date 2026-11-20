"""Restore 16-kHz mono WAVs using the trained 3-layer MaskCNN baseline.

Usage:
    python src/inference.py --input input --output output --checkpoint checkpoints/best_mask_cnn.pt
"""

import argparse
from pathlib import Path

import soundfile as sf
import torch
from torch import nn


SAMPLE_RATE = 16000
N_FFT = 512
HOP = 128


class MaskCNN(nn.Module):
    """Architecture matching the distributed baseline notebook."""

    def __init__(self):
        super().__init__()
        self.n_fft, self.hop = N_FFT, HOP
        bins = self.n_fft // 2 + 1
        self.mask_net = nn.Sequential(
            nn.Conv1d(bins, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, bins, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, noisy):
        spectrum = torch.stft(
            noisy, n_fft=self.n_fft, hop_length=self.hop,
            window=self.window, return_complex=True,
        )
        mask = self.mask_net(torch.log1p(spectrum.abs()))
        return mask * spectrum

    def enhance(self, noisy):
        return torch.istft(
            self(noisy), n_fft=self.n_fft, hop_length=self.hop,
            window=self.window, length=noisy.shape[-1],
        )


def main():
    parser = argparse.ArgumentParser(description="CNN speech enhancement")
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_mask_cnn.pt"))
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"Input directory not found: {args.input}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    inputs = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if not inputs:
        parser.error(f"No WAV files in: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    expected = {p.name for p in inputs}
    extras = {p.name for p in args.output.iterdir() if p.is_file() and p.suffix.lower() == ".wav"} - expected
    if extras:
        parser.error("Remove stale WAV files from output/ before running: " + ", ".join(sorted(extras)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskCNN().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.inference_mode():
        for i, source in enumerate(inputs, 1):
            wave, sample_rate = sf.read(source, dtype="float32", always_2d=True)
            if sample_rate != SAMPLE_RATE or wave.shape[1] != 1:
                raise ValueError(f"Expected 16 kHz mono WAV: {source} (sr={sample_rate}, channels={wave.shape[1]})")
            noisy = torch.from_numpy(wave[:, 0].copy()).unsqueeze(0).to(device)
            restored = model.enhance(noisy)[0].cpu().numpy()
            if restored.shape[0] != wave.shape[0]:
                raise RuntimeError(f"Sample count mismatch: {source}")
            sf.write(args.output / source.name, restored, SAMPLE_RATE, subtype="PCM_16")
            print(f"[{i}/{len(inputs)}] {source.name}")

    outputs = {p.name for p in args.output.iterdir() if p.is_file() and p.suffix.lower() == ".wav"}
    if outputs != expected:
        raise RuntimeError(f"Output file set mismatch: missing={expected - outputs}, extra={outputs - expected}")
    print(f"Completed: {len(inputs)} WAV files -> {args.output} (16 kHz, mono)")


if __name__ == "__main__":
    main()
