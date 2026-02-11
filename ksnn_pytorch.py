#!/usr/bin/env python3
"""
Single-file PyTorch rewrite of the KSNN TensorFlow/Keras implementation.

Includes:
- Method-of-slices phase operators (Shift / ShiftBack)
- KSNN model definition (state branch + phase branch)
- build_model helper that mirrors original API style
- trajectory rollout helper
- simple demo `main()` that reads Compile_Dat.p
"""

from __future__ import annotations

import pickle
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc() -> int:
    """Reduced latent dimension from the original implementation."""
    return 7



class Shift(nn.Module):
    """
    Method-of-slices input shift.

    Input:  [B, N]
    Output: [B, N+1] where the extra column is extracted phase.

    Notes:
    - If n_fixed is provided, behavior matches original fixed-size setup.
    - If n_fixed is None, N is inferred from the incoming tensor shape.
    """

    def __init__(self, n_fixed: Optional[int] = None):
        super().__init__()
        self.n_fixed = n_fixed

    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        x = dense.to(torch.float64)
        n = self.n_fixed if self.n_fixed is not None else x.shape[1]
        if x.shape[1] != n:
            raise ValueError(f"Shift expected input with N={n}, got {x.shape[1]}")

        # phase from first Fourier mode of ifft(x)
        utifft = torch.fft.ifft(x.to(torch.complex128), dim=1)
        first_mode = utifft[:, 1:2]
        phase = torch.atan2(first_mode.imag, first_mode.real)  # [B,1]

        # shift in Fourier space
        utfft = torch.fft.fft(x.to(torch.complex128), dim=1)
        seq1 = torch.linspace(0.0, n / 2 - 1.0, int(n / 2), dtype=torch.float64, device=x.device)
        seq2 = torch.linspace(-n / 2, -1.0, int(n / 2), dtype=torch.float64, device=x.device)
        k = torch.cat([seq1, seq2], dim=0).to(torch.complex128)  # [N]

        exp_term = torch.exp(1j * (phase.to(torch.complex128) @ k.unsqueeze(0)))  # [B,N]
        shifted_fft = utfft * exp_term
        shifted = torch.fft.ifft(shifted_fft, dim=1).real.to(torch.float64)

        return torch.cat([shifted, phase.to(torch.float64)], dim=1)


class ShiftBack(nn.Module):
    """
    Inverse method-of-slices shift.

    Input:  [B, N+1] as [state, phase]
    Output: [B, N]
    """

    def __init__(self, n_fixed: Optional[int] = None):
        super().__init__()
        self.n_fixed = n_fixed

    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        x = dense.to(torch.float64)
        if self.n_fixed is not None:
            n = self.n_fixed
        else:
            n = x.shape[1] - 1

        if x.shape[1] != n + 1:
            raise ValueError(f"ShiftBack expected input with N+1={n+1}, got {x.shape[1]}")

        phase = x[:, n:n + 1]
        state = x[:, :n]

        utfft = torch.fft.fft(state.to(torch.complex128), dim=1)
        seq1 = torch.linspace(0.0, n / 2 - 1.0, int(n / 2), dtype=torch.float64, device=x.device)
        seq2 = torch.linspace(-n / 2, -1.0, int(n / 2), dtype=torch.float64, device=x.device)
        k = torch.cat([seq1, seq2], dim=0).to(torch.complex128)

        exp_term = torch.exp(-1j * (phase.to(torch.complex128) @ k.unsqueeze(0)))
        unshifted_fft = utfft * exp_term
        return torch.fft.ifft(unshifted_fft, dim=1).real.to(torch.float64)


class KSNNModel(nn.Module):
    """PyTorch version of the original KSNN architecture."""

    def __init__(self, n: int, u_mat: np.ndarray, dtheta_mean: float, dtheta_std: float):
        super().__init__()
        self.n = n
        self.r = trunc()

        u_tensor = torch.as_tensor(u_mat, dtype=torch.float64)
        self.register_buffer("u_mat", u_tensor)

        self.register_buffer("dtheta_mean", torch.tensor(float(dtheta_mean), dtype=torch.float64))
        self.register_buffer("dtheta_std", torch.tensor(float(dtheta_std), dtype=torch.float64))

        self.shift_in = Shift(n_fixed=n)
        self.shift_out = ShiftBack(n_fixed=n)

        # Encoder
        self.dense_in1 = nn.Linear(n, 500, dtype=torch.float64)
        self.dense_in2 = nn.Linear(500, self.r, dtype=torch.float64)

        # State evolution
        self.timestep1 = nn.Linear(self.r, 200, dtype=torch.float64)
        self.timestep2 = nn.Linear(200, 200, dtype=torch.float64)
        self.timestep3 = nn.Linear(200, self.r, dtype=torch.float64)

        # Decoder
        self.dense_out1 = nn.Linear(self.r, 500, dtype=torch.float64)
        self.dense_out2 = nn.Linear(500, n - self.r, dtype=torch.float64)

        # Additional reg branch used by original code path
        self.dense_out_reg1 = nn.Linear(self.r, 500, dtype=torch.float64)
        self.dense_out_reg2 = nn.Linear(500, self.r, dtype=torch.float64)

        # Phase branch
        self.phase1 = nn.Linear(self.r, 500, dtype=torch.float64)
        self.phase2 = nn.Linear(500, 50, dtype=torch.float64)
        self.phase3 = nn.Linear(50, 500, dtype=torch.float64)
        self.phase4 = nn.Linear(500, 1, dtype=torch.float64)

    def _pca_input(self, x: torch.Tensor) -> torch.Tensor:
        # TensorFlow: einsum("ij,jk->ik", x, U)
        return x @ self.u_mat

    def _output_map(self, x: torch.Tensor) -> torch.Tensor:
        # TensorFlow: einsum("ij,kj->ik", x, U)
        return x @ self.u_mat.T

    def forward(self, main_input: torch.Tensor) -> torch.Tensor:
        x = main_input.to(torch.float64)

        # Remove phase
        shifted = self.shift_in(x)  # [B, N+1]
        phase_input = shifted[:, self.n:self.n + 1]
        shift_input = shifted[:, :self.n]

        # Linear reduction
        pca_input = self._pca_input(shift_input)

        # Nonlinear reduction
        encode = torch.sigmoid(self.dense_in1(pca_input))
        encode = torch.tanh(self.dense_in2(encode))

        # Fusion
        hidden = encode + pca_input[:, :self.r]

        # Save for phase branch
        phase_hidden = hidden

        # Timestepping
        hidden = torch.sigmoid(self.timestep1(hidden))
        hidden = torch.sigmoid(self.timestep2(hidden))
        hidden = self.timestep3(hidden)

        # Decoder (linear + nonlinear)
        pca_output_linear = F.pad(hidden, (0, self.n - self.r), mode="constant", value=0.0)

        decode = torch.sigmoid(self.dense_out1(hidden))
        decode = self.dense_out2(decode)

        conc = torch.sigmoid(self.dense_out_reg1(hidden))
        conc = self.dense_out_reg2(conc)

        decode_full = torch.cat([conc, decode], dim=1)
        pca_output = decode_full + pca_output_linear

        # Map to full space
        state_output = self._output_map(pca_output)

        # Phase branch
        phase = torch.sigmoid(self.phase1(phase_hidden))
        phase = torch.sigmoid(self.phase2(phase))
        phase = torch.sigmoid(self.phase3(phase))
        phase = self.phase4(phase)

        phase = phase * self.dtheta_std + self.dtheta_mean
        phase_output = phase + phase_input

        # Shift back
        final_cat = torch.cat([state_output, phase_output], dim=1)
        return self.shift_out(final_cat)


def build_model(
    n: int,
    u_mat: np.ndarray,
    dtheta_mean: float,
    dtheta_std: float,
    lr: float = 1e-4,
    epochs: int = 200,
) -> Tuple[KSNNModel, int, torch.optim.Optimizer]:
    """Build model and optimizer in a style similar to original buildmodel()."""
    model = KSNNModel(n=n, u_mat=u_mat, dtheta_mean=dtheta_mean, dtheta_std=dtheta_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, epochs, optimizer


def trajectory(
    model: KSNNModel,
    u0: np.ndarray,
    time_units: float,
    u_std: float,
    u_mean: float,
    device: str = "cpu",
):
    """Autoregressive rollout equivalent to KSNN_Example.py trajectory behavior."""
    model = model.to(device)
    model.eval()

    ut_predict = np.array(u0, dtype=np.float64)
    tt_pred = np.zeros(1, dtype=np.float64)
    h = 2.0

    with torch.no_grad():
        while tt_pred[-1] < time_units:
            x = torch.from_numpy(ut_predict[-1:, :]).to(device=device, dtype=torch.float64)
            y = model(x).cpu().numpy()
            ut_predict = np.append(ut_predict, y, axis=0)

            # Convert in-slice time to real time
            fmode = abs(np.fft.ifft(ut_predict[-1, :] * u_std + u_mean)[1])
            tt_pred = np.append(tt_pred, tt_pred[-1] + h * fmode)

    return tt_pred, ut_predict.T


def main() -> None:
    u, tt, stats, u_mat = pickle.load(open("Compile_Dat.p", "rb"))
    u = (u - stats[0]) / stats[1]
    n, m = u.shape

    model, epochs, optimizer = build_model(n, u_mat, stats[2], stats[3])
    print(model)
    print(f"epochs={epochs}, optimizer={optimizer.__class__.__name__}")

    time_units = 80
    start = np.random.randint(200, m - 4 * time_units)
    u0 = u[:, start:start + 1].T

    ttpred, uts_nn = trajectory(model, u0, time_units, stats[1], stats[0])
    print("rollout finished", ttpred.shape, uts_nn.shape)


if __name__ == "__main__":
    main()
