from typing import Sequence
from abc import ABC, abstractmethod
import math

import torch
from torch import nn, Tensor

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import trange


def solve_sde(
        sde,
        z: Tensor,
        ts: float,
        tf: float,
        n_steps: int
) -> Tensor:
    tt = torch.linspace(ts, tf, n_steps + 1)[:-1]
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5

    path = [z]
    for t in tt:
        f, g = sde(z, t)
        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2
        path.append(z)

    return torch.stack(path)


def visualise_data(xs: Tensor, filename: str | None = None):
    # xs: (B, T, 3)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    xs = xs.detach().cpu()
    for xs_i in xs:
        ax.plot(xs_i[:, 0].numpy(), xs_i[:, 1].numpy(), xs_i[:, 2].numpy())

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    ax.set_xlabel('$z_1$', labelpad=0., fontsize=12)
    ax.set_ylabel('$z_2$', labelpad=.5, fontsize=12)
    ax.set_zlabel('$z_3$', labelpad=0, ha='center', fontsize=12)

    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


class SDE(nn.Module, ABC):
    @abstractmethod
    def drift(self, z: Tensor, t: Tensor, *args) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def vol(self, z: Tensor, t: Tensor, *args) -> Tensor:
        raise NotImplementedError

    def forward(self, z: Tensor, t: Tensor, *args) -> tuple[Tensor, Tensor]:
        drift = self.drift(z, t, *args)
        vol = self.vol(z, t, *args)
        return drift, vol


class StochasticLorenzSDE(SDE):
    def __init__(self, a: Sequence = (10., 28., 8 / 3), b: Sequence = (.15, .15, .15)):
        super().__init__()
        self.a = a
        self.b = b

    def drift(self, x: Tensor, t: Tensor, *args) -> Tensor:
        x1, x2, x3 = torch.split(x, [1, 1, 1], dim=1)
        a1, a2, a3 = self.a

        f1 = a1 * (x2 - x1)
        f2 = a2 * x1 - x2 - x1 * x3
        f3 = x1 * x2 - a3 * x3

        return torch.cat([f1, f2, f3], dim=1)

    def vol(self, x: Tensor, t: Tensor, *args) -> Tensor:
        x1, x2, x3 = torch.split(x, [1, 1, 1], dim=1)
        b1, b2, b3 = self.b

        g1 = x1 * b1
        g2 = x2 * b2
        g3 = x3 * b3

        return torch.cat([g1, g2, g3], dim=1)


def gen_data(
        batch_size: int,
        ts: float,
        tf: float,
        n_steps: int,
        noise_std: float,
        n_inner_steps: int = 100
) -> tuple[Tensor, Tensor]:
    sde = StochasticLorenzSDE()

    z0 = torch.randn(batch_size, 3)
    zs = solve_sde(sde, z0, ts, tf, n_steps=n_steps * n_inner_steps)
    zs = zs[::n_inner_steps]
    zs = zs.permute(1, 0, 2)  # (B, T+1, 3)

    mean = torch.mean(zs, dim=(0, 1))
    std = torch.std(zs, dim=(0, 1))

    eps = torch.randn_like(zs)
    xs = (zs - mean) / std + noise_std * eps

    ts_grid = torch.linspace(ts, tf, n_steps + 1)
    ts_grid = ts_grid[None, :, None].repeat(batch_size, 1, 1)

    return xs, ts_grid


# -------------------------------------------------------
# Time-conditional coupling flow x_t <-> z_t
# -------------------------------------------------------

class CouplingLayer(nn.Module):
    def __init__(self, data_size: int, hidden_size: int):
        super().__init__()
        self.s_net = nn.Sequential(
            nn.Linear(data_size + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, data_size),
        )
        self.t_net = nn.Sequential(
            nn.Linear(data_size + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, data_size),
        )

    def forward(self, x: Tensor, t: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        mask = mask.to(x.device)
        x_masked = x * mask

        inp = torch.cat([x_masked, t], dim=-1)
        s = self.s_net(inp)
        t_param = self.t_net(inp)

        s = s * (1.0 - mask)
        t_param = t_param * (1.0 - mask)

        y = x_masked + (1.0 - mask) * (x * torch.exp(s) + t_param)
        logdet = (s * (1.0 - mask)).sum(dim=-1)
        return y, logdet

    def inverse(self, y: Tensor, t: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        mask = mask.to(y.device)
        y_masked = y * mask

        inp = torch.cat([y_masked, t], dim=-1)
        s = self.s_net(inp)
        t_param = self.t_net(inp)

        s = s * (1.0 - mask)
        t_param = t_param * (1.0 - mask)

        x = y_masked + (1.0 - mask) * ((y - t_param) * torch.exp(-s))
        logdet = -(s * (1.0 - mask)).sum(dim=-1)
        return x, logdet


class TimeConditionalFlow(nn.Module):
    """
    Time-conditional RealNVP acting pointwise in time
    """

    def __init__(self, data_size: int, hidden_size: int, n_couplings: int):
        super().__init__()
        self.data_size = data_size

        self.couplings = nn.ModuleList(
            [CouplingLayer(data_size, hidden_size) for _ in range(n_couplings)]
        )

        mask1 = torch.tensor([1.0, 1.0, 0.0])
        mask2 = 1.0 - mask1
        self.masks = [mask1, mask2]

    def _forward_flow_single(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        # x: (N,D), t: (N,1)
        z = x
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i, layer in enumerate(self.couplings):
            mask = self.masks[i % 2]
            z, ld = layer(z, t, mask)
            logdet = logdet + ld
        return z, logdet

    def _inverse_flow_single(self, z: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        x = z
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for i, layer in reversed(list(enumerate(self.couplings))):
            mask = self.masks[i % 2]
            x, ld = layer.inverse(x, t, mask)
            logdet = logdet + ld
        return x, logdet

    def encode(self, x_seq: Tensor, t_seq: Tensor) -> tuple[Tensor, Tensor]:
        """
        x_seq: (B, T+1, D)
        t_seq: (B, T+1, 1)
        Returns:
          z_seq: (B, T+1, D)
          logdet_flow: (B,)
        """
        B, T1, D = x_seq.shape
        x_flat = x_seq.reshape(B * T1, D)
        t_flat = t_seq.reshape(B * T1, 1)

        z_flat, logdet_flat = self._forward_flow_single(x_flat, t_flat)
        z_seq = z_flat.view(B, T1, D)
        logdet_traj = logdet_flat.view(B, T1).sum(dim=-1)
        return z_seq, logdet_traj

    def sample(self, z_seq: Tensor, t_seq: Tensor) -> Tensor:
        """
        z_seq: (B, T+1, D)
        t_seq: (B, T+1, 1)
        Returns:
          x_seq: (B, T+1, D)
        """
        B, T1, D = z_seq.shape
        z_flat = z_seq.reshape(B * T1, D)
        t_flat = t_seq.reshape(B * T1, 1)

        x_flat, _ = self._inverse_flow_single(z_flat, t_flat)
        x_seq = x_flat.view(B, T1, D)
        return x_seq


# -------------------------------------------------------
# Affine, delta t-aware AR base as a flow eps => z
# -------------------------------------------------------

class AffineARBase(nn.Module):
    """
    Neural, Δt-aware AR base as a flow:

      z0 = m0(t0) + s0(t0) ⊙ ε0
      zk = zk-1 + Δt_k fθ(zk-1, tk) + sqrt(Δt_k) gθ(zk-1, tk) ⊙ εk

    εk ~ N(0, I). The map ε -> z is affine and triangular in time.
    """

    def __init__(self, latent_dim: int, hidden_size: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Initial: z0 | t0
        self.init_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * latent_dim),
        )

        # Drift f(z, t)
        self.drift_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, latent_dim),
        )

        # Diffusion log-scale log g(z, t)
        self.diff_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, latent_dim),
        )

    def encode(self, z_seq: Tensor, t_seq: Tensor) -> tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]:
        """
        Inverse base flow: z_{0:T}, t_{0:T} -> ε_{0:T}, logdet(z -> ε).

        z_seq: (B, T+1, D)
        t_seq: (B, T+1, 1)

        Returns:
          eps_seq:     (B, T+1, D)
          logdet_base: (B,)       log|det ∂ε/∂z|
          ldj_steps:   (B, T)     forward LDJ per step ε_k -> z_k (k>=1)
          dt_line:     (T,)       step sizes (shared across batch)
          drift:       (B, T, D)  drift at steps k>=1
          log_g:       (B, T, D)  log diffusion scale g(z_{k-1}, t_k)
        """
        B, T1, D = z_seq.shape
        device = z_seq.device
        dtype = z_seq.dtype

        eps_seq = torch.zeros_like(z_seq)

        # Shared time grid and Δt
        t = t_seq[..., 0]          # (B, T+1)
        t_line = t[0]              # (T+1,)

        if T1 > 1:
            dt_line = (t_line[1:] - t_line[:-1]).clamp_min(1e-6)  # (T,)
        else:
            dt_line = t_line.new_zeros(0)

        # Step 0: z0 = m0(t0) + s0(t0) * ε0
        t0 = t_seq[:, 0, :]  # (B,1)
        h0 = self.init_net(t0)  # (B,2D)
        m0, log_s0 = h0.chunk(2, dim=-1)
        std0 = torch.exp(log_s0)

        eps_seq[:, 0, :] = (z_seq[:, 0, :] - m0) / std0
        ldj0 = log_s0.sum(dim=-1)  # (B,)

        if T1 == 1:
            forward_ldj = ldj0
            logdet_base = -forward_ldj
            ldj_steps = z_seq.new_zeros(B, 0)
            drift = z_seq.new_zeros(B, 0, D)
            log_g = z_seq.new_zeros(B, 0, D)
            return eps_seq, logdet_base, ldj_steps, dt_line, drift, log_g

        #  transitions
        T = T1 - 1
        dt = dt_line.view(1, T, 1).expand(B, -1, 1)   # (B,T,1)

        z_prev = z_seq[:, :-1, :]  # (B,T,D)
        z_cur = z_seq[:, 1:, :]    # (B,T,D)
        t_k = t_seq[:, 1:, :]      # (B,T,1)

        inp = torch.cat([z_prev, t_k], dim=-1)        # (B,T,D+1)
        inp_flat = inp.reshape(B * T, D + 1)

        drift_flat = self.drift_net(inp_flat)         # (B*T,D)
        log_g_flat = self.diff_net(inp_flat)          # (B*T,D)

        drift = drift_flat.view(B, T, D)
        log_g = log_g_flat.view(B, T, D)

        dt_b = dt.expand(B, T, D)                     # (B,T,D)
        log_dt = 0.5 * torch.log(dt_b)                # (B,T,D)

        mu = z_prev + drift * dt_b                    # (B,T,D)
        log_std = log_g + log_dt                      # (B,T,D)
        std = torch.exp(log_std)                      # (B,T,D)

        eps_trans = (z_cur - mu) / std                # (B,T,D)
        eps_seq[:, 1:, :] = eps_trans

        # Forward LDJ
        ldj_steps = log_std.sum(dim=-1)               # (B,T)

        forward_ldj = ldj0 + ldj_steps.sum(dim=-1)    # (B,)
        logdet_base = -forward_ldj                    # z->ε

        return eps_seq, logdet_base, ldj_steps, dt_line, drift, log_g

    def sample(self, n_traj: int, t_line: Tensor) -> Tensor:
        """
        Sample z_{0:T} from the base flow: ε_{0:T} ~ N(0,I) -> z_{0:T}.

        n_traj: int
        t_line: (T+1,) time grid

        Returns:
          z_seq: (n_traj, T+1, D)
        """
        device = t_line.device
        dtype = t_line.dtype

        T1 = t_line.shape[0]
        T = T1 - 1
        D = self.latent_dim

        z_seq = torch.zeros(n_traj, T1, D, device=device, dtype=dtype)
        eps = torch.randn(n_traj, T1, D, device=device, dtype=dtype)

        # Step 0
        t0 = t_line[0].view(1, 1).expand(n_traj, 1)   # (B,1)
        h0 = self.init_net(t0)
        m0, log_s0 = h0.chunk(2, dim=-1)
        std0 = torch.exp(log_s0)
        z_seq[:, 0, :] = m0 + std0 * eps[:, 0, :]

        if T == 0:
            return z_seq

        dt_line = (t_line[1:] - t_line[:-1]).clamp_min(1e-6)  # (T,)

        z_prev = z_seq[:, 0, :]

        for k in range(T):
            t_k = t_line[k + 1].view(1, 1).expand(n_traj, 1)   # (B,1)
            dt_k = dt_line[k].view(1, 1).expand(n_traj, 1)     # (B,1)

            inp = torch.cat([z_prev, t_k], dim=-1)             # (B,D+1)
            drift = self.drift_net(inp)                        # (B,D)
            log_g = self.diff_net(inp)                         # (B,D)

            std_k = torch.exp(log_g) * dt_k.sqrt()             # (B,D)
            mean_k = z_prev + drift * dt_k                     # (B,D)

            z_next = mean_k + std_k * eps[:, k + 1, :]
            z_seq[:, k + 1, :] = z_next
            z_prev = z_next

        return z_seq


def sample_trajectories(
        flow: TimeConditionalFlow,
        base: AffineARBase,
        n_traj: int,
        ts_min: float,
        ts_max: float,
        n_steps: int,
        device: torch.device
) -> Tensor:
    t_line = torch.linspace(ts_min, ts_max, n_steps + 1, device=device)  # (T+1,)
    z_seq = base.sample(n_traj, t_line)                                  # (B,T+1,D)
    t_seq = t_line.view(1, -1, 1).expand(n_traj, -1, 1)                  # (B,T+1,1)
    x_seq = flow.sample(z_seq, t_seq)                                    # (B,T+1,D)
    return x_seq

def action_reg(
        drift: Tensor,
        log_g: Tensor,
        dt_line: Tensor,
        alpha_diff: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:

    B, T, D = drift.shape
    device = drift.device

    dt = dt_line.view(1, T).to(device)  # (1,T)

    # Path action: ∑ Δt (||f||² + α ||g||²), averaged over batch/time
    kinetic = (drift ** 2).sum(dim=-1) * dt          # (B,T)
    g = torch.exp(log_g)
    diffusion_energy = (g ** 2).sum(dim=-1) * dt     # (B,T)

    action = (kinetic + alpha_diff * diffusion_energy).mean()

    return action



# -------------------------------------------------------
# Training loop with geometric path regularisation
# -------------------------------------------------------

def train_flow(
        flow: TimeConditionalFlow,
        base: AffineARBase,
        xs: Tensor,
        ts: Tensor,
        ts_min: float,
        ts_max: float,
        n_steps: int,
        n_iter: int = 4000,
        batch_size: int = 64,
        lam_act: float = 1e-3,
):
    """
    xs: (N_traj, T+1, D)
    ts: (N_traj, T+1, 1)
    """
    device = xs.device
    N_traj = xs.shape[0]

    params = list(flow.parameters()) + list(base.parameters())
    optim = torch.optim.Adam(params, lr=1e-3)

    pbar = trange(n_iter)
    for step in pbar:
        idx = torch.randint(0, N_traj, (batch_size,), device=device)
        x_batch = xs[idx]  # (B,T+1,D)
        t_batch = ts[idx]  # (B,T+1,1)

        # x -> z via RealNVP
        z_batch, logdet_flow = flow.encode(x_batch, t_batch)  # (B,T+1,D),(B,)

        # z -> eps via affine AR base, plus per-step LDJ and dt and drift/log_g
        eps_batch, logdet_base, _, dt_line, drift, log_g = base.encode(z_batch, t_batch)


        # Standard normal prior on ε (drop constant)
        log_p_eps = -0.5 * (eps_batch ** 2).sum(dim=(1, 2))  # (B,)

        log_px = log_p_eps + logdet_flow + logdet_base       # (B,)
        nll = -log_px.mean()

        # Geometric path + volume terms
        action = action_reg(
            drift, log_g, dt_line
        )

        loss = nll + lam_act * action

        optim.zero_grad()
        loss.backward()
        optim.step()

        pbar.set_description(
            f"step {step} | loss {loss.item():.4f} | nll {nll.item():.4f} | action {action.item():.4f}"
        )

        if step % 100 == 0:
            xs_gen = sample_trajectories(
                flow,
                base,
                n_traj=6,
                ts_min=ts_min,
                ts_max=ts_max,
                n_steps=n_steps,
                device=device,
            )
            visualise_data(xs_gen, filename="sample.jpg")


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2 ** 10
    ts0 = 0.
    tf = 1.
    n_steps = 40
    noise_std = .01
    data_size = 3

    xs, ts = gen_data(batch_size, ts0, tf, n_steps, noise_std)

    visualise_data(xs[:6], filename="lorenz_data_example.jpg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs = xs.to(device)
    ts = ts.to(device)

    hidden_size = 64
    n_couplings = 4

    flow = TimeConditionalFlow(data_size, hidden_size, n_couplings).to(device)
    base = AffineARBase(latent_dim=data_size, hidden_size=hidden_size).to(device)

    train_flow(
        flow,
        base,
        xs,
        ts,
        ts_min=ts0,
        ts_max=tf,
        n_steps=n_steps,
        n_iter=4000,
        batch_size=64,
        lam_act=1.0,      # path action (kinetic + diffusion)
    )

    xs_gen_final = sample_trajectories(
        flow,
        base,
        n_traj=6,
        ts_min=ts0,
        ts_max=tf,
        n_steps=n_steps,
        device=device,
    )
    visualise_data(xs_gen_final, filename="lorenz_flow_final.jpg")
