# Action-regularised sde flows for stochastic dynamics

A small experiment in learning noisy Lorenz-63 trajectories with an explicit SDE base in latent space and a time dependent coupling flow in data space.

  - a time conditional RealNVP acting pointwise in time, x_t <-> z_t,
  - and an affine, Δt-aware autoregressive base that looks like a discretised SDE for z_{0:T}.

The base flow is triangular in time:
```
  ε_{0:T} ~ N(0, I),

  z_0 = m_θ(t_0) + s_θ(t_0) ⊙ ε_0,
  z_k = z_{k-1}
        + Δt_k f_θ(z_{k-1}, t_k)
        + sqrt(Δt_k) g_θ(z_{k-1}, t_k) ⊙ ε_k,   k = 1,...,T.
```
Together with the time conditional flow x_t <-> z_t, each trajectory x_{0:T} has density
```
  log p_θ(x_{0:T})
    = log p(ε_{0:T})
      + log |det ∂z_{0:T} / ∂ε_{0:T}|
      + log |det ∂x_{0:T} / ∂z_{0:T}|,
```
with all three terms computed exactly from the affine AR base and the coupling flow.

To bias the latent dynamics toward “nice” SDE paths, we add a geometric path regulariser on the drift and diffusion of the base,
```
  A_θ(z_{0:T}, t_{0:T})
    ≈ sum_{k=1}^T Δt_k (
         ||f_θ(z_{k-1}, t_k)||^2
       + α_diff ||g_θ(z_{k-1}, t_k)||^2
      ),
```
and optimise
```
  maximize_{θ}  E_{x_{0:T} ~ data} [
      log p_θ(x_{0:T})
    - λ_act A_θ(z_{0:T}, t_{0:T})
  ];
```
where
  x_{0:T} is the observed trajectory,
  z_{0:T} comes from the affine AR SDE base,
  x_t <-> z_t is given by the time conditional RealNVP.
