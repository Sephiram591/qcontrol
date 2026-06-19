"""Time-domain jaxquantum mesolve setup for the SnV dynamic Hamiltonian.

This file is meant to sit next to your Hamiltonian helper module.  It keeps all
states by calling ``get_dynamic_hamiltonian(..., included_states=None)`` and then
applies real-valued time signals to the microwave and optical drive operators:

    H(t) = H0 + b_signal(t) * Hb + optical_signal(t) * H_optical

The Hamiltonian and time units must be reciprocal.  For example, if H is in
rad / us, then tlist should be in us and the carrier frequencies below should be
angular frequencies in rad / us.  If your Hamiltonian entries are in cycles / us
(MHz), multiply the Hamiltonian or the signal carrier frequencies by 2*pi so the
Schroedinger/Master-equation phase convention is consistent.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import jaxquantum as jqt
import matplotlib.pyplot as plt
from qcontrol.snv120.hamiltonian_jqt import get_dynamic_hamiltonian
import numpy as np
from typing import NamedTuple


RealSignal = Callable[[float], jnp.ndarray]


class TimeDomainResult(NamedTuple):
    """Container for time-domain simulation outputs."""
    tlist: jnp.ndarray
    rhos: jqt.Qarray
    H0: jqt.Qarray
    Hb: jqt.Qarray
    H_optical: jqt.Qarray
    c_ops: jqt.Qarray
    populations_basis: jnp.ndarray
    populations_eigenbasis: jnp.ndarray


def make_real_cosine_signal(
    amplitude: float,
    omega: float,
    phase: float = 0.0,
    dc_offset: float = 0.0,
    envelope: Optional[RealSignal] = None,
) -> RealSignal:
    """Return a real-valued cosine signal usable inside a JAX Hamiltonian.

    Parameters
    ----------
    amplitude : float
        Signal amplitude multiplying the target operator.
    omega : float
        Angular carrier frequency in inverse ``tlist`` units.
    phase : float, optional
        Carrier phase in radians.
    dc_offset : float, optional
        Constant offset added to the signal.
    envelope : callable, optional
        Real envelope function ``envelope(t)``.  If omitted, the envelope is 1.
    """

    def signal(t):
        env = 1.0 if envelope is None else envelope(t)
        return dc_offset + amplitude * env * jnp.cos(omega * t + phase)

    return signal


def make_real_sampled_signal(t_samples, y_samples) -> RealSignal:
    """Return a real-valued interpolated signal from sampled data.

    ``t_samples`` must cover the full integration interval.  Values outside the
    sampled range are held at the first or last sample by ``jnp.interp``.
    """

    t_samples = jnp.asarray(t_samples)
    y_samples = jnp.asarray(y_samples)

    def signal(t):
        return jnp.interp(t, t_samples, y_samples)

    return signal


def square_window(t_start: float, t_stop: float) -> RealSignal:
    """Return a JAX-compatible real square pulse envelope."""

    def envelope(t):
        return jnp.where((t >= t_start) & (t <= t_stop), 1.0, 0.0)

    return envelope

def _basis_labels(space_dims: Sequence[int]) -> list[str]:
    """Generate compact labels for every tensor-product basis vector."""

    labels = []
    total_dim = int(np.prod(space_dims))
    for flat_index in range(total_dim):
        multi_index = np.unravel_index(flat_index, tuple(space_dims))
        if len(space_dims) == 4:
            manifold = "g" if multi_index[0] == 0 else "e"
            labels.append(
                f"{flat_index}: {manifold}, orb={multi_index[1]}, "
                f"S={multi_index[2]}, I={multi_index[3]}"
            )
        elif len(space_dims) == 2 and space_dims[0] == 2:
            manifold = "g" if multi_index[0] == 0 else "e"
            labels.append(f"{flat_index}: {manifold}, pair={multi_index[1]}")
        else:
            labels.append(f"{flat_index}: {multi_index}")
    return labels


def populations_in_basis(rhos: jqt.Qarray) -> jnp.ndarray:
    """Return populations from the diagonal of density matrices in current basis.

    Returns
    -------
    jax.Array, shape (num_saved_times, hilbert_dim)
        Diagonal density-matrix populations.
    """

    rho_data = rhos.to_dense().data
    return jnp.real(jnp.diagonal(rho_data, axis1=-2, axis2=-1))


def populations_in_static_eigenbasis(rhos: jqt.Qarray, H0: jqt.Qarray) -> jnp.ndarray:
    """Return populations after rotating density matrices into H0 eigenbasis."""

    _, eigenstates = jqt.eigenstates(H0)
    states = eigenstates.to_dense().data[..., :, 0]  # (num_states, dim)
    U = jnp.swapaxes(states, -1, -2)                 # columns are eigenvectors
    rho_data = rhos.to_dense().data                  # (num_times, dim, dim)
    rho_eig = jnp.einsum("ia,tij,jb->tab", jnp.conj(U), rho_data, U)
    return jnp.real(jnp.diagonal(rho_eig, axis1=-2, axis2=-1))

@jax.jit(static_argnames=["included_states", "solver_options", "saveat_tlist"])
def run_time_domain_mesolve(
    B,
    theta,
    phi,
    excited_ground_split,
    excited_state_lifetime,
    resonant_pump_polarization,
    B_drive_orientation,
    alpha,
    beta,
    alpha_exc,
    beta_exc,
    A_gnd,
    Ax_gnd,
    Ay_gnd,
    A_exc,
    Ax_exc,
    Ay_exc,
    upsilon_gnd,
    upsilon_exc,
    tlist: jnp.ndarray,
    b_signal: jnp.ndarray,
    optical_signal: jnp.ndarray,
    initial_flat_index: int = 0,
    saveat_tlist=None,
    solver_options=None,
    included_states=None,
) -> TimeDomainResult:
    """Run master-equation dynamics with real signals on Hb and H_optical.

    Parameters
    ----------
    dynamic_hamiltonian_kwargs : mapping
        Keyword arguments for ``get_dynamic_hamiltonian`` other than
        ``included_states``/``include_states``.
    tlist : array_like
        Integration times.
    b_signal, optical_signal : array_like
        Real scalar functions of time.  They multiply ``Hb`` and ``H_optical``.
    initial_flat_index : int, optional
        Initial tensor-product basis state as a flat basis index.  The default
        is 0, i.e. usually ``|ground, orbital=0, electron=0, nuclear=0>``.
    saveat_tlist : array_like, optional
        Times at which to save density matrices.  Defaults to all ``tlist``.
    solver_options : jaxquantum SolverOptions, optional
        Passed through to ``jqt.mesolve``.
    """

    H0, Hb, Hs_optical, c_ops = get_dynamic_hamiltonian(
        B,
        theta,
        phi,
        excited_ground_split,
        excited_state_lifetime,
        resonant_pump_polarization,
        B_drive_orientation,
        alpha,
        beta,
        alpha_exc,
        beta_exc,
        A_gnd,
        Ax_gnd,
        Ay_gnd,
        A_exc,
        Ax_exc,
        Ay_exc,
        upsilon_gnd,
        upsilon_exc,
        included_states=included_states,
    )

    b_signal_fn = make_real_sampled_signal(tlist, b_signal)
    optical_signal_fn = make_real_sampled_signal(tlist, optical_signal)
    H_optical = Hs_optical[0] + Hs_optical[1]

    def H_t(t):
        # b_signal and optical_signal are real scalars; Hb/H_optical carry the
        # operator structure.  This is the lab-frame Hamiltonian, not an RWA form.
        return H0 + b_signal_fn(t) * Hb + optical_signal_fn(t) * H_optical

    tlist = jnp.asarray(tlist)
    saveat_tlist = tlist if saveat_tlist is None else jnp.asarray(saveat_tlist)

    space_dims = tuple(int(d) for d in H0.space_dims)
    initial_multi_index = jnp.unravel_index(initial_flat_index, space_dims)
    psi0 = jqt.basis_like(H0, initial_multi_index)
    rho0 = psi0.to_dm()
    rhos = jqt.mesolve(
        H_t,
        rho0,
        tlist,
        saveat_tlist=saveat_tlist,
        c_ops=c_ops,
        solver_options=solver_options,
    )

    basis_pops = populations_in_basis(rhos)
    eigen_pops = populations_in_static_eigenbasis(rhos, H0)
    # labels = _basis_labels(space_dims)
    # eigen_labels = [f"eig {i}" for i in range(basis_pops.shape[-1])]

    return TimeDomainResult(
        tlist=saveat_tlist,
        rhos=rhos,
        H0=H0,
        Hb=Hb,
        H_optical=H_optical,
        c_ops=c_ops,
        populations_basis=basis_pops,
        populations_eigenbasis=eigen_pops,
        # basis_labels=labels,
        # eigenbasis_labels=eigen_labels,
    )


def plot_state_populations(
    result: TimeDomainResult,
    basis: str = "eigen",
    max_states: Optional[int] = None,
    min_peak_population: float = 0.0,
):
    """Plot all, or a filtered subset, of state populations versus time.

    Parameters
    ----------
    result : TimeDomainResult
        Output of ``run_time_domain_mesolve``.
    basis : {'eigen', 'basis'}, optional
        ``'eigen'`` plots populations in the static-Hamiltonian eigenbasis.
        ``'basis'`` plots tensor-product basis populations.
    max_states : int, optional
        If provided, only the largest-peak-population states are plotted.
    min_peak_population : float, optional
        Suppress states whose peak population never exceeds this threshold.
    """

    if basis == "eigen":
        populations = np.asarray(result.populations_eigenbasis)
        title = "State populations in static eigenbasis"
    elif basis == "basis":
        populations = np.asarray(result.populations_basis)
        title = "State populations in tensor-product basis"
    else:
        raise ValueError("basis must be 'eigen' or 'basis'.")

    t = np.asarray(result.tlist)
    peak = populations.max(axis=0)
    keep = np.where(peak >= min_peak_population)[0]
    if max_states is not None and len(keep) > max_states:
        keep = keep[np.argsort(peak[keep])[-max_states:]]
        keep = keep[np.argsort(keep)]

    fig, ax = plt.subplots(figsize=(11, 6))
    for state_index in keep:
        ax.plot(t, populations[:, state_index], label=state_index)

    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    if len(keep) <= 24:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(
            0.01,
            0.99,
            f"{len(keep)} states plotted; legend omitted",
            transform=ax.transAxes,
            va="top",
        )
    fig.tight_layout()
    return fig, ax