"""
JAX-differentiable Hamiltonian model for SnV117 center in diamond.

This is a pure-JAX rewrite of the QuTiP/NumPy implementation.  The important
change is that all operators are dense `jax.numpy` arrays and diagonalization is
done with `jnp.linalg.eigh`, so gradients can flow through scalar parameters
such as strain, magnetic-field angles, hyperfine tensor entries, SOC, etc.

Notes:
- Gradients through `eigh` are well-defined only away from exact/near
  degeneracies.  Eigenvector gradients are especially ill-conditioned near
  level crossings.
- `PLE_spectrum` returns shape `(n_B, n_f)` even for a single B value.  This is
  usually more convenient for `jit`, `vmap`, and optimization.  Index `[0]` if
  you want the old 1D single-B behavior.
"""

from __future__ import annotations

from functools import partial
import numpy as _np

import jax
import jax.numpy as jnp

import qcontrol.snv120.parameters as params


# Enable this if your energy splittings need float64 precision.  This must run
# before arrays are created.
jax.config.update("jax_enable_x64", True)

_DTYPE = jnp.complex128
_RTOL_DTYPE = jnp.float64


def _as_complex(x):
    return jnp.asarray(x, dtype=_DTYPE)


def _as_real(x):
    return jnp.asarray(x, dtype=_RTOL_DTYPE)


def _jmat_numpy(j: float, axis: str) -> _np.ndarray:
    """Dense angular momentum matrix matching QuTiP's jmat convention.

    Basis ordering is |j>, |j-1>, ..., |-j>.
    """
    n = int(round(2 * float(j) + 1))
    m_vals = _np.arange(float(j), -float(j) - 1, -1, dtype=float)

    jp = _np.zeros((n, n), dtype=_np.complex128)
    for col, m in enumerate(m_vals):
        mp = m + 1.0
        if mp <= float(j) + 1e-12:
            row = int(round(float(j) - mp))
            if 0 <= row < n:
                jp[row, col] = _np.sqrt(float(j) * (float(j) + 1.0) - m * (m + 1.0))

    jm = jp.conj().T
    if axis == "x":
        return 0.5 * (jp + jm)
    if axis == "y":
        return -0.5j * (jp - jm)
    if axis == "z":
        return _np.diag(m_vals).astype(_np.complex128)
    raise ValueError("axis must be one of 'x', 'y', or 'z'")


def jmat(j: float, axis: str):
    return jnp.asarray(_jmat_numpy(j, axis), dtype=_DTYPE)


def qeye(n: int):
    return jnp.eye(int(n), dtype=_DTYPE)


def kron3(a, b, c):
    return jnp.kron(jnp.kron(a, b), c)


def _build_static_ops(S: float | None = None, Sn: float | None = None):
    """Build operators whose shapes are static and not differentiated."""
    S = params.S if S is None else S
    Sn = params.Sn if Sn is None else Sn

    X = jmat(S, "x")
    Y = jmat(S, "y")
    Z = jmat(S, "z")
    I = qeye(int(round(2 * float(S) + 1)))

    Xn = jmat(Sn, "x")
    Yn = jmat(Sn, "y")
    Zn = jmat(Sn, "z")
    In = qeye(int(round(2 * float(Sn) + 1)))

    J2 = (
        (float(S) * (float(S) + 1.0) + float(Sn) * (float(Sn) + 1.0))
        * kron3(I, I, In)
        + 2.0 * (kron3(I, X, Xn) + kron3(I, Y, Yn) + kron3(I, Z, Zn))
    )

    # In the e+, e- basis:
    # sigma_z from the ex, ey basis becomes -sigma_x
    # sigma_x from the ex, ey basis becomes -sigma_y
    sigma_z_xy_in_pm = -2.0 * X
    sigma_x_xy_in_pm = -2.0 * Y

    p = jnp.stack(
        [
            kron3(2.0 * X, I, In),  # px
            kron3(2.0 * Y, I, In),  # py
            kron3(2.0 * I, I, In),  # pz
        ],
        axis=0,
    )

    return {
        "S": S,
        "Sn": Sn,
        "X": X,
        "Y": Y,
        "Z": Z,
        "I": I,
        "Xn": Xn,
        "Yn": Yn,
        "Zn": Zn,
        "In": In,
        "S_ops": (X, Y, Z),
        "N_ops": (Xn, Yn, Zn),
        "sigma_z_xy_in_pm": sigma_z_xy_in_pm,
        "sigma_x_xy_in_pm": sigma_x_xy_in_pm,
        "J2": J2,
        "p": p,
        "dim": int(I.shape[0] * I.shape[0] * In.shape[0]),
    }


OPS = _build_static_ops()


def hf_block(B, orbital_op, ops=OPS):
    """sum_ij B_ij orbital_op ⊗ S_i ⊗ N_j."""
    B = _as_complex(B)
    out = jnp.zeros_like(ops["J2"])
    for i in range(3):
        for j in range(3):
            out = out + B[i, j] * kron3(orbital_op, ops["S_ops"][i], ops["N_ops"][j])
    return out


def Hhf(A, Ax, Ay, ops=OPS):
    return (
        hf_block(A, ops["I"], ops)
        + hf_block(Ax, ops["sigma_z_xy_in_pm"], ops)
        + hf_block(Ay, ops["sigma_x_xy_in_pm"], ops)
    )


def Hsoc(L, ops=OPS):
    return 2.0 * _as_complex(L) * kron3(ops["Z"], ops["Z"], ops["In"])


def Hioc(upsilon, ops=OPS):
    return 2.0 * _as_complex(upsilon) * kron3(ops["Z"], ops["I"], ops["Zn"])


def Hegx(alpha, ops=OPS):
    return -2.0 * _as_complex(alpha) * kron3(ops["X"], ops["I"], ops["In"])


def Hegy(beta, ops=OPS):
    return 2.0 * _as_complex(beta) * kron3(ops["Y"], ops["I"], ops["In"])


def Href_matrix(L, alpha, beta, ops=OPS):
    return Hsoc(L, ops) + Hegx(alpha, ops) + Hegy(beta, ops)


def hamiltonian_matrix(
    bx,
    by,
    bz,
    rg,
    q,
    A,
    Ax,
    Ay,
    L,
    alpha,
    beta,
    upsilon=0.0,
    ops=OPS,
):
    """Dense Hermitian Hamiltonian matrix."""
    I, X, Y, Z, In = ops["I"], ops["X"], ops["Y"], ops["Z"], ops["In"]
    Xn, Yn, Zn = ops["Xn"], ops["Yn"], ops["Zn"]

    bx = _as_complex(bx)
    by = _as_complex(by)
    bz = _as_complex(bz)
    rg = _as_complex(rg)
    q = _as_complex(q)

    Hbxe = bx * kron3(I, X, In)
    Hbye = by * kron3(I, Y, In)
    Hbze = bz * kron3(I, Z, In)

    Hbxn = rg * bx * kron3(I, I, Xn)
    Hbyn = rg * by * kron3(I, I, Yn)
    Hbzn = rg * bz * kron3(I, I, Zn)

    Hbzo = (q / 2.0) * bz * kron3(Z, I, In)

    H = (
        Href_matrix(L, alpha, beta, ops)
        + Hbxe
        + Hbye
        + Hbze
        + Hbxn
        + Hbyn
        + Hbzn
        + Hbzo
        + Hhf(A, Ax, Ay, ops)
        + Hioc(upsilon, ops)
    )

    # Symmetrize to protect `eigh` from tiny numerical anti-Hermitian pieces
    # when traced values are complex.
    return 0.5 * (H + H.conj().T)


def alignment_from_eigenvectors(U, J2):
    """Return <psi_i|J2|psi_i> for eigenvectors stored as columns of U."""
    return jnp.real(jnp.einsum("ai,ab,bi->i", U.conj(), J2, U))


def solve_hamiltonian(
    B,
    theta,
    phi,
    q,
    A,
    Ax,
    Ay,
    L,
    alpha,
    beta,
    rg=None,
    upsilon=0.0,
):
    """Solve the Hamiltonian for a sweep of B-field strength.

    Returns:
        E:         shape (n_B, n_states)
        Eref:      shape (n_states,)
        U:         shape (n_B, dim, n_states), eigenvectors are columns
        alignment: shape (n_B, n_states)
    """
    rg = params.rg if rg is None else rg

    # B = jnp.atleast_1d(_as_real(B))
    B = _as_real(B)
    theta = _as_real(theta)
    phi = _as_real(phi)

    bz = B * jnp.cos(theta)
    bx = B * jnp.sin(theta) * jnp.cos(phi)
    by = B * jnp.sin(theta) * jnp.sin(phi)

    # def solve_one(bx, by, bz):
    H = hamiltonian_matrix(bx, by, bz, rg, q, A, Ax, Ay, L, alpha, beta, upsilon)
    E, U = jnp.linalg.eigh(H)
    alignment = alignment_from_eigenvectors(U, OPS["J2"])
    # return evals, U, alignment

    # E, U, alignment = jax.vmap(solve_one)(bx_vals, by_vals, bz_vals)

    # Original code used beta=0 for the reference Hamiltonian.
    Eref, _ = jnp.linalg.eigh(Href_matrix(L, alpha, 0.0))

    return E, Eref, U, alignment


def calculate_cyclicity(transition, eps=0.0):
    """Row-normalize transition rates into branching ratios.

    Args:
        transition: shape (..., n_exc, n_gnd)
    """
    total_rate = jnp.sum(transition, axis=-1, keepdims=True)
    return jnp.where(total_rate > eps, transition / total_rate, 0.0)


def _zeros_hf_like(A):
    return jnp.zeros_like(_as_complex(A))


def _get_param(name, default=None):
    return getattr(params, name, default)


def _default_hyperfine(prefix: str):
    """Best-effort defaults for old/new parameter naming conventions.

    Preferred names:
        A_gnd, Ax_gnd, Ay_gnd
        A_exc, Ax_exc, Ay_exc

    Legacy names in the pasted code:
        Aperp_gnd, Apar_gnd
        Aperp_exc, Apar_exc

    Since the pasted code calls solve_hamiltonian with too few tensors, this
    helper treats missing anisotropic tensors as zero unless explicitly present.
    """
    A = _get_param(f"A_{prefix}", None)
    Ax = _get_param(f"Ax_{prefix}", None)
    Ay = _get_param(f"Ay_{prefix}", None)

    if A is None:
        A = _get_param(f"Aperp_{prefix}", None)
    if Ax is None:
        Ax = _get_param(f"Apar_{prefix}", None)
    if Ay is None:
        if A is None:
            raise AttributeError(
                f"Could not infer hyperfine tensors for {prefix!r}. Pass "
                f"A_{prefix}, Ax_{prefix}, and Ay_{prefix} explicitly."
            )
        Ay = _zeros_hf_like(A)

    if A is None or Ax is None:
        raise AttributeError(
            f"Could not infer hyperfine tensors for {prefix!r}. Pass "
            f"A_{prefix}, Ax_{prefix}, and Ay_{prefix} explicitly."
        )

    return A, Ax, Ay


def PLE_transitions(
    B,
    theta,
    phi,
    eta,
    alpha=0.0,
    beta=0.0,
    alpha_exc=0.0,
    beta_exc=0.0,
    A_gnd=None,
    Ax_gnd=None,
    Ay_gnd=None,
    A_exc=None,
    Ax_exc=None,
    Ay_exc=None,
    rg=None,
    upsilon_gnd=0.0,
    upsilon_exc=0.0,
):
    """Calculate PLE transition intensities for a B-field sweep.

    Returns:
        E, Eref, U, alignment,
        E_exc, Eref_exc, U_exc, alignment_exc,
        transition, cyclicity
    """
    eta = _as_complex(eta)

    if A_gnd is None or Ax_gnd is None or Ay_gnd is None:
        dA, dAx, dAy = _default_hyperfine("gnd")
        A_gnd = dA if A_gnd is None else A_gnd
        Ax_gnd = dAx if Ax_gnd is None else Ax_gnd
        Ay_gnd = dAy if Ay_gnd is None else Ay_gnd

    if A_exc is None or Ax_exc is None or Ay_exc is None:
        dA, dAx, dAy = _default_hyperfine("exc")
        A_exc = dA if A_exc is None else A_exc
        Ax_exc = dAx if Ax_exc is None else Ax_exc
        Ay_exc = dAy if Ay_exc is None else Ay_exc

    E, Eref, U, alignment = solve_hamiltonian(
        B,
        theta,
        phi,
        params.q,
        A_gnd,
        Ax_gnd,
        Ay_gnd,
        params.L,
        alpha,
        beta,
        rg=rg,
        upsilon=upsilon_gnd,
    )

    E_exc, Eref_exc, U_exc, alignment_exc = solve_hamiltonian(
        B,
        theta,
        phi,
        params.q_exc,
        A_exc,
        Ax_exc,
        Ay_exc,
        params.L_exc,
        alpha_exc,
        beta_exc,
        rg=rg,
        upsilon=upsilon_exc,
    )

    p_eta = jnp.tensordot(eta, OPS["p"], axes=(0, 0))

    M = U_exc.conj().T @ p_eta @ U
    transition = jnp.real(jnp.abs(M) ** 2)
    cyclicity = calculate_cyclicity(transition)

    return (
        E,
        Eref,
        U,
        alignment,
        E_exc,
        Eref_exc,
        U_exc,
        alignment_exc,
        transition,
        cyclicity,
    )


def lorentzian(f, f0, amplitude, lw):
    half = lw / 2.0
    return amplitude * half**2 / ((f - f0) ** 2 + half**2)


def PLE_spectrum(
    f_meas,
    B,
    theta,
    phi,
    eta,
    intensity=1.0,
    lw=0.080,
    alpha=0.0,
    beta=0.0,
    alpha_exc=0.0,
    beta_exc=0.0,
    A_gnd=None,
    Ax_gnd=None,
    Ay_gnd=None,
    A_exc=None,
    Ax_exc=None,
    Ay_exc=None,
):
    """Calculate a differentiable PLE spectrum.

    Returns:
        PLE with shape (n_B, n_f).  For a scalar B, use `PLE[0]` for the old
        1D behavior.
    """
    f_meas = _as_real(f_meas)
    lw = _as_real(lw)
    intensity = _as_real(intensity)

    (
        E,
        Eref,
        _U,
        _alignment,
        E_exc,
        Eref_exc,
        _U_exc,
        _alignment_exc,
        transition,
        _cyclicity,
    ) = PLE_transitions(
        B,
        theta,
        phi,
        eta,
        alpha=alpha,
        beta=beta,
        alpha_exc=alpha_exc,
        beta_exc=beta_exc,
        A_gnd=A_gnd,
        Ax_gnd=Ax_gnd,
        Ay_gnd=Ay_gnd,
        A_exc=A_exc,
        Ax_exc=Ax_exc,
        Ay_exc=Ay_exc,
    )

    # Shape: (n_B, n_exc, n_gnd)
    f_transition = (E_exc[:, None] - Eref_exc[0]) - (E[None, :] - Eref[0])
    # Broadcast over measured frequencies.
    PLE = lorentzian(
        f_meas[None, None, :],
        f_transition[:, :, None],
        transition[:, :, None],
        lw,
    ).sum(axis=(0, 1))

    return intensity * PLE


# JIT-compiled wrappers.  Keeping the raw functions above is useful for debugging.
solve_hamiltonian_jit = jax.jit(solve_hamiltonian)
PLE_transitions_jit = jax.jit(PLE_transitions)
PLE_spectrum_jit = jax.jit(PLE_spectrum)
