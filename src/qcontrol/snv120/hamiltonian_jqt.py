"""Hamiltonian helpers for the SnV117/SnV120 center in diamond.

This module builds single-parameter-point Hamiltonians with JAX and
``jaxquantum``. The functions do not perform magnetic-field sweeps internally;
wrap the single-point functions with ``jax.vmap`` to evaluate sweeps.

The single-manifold tensor-product order is

    orbital ⊗ electron ⊗ nuclear.
"""

from __future__ import annotations

# Uncomment these lines before importing jax.numpy if float64/complex128 support
# is required by the runtime configuration.
# from jax import config
# config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxquantum as jqt

import qcontrol.snv120.parameters as params


import jax

def _spin_op(j: float, axis: str):
    """Create one angular-momentum operator.

    Parameters
    ----------
    j : float
        Spin quantum number. The operator dimension is ``2 * j + 1``.
    axis : {'x', 'y', 'z'}
        Cartesian component to construct.

    Returns
    -------
    jaxquantum.Qarray
        Dense ``Qarray`` containing the selected angular-momentum operator with
        dimensions ``(2 * j + 1,)``.

    Raises
    ------
    ValueError
        If ``axis`` is not one of ``'x'``, ``'y'``, or ``'z'``.
    """
    dim = int(round(2 * float(j) + 1))
    jj = j

    j_plus = jnp.zeros((dim, dim), dtype=jnp.complex128)
    for col in range(dim):
        m = jj - col
        if col > 0:
            coeff = jnp.sqrt(jj * (jj + 1.0) - m * (m + 1.0))
            j_plus = j_plus.at[col - 1, col].set(coeff)

    j_minus = jnp.conj(j_plus.T)

    if axis == "x":
        data = 0.5 * (j_plus + j_minus)
    elif axis == "y":
        data = (j_plus - j_minus) / (2.0j)
    elif axis == "z":
        m_vals = jj - jnp.arange(dim, dtype=jnp.float64)
        data = jnp.diag(m_vals).astype(jnp.complex128)
    else:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")

    return jqt.Qarray.create(data, dims=(dim,))


def _state_matrix(eigenstates_qarr):
    """Convert batched ket eigenstates to a dense state matrix.

    Parameters
    ----------
    eigenstates_qarr : jaxquantum.Qarray
        Batched ket ``Qarray`` returned by ``jqt.eigenstates``. Its dense data
        is expected to have shape ``(num_states, dim, 1)``.

    Returns
    -------
    jax.Array
        Dense state matrix with shape ``(num_states, dim)``. Row ``s`` contains
        eigenstate ``|psi_s>``.
    """
    return eigenstates_qarr.to_dense().data[..., :, 0]


def _eigenvector_columns(eigenstates_qarr):
    """Return eigenvectors in column-major convention.

    Parameters
    ----------
    eigenstates_qarr : jaxquantum.Qarray
        Batched ket ``Qarray`` returned by ``jqt.eigenstates``.

    Returns
    -------
    jax.Array
        Dense matrix with shape ``(dim, num_states)``. Column ``s`` contains
        eigenstate ``|psi_s>``.
    """
    states = _state_matrix(eigenstates_qarr)  # Shape: (num_states, dim).
    return jnp.swapaxes(states, -1, -2)       # Shape: (dim, num_states).


def _expect_batched(operator, eigenstates_qarr):
    """Evaluate one expectation value for each eigenstate.

    Parameters
    ----------
    operator : jaxquantum.Qarray
        Operator with dense shape ``(dim, dim)``.
    eigenstates_qarr : jaxquantum.Qarray
        Batched ket eigenstates with dense shape ``(num_states, dim, 1)``.

    Returns
    -------
    jax.Array
        Real expectation values ``<psi_s|operator|psi_s>`` with shape
        ``(num_states,)``.
    """
    states = _state_matrix(eigenstates_qarr)       # Shape: (num_states, dim).
    op = operator.to_dense().data                  # Shape: (dim, dim).
    return jnp.real(jnp.einsum("si,ij,sj->s", jnp.conj(states), op, states))


def create_hamiltonian_nuclear():
    """Create Hamiltonian, reference Hamiltonian, dipole, and ``J2`` builders.

    The returned callables operate on one parameter point at a time in the
    single-manifold Hilbert space ``orbital ⊗ electron ⊗ nuclear``.

    Returns
    -------
    H : callable
        Function that returns the full single-manifold Hamiltonian for one set
        of magnetic-field, hyperfine, spin-orbit, strain, and iso-orbital
        parameters.
    Href : callable
        Function that returns the reference Hamiltonian containing only
        spin-orbit coupling and strain/Jahn-Teller terms.
    p : list of jaxquantum.Qarray
        Dipole-moment operators ``[p_x, p_y, p_z]`` in the full
        single-manifold Hilbert space.
    J2 : jaxquantum.Qarray
        Total electron-plus-nuclear angular-momentum-squared operator.
    """
    # Electron and nuclear spin quantum numbers.
    S = params.S
    Sn = params.Sn

    X = _spin_op(S, "x")
    Y = _spin_op(S, "y")
    Z = _spin_op(S, "z")
    I = jqt.identity(int(2 * S + 1))

    Xn = _spin_op(Sn, "x")
    Yn = _spin_op(Sn, "y")
    Zn = _spin_op(Sn, "z")
    In = jqt.identity(int(2 * Sn + 1))

    # Total angular momentum squared, J^2 = S^2 + I^2 + 2 S·I.
    J2 = (
        (S * (S + 1) + Sn * (Sn + 1)) * jqt.tensor(I, I, In)
        + 2.0
        * (
            jqt.tensor(I, X, Xn)
            + jqt.tensor(I, Y, Yn)
            + jqt.tensor(I, Z, Zn)
        )
    )

    # Electron Zeeman terms.
    Hbxe = lambda bx: bx * jqt.tensor(I, X, In)
    Hbye = lambda by: by * jqt.tensor(I, Y, In)
    Hbze = lambda bz: bz * jqt.tensor(I, Z, In)

    # Nuclear Zeeman terms, scaled by the gyromagnetic-ratio ratio rg.
    Hbxn = lambda bx, rg: rg * bx * jqt.tensor(I, I, Xn)
    Hbyn = lambda by, rg: rg * by * jqt.tensor(I, I, Yn)
    Hbzn = lambda bz, rg: rg * bz * jqt.tensor(I, I, Zn)

    # Orbital magnetic-field term along the local z axis.
    Hbzo = lambda bz, q: (q / 2.0) * bz * jqt.tensor(Z, I, In)

    S_ops = (X, Y, Z)
    N_ops = (Xn, Yn, Zn)

    # Orbital matrices used for the Ax and Ay hyperfine-modulation tensors in
    # the e+/e- basis.
    sigma_z_xy_in_pm = -2.0 * X
    sigma_x_xy_in_pm = -2.0 * Y

    def hf_block(B_tensor, orbital_op):
        """Build one hyperfine block from a 3-by-3 coupling tensor.

        Parameters
        ----------
        B_tensor : array_like, shape (3, 3)
            Hyperfine tensor coupling electron spin component ``i`` to nuclear
            spin component ``j``.
        orbital_op : jaxquantum.Qarray
            Orbital prefactor for this hyperfine block.

        Returns
        -------
        jaxquantum.Qarray
            Hyperfine contribution ``sum_ij B_tensor[i, j] O ⊗ S_i ⊗ I_j``.
        """
        H_block = 0.0 * jqt.tensor(orbital_op, S_ops[0], N_ops[0])
        for i in range(3):
            for j in range(3):
                H_block = H_block + B_tensor[i, j] * jqt.tensor(
                    orbital_op, S_ops[i], N_ops[j]
                )
        return H_block

    def Hhf(A, Ax, Ay):
        """Build the full hyperfine contribution.

        Parameters
        ----------
        A, Ax, Ay : array_like, shape (3, 3)
            Hyperfine tensors for the orbital-independent and orbital-modulated
            terms.

        Returns
        -------
        jaxquantum.Qarray
            Hyperfine Hamiltonian in the single-manifold Hilbert space.
        """
        return (
            hf_block(A, I)
            + hf_block(Ax, sigma_z_xy_in_pm)
            + hf_block(Ay, sigma_x_xy_in_pm)
        )

    # Spin-orbit coupling.
    Hsoc = lambda L: 2.0 * L * jqt.tensor(Z, Z, In)

    # Iso-orbital coupling, also called upsilon.
    Hioc = lambda u: 2.0 * u * jqt.tensor(Z, I, Zn)

    # Strain/Jahn-Teller terms.
    Hegx = lambda alpha: -2.0 * alpha * jqt.tensor(X, I, In)
    Hegy = lambda beta: 2.0 * beta * jqt.tensor(Y, I, In)

    # Dipole moment operators [p_x, p_y, p_z] in the e+/e- basis.
    p_orbital = [
        2.0 * X,
        2.0 * Y,
        2.0 * I,
    ]
    p = [jqt.tensor(p_op, I, In) for p_op in p_orbital]

    Href = lambda L, alpha, beta: Hsoc(L) + Hegx(alpha) + Hegy(beta)

    def H(bx, by, bz, rg, q, A, Ax, Ay, L, alpha, beta, upsilon=0.0):
        """Evaluate the full single-manifold Hamiltonian.

        Parameters
        ----------
        bx, by, bz : scalar
            Magnetic-field components in the same frequency-like units used by
            the Hamiltonian terms.
        rg : scalar
            Nuclear-to-electron gyromagnetic-ratio scaling factor.
        q : scalar
            Orbital magnetic-field susceptibility.
        A, Ax, Ay : array_like, shape (3, 3)
            Hyperfine tensors.
        L : scalar
            Spin-orbit coupling strength.
        alpha, beta : scalar
            Strain/Jahn-Teller parameters.
        upsilon : scalar, optional
            Iso-orbital coupling strength.

        Returns
        -------
        jaxquantum.Qarray
            Total Hamiltonian for one parameter point.
        """
        return (
            Href(L, alpha, beta)
            + (Hbxe(bx) + Hbxn(bx, rg))
            + (Hbye(by) + Hbyn(by, rg))
            + (Hbze(bz) + Hbzn(bz, rg) + Hbzo(bz, q))
            + Hhf(A, Ax, Ay)
            + Hioc(upsilon)
        )

    return H, Href, p, J2


def create_B_hamiltonian():
    """Create a magnetic-field-only Hamiltonian builder.

    Returns
    -------
    Hb : callable
        Function ``Hb(bx, by, bz, rg, q)`` that returns the sum of electron,
        nuclear, and orbital magnetic-field terms in the single-manifold
        Hilbert space.
    """
    # Electron and nuclear spin quantum numbers.
    S = params.S
    Sn = params.Sn

    X = _spin_op(S, "x")
    Y = _spin_op(S, "y")
    Z = _spin_op(S, "z")
    I = jqt.identity(int(2 * S + 1))

    Xn = _spin_op(Sn, "x")
    Yn = _spin_op(Sn, "y")
    Zn = _spin_op(Sn, "z")
    In = jqt.identity(int(2 * Sn + 1))

    # Electron Zeeman terms.
    Hbxe = lambda bx: bx * jqt.tensor(I, X, In)
    Hbye = lambda by: by * jqt.tensor(I, Y, In)
    Hbze = lambda bz: bz * jqt.tensor(I, Z, In)

    # Nuclear Zeeman terms, scaled by rg.
    Hbxn = lambda bx, rg: rg * bx * jqt.tensor(I, I, Xn)
    Hbyn = lambda by, rg: rg * by * jqt.tensor(I, I, Yn)
    Hbzn = lambda bz, rg: rg * bz * jqt.tensor(I, I, Zn)

    # Orbital magnetic-field term along the local z axis.
    Hbzo = lambda bz, q: (q / 2.0) * bz * jqt.tensor(Z, I, In)

    def Hb(bx, by, bz, rg, q):
        """Evaluate magnetic-field terms for one field vector.

        Parameters
        ----------
        bx, by, bz : scalar
            Magnetic-field components in Hamiltonian units.
        rg : scalar
            Nuclear-to-electron gyromagnetic-ratio scaling factor.
        q : scalar
            Orbital magnetic-field susceptibility.

        Returns
        -------
        jaxquantum.Qarray
            Electron, nuclear, and orbital magnetic-field Hamiltonian terms.
        """
        return (
            Hbxe(bx) + Hbxn(bx, rg)
            + Hbye(by) + Hbyn(by, rg)
            + Hbze(bz) + Hbzn(bz, rg) + Hbzo(bz, q)
        )

    return Hb


def solve_hamiltonian(B, theta, phi, q, A, Ax, Ay, L, alpha, beta):
    """Diagonalize the single-manifold Hamiltonian at one field point.

    Parameters
    ----------
    B : scalar
        Magnetic-field magnitude in Hamiltonian units.
    theta : scalar
        Polar angle of the magnetic field in radians.
    phi : scalar
        Azimuthal angle of the magnetic field in radians.
    q : scalar
        Orbital magnetic-field susceptibility.
    A, Ax, Ay : array_like, shape (3, 3)
        Hyperfine tensors.
    L : scalar
        Spin-orbit coupling strength.
    alpha, beta : scalar
        Strain/Jahn-Teller parameters.

    Returns
    -------
    E : jax.Array, shape (num_states,)
        Eigenvalues of the full Hamiltonian.
    Eref : jax.Array, shape (num_states,)
        Eigenvalues of ``Href(L, alpha, 0.0)``.
    U : jax.Array, shape (dim, num_states)
        Eigenvectors arranged as columns.
    U_states : jaxquantum.Qarray
        Batched ket eigenstates returned by ``jqt.eigenstates``.
    alignment : jax.Array, shape (num_states,)
        Expectation values of ``J2`` for the eigenstates.
    """
    H, Href, _, J2 = create_hamiltonian_nuclear()

    bz = B * jnp.cos(theta)
    bx = B * jnp.sin(theta) * jnp.cos(phi)
    by = B * jnp.sin(theta) * jnp.sin(phi)

    H_qarr = H(bx, by, bz, params.rg, q, A, Ax, Ay, L, alpha, beta)
    E, U_states = jqt.eigenstates(H_qarr)
    U = _eigenvector_columns(U_states)
    alignment = _expect_batched(J2, U_states)

    # Preserve the original reference convention: beta is fixed to zero for
    # the reference Hamiltonian even when beta is nonzero in the full model.
    Href_qarr = Href(L, alpha, 0.0)
    Eref, _ = jqt.eigenstates(Href_qarr)

    return E, Eref, U, U_states, alignment


def calculate_cyclicity(transition):
    """Normalize transition intensities into branching ratios.

    Parameters
    ----------
    transition : array_like, shape (num_exc_states, num_gnd_states)
        Transition intensities, where ``transition[l, k]`` is proportional to
        ``|<exc_l|p · eta|gnd_k>|**2``.

    Returns
    -------
    jax.Array, shape (num_exc_states, num_gnd_states)
        Row-normalized branching ratios. Rows with zero total transition rate
        are returned as zeros.
    """
    total_rate = jnp.sum(transition, axis=1, keepdims=True)
    safe_total = jnp.where(total_rate > 0.0, total_rate, 1.0)
    return jnp.where(total_rate > 0.0, transition / safe_total, 0.0)


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
):
    """Calculate PLE transition intensities at one parameter point.

    Parameters
    ----------
    B : scalar
        Magnetic-field magnitude in Hamiltonian units.
    theta : scalar
        Polar angle of the magnetic field in radians.
    phi : scalar
        Azimuthal angle of the magnetic field in radians.
    eta : array_like, shape (3,)
        Optical polarization coefficients multiplying ``[p_x, p_y, p_z]``.
    alpha, beta : scalar, optional
        Ground-state strain/Jahn-Teller parameters.
    alpha_exc, beta_exc : scalar, optional
        Excited-state strain/Jahn-Teller parameters.
    A_gnd, Ax_gnd, Ay_gnd : array_like, shape (3, 3), optional
        Ground-state hyperfine tensors. Missing tensors are replaced with
        zeros.
    A_exc, Ax_exc, Ay_exc : array_like, shape (3, 3), optional
        Excited-state hyperfine tensors. Missing tensors are replaced with
        zeros.

    Returns
    -------
    E : jax.Array, shape (num_gnd_states,)
        Ground-state eigenvalues.
    Eref : jax.Array, shape (num_gnd_states,)
        Ground-state reference eigenvalues.
    U : jax.Array, shape (dim, num_gnd_states)
        Ground-state eigenvectors arranged as columns.
    alignment : jax.Array, shape (num_gnd_states,)
        Ground-state ``J2`` expectation values.
    E_exc : jax.Array, shape (num_exc_states,)
        Excited-state eigenvalues.
    Eref_exc : jax.Array, shape (num_exc_states,)
        Excited-state reference eigenvalues.
    U_exc : jax.Array, shape (dim, num_exc_states)
        Excited-state eigenvectors arranged as columns.
    alignment_exc : jax.Array, shape (num_exc_states,)
        Excited-state ``J2`` expectation values.
    transition : jax.Array, shape (num_exc_states, num_gnd_states)
        Squared dipole matrix elements ``|<exc_l|p · eta|gnd_k>|**2``.
    cyclicity : jax.Array, shape (num_exc_states, num_gnd_states)
        Row-normalized transition branching ratios.
    """
    H, Href, p, J2 = create_hamiltonian_nuclear()
    del H, Href, J2  # Only p is used locally; solve_hamiltonian builds H/Href/J2.

    zero3 = jnp.zeros((3, 3))
    if A_gnd is None:
        A_gnd = zero3
    if Ax_gnd is None:
        Ax_gnd = zero3
    if Ay_gnd is None:
        Ay_gnd = zero3
    if A_exc is None:
        A_exc = zero3
    if Ax_exc is None:
        Ax_exc = zero3
    if Ay_exc is None:
        Ay_exc = zero3

    E, Eref, U, U_states, alignment = solve_hamiltonian(
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
    )

    E_exc, Eref_exc, U_exc, U_exc_states, alignment_exc = solve_hamiltonian(
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
    )

    dipole = 0.0 * p[0]
    for j in range(3):
        dipole = dipole + eta[j] * p[j]

    gnd_states = _state_matrix(U_states)          # Shape: (num_gnd_states, dim).
    exc_states = _state_matrix(U_exc_states)      # Shape: (num_exc_states, dim).
    dipole_data = dipole.to_dense().data          # Shape: (dim, dim).

    # Matrix elements M[l, k] = <exc_l|dipole|gnd_k>.
    matrix_elements = jnp.einsum(
        "li,ij,kj->lk", jnp.conj(exc_states), dipole_data, gnd_states
    )
    transition = jnp.abs(matrix_elements) ** 2
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
    """Calculate a Lorentzian-broadened PLE spectrum.

    Parameters
    ----------
    f_meas : array_like, shape (num_frequencies,)
        Frequency grid at which to evaluate the spectrum.
    B : scalar
        Magnetic-field magnitude in Hamiltonian units.
    theta : scalar
        Polar angle of the magnetic field in radians.
    phi : scalar
        Azimuthal angle of the magnetic field in radians.
    eta : array_like, shape (3,)
        Optical polarization coefficients multiplying ``[p_x, p_y, p_z]``.
    intensity : scalar, optional
        Overall multiplicative spectrum scale.
    lw : scalar, optional
        Full linewidth used in the Lorentzian denominator.
    alpha, beta : scalar, optional
        Ground-state strain/Jahn-Teller parameters.
    alpha_exc, beta_exc : scalar, optional
        Excited-state strain/Jahn-Teller parameters.
    A_gnd, Ax_gnd, Ay_gnd : array_like, shape (3, 3), optional
        Ground-state hyperfine tensors. Missing tensors are replaced with
        zeros.
    A_exc, Ax_exc, Ay_exc : array_like, shape (3, 3), optional
        Excited-state hyperfine tensors. Missing tensors are replaced with
        zeros.

    Returns
    -------
    jax.Array, shape (num_frequencies,)
        Sum of Lorentzian peaks over all excited-to-ground transitions.
    """
    (
        E,
        Eref,
        _,
        _,
        E_exc,
        Eref_exc,
        _,
        _,
        transition,
        _,
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

    f_transition = (E_exc[:, None] - Eref_exc[0]) - (E[None, :] - Eref[0])

    half_lw_sq = (lw / 2.0) ** 2
    peaks = (
        transition[:, :, None]
        * half_lw_sq
        / ((f_meas[None, None, :] - f_transition[:, :, None]) ** 2 + half_lw_sq)
    )

    return intensity * jnp.sum(peaks, axis=(0, 1))

@jax.jit(static_argnames=["included_states"])
def get_dynamic_hamiltonian(
    B,
    theta,
    phi,
    excited_ground_split,
    excited_state_lifetime,
    resonant_pump_polarization,
    B_drive_orientation,
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
    upsilon_gnd=0.0,
    upsilon_exc=0.0,
    included_states=None,
):
    """Build expanded static, drive, optical, and collapse operators.

    The expanded Hilbert-space order is

        manifold ⊗ orbital ⊗ electron ⊗ nuclear,

    where manifold index 0 is the ground-state manifold and manifold index 1 is
    the excited-state manifold. The excited block is shifted by
    ``excited_ground_split``.

    Parameters
    ----------
    included_states : None or tuple of int, optional
        If ``None``, keep the full expanded basis.

        If a tuple of ints, for example ``(0, 1, 2)``, keep only the matching
        ground/excited eigenstate pairs

            (|g_0>, |e_0>), (|g_1>, |e_1>), (|g_2>, |e_2>).

        This argument is compile-time structure: when using ``jax.jit``, pass it
        as a static argument.

    Returns
    -------
    H0 : jaxquantum.Qarray
        Static expanded block Hamiltonian.
    Hb : jaxquantum.Qarray
        Fixed expanded microwave-drive Hamiltonian.
    H_optical : jaxquantum.Qarray
        Fixed off-diagonal optical/electric-dipole coupling operator.
    c_ops : jaxquantum.Qarray
        Batched collapse operators for spontaneous decay.
    """

    H_single, _, p, _ = create_hamiltonian_nuclear()
    H_b = create_B_hamiltonian()

    zero3 = jnp.zeros((3, 3))
    if A_gnd is None:
        A_gnd = zero3
    if Ax_gnd is None:
        Ax_gnd = zero3
    if Ay_gnd is None:
        Ay_gnd = zero3
    if A_exc is None:
        A_exc = zero3
    if Ax_exc is None:
        Ax_exc = zero3
    if Ay_exc is None:
        Ay_exc = zero3

    bz0 = B * jnp.cos(theta)
    bx0 = B * jnp.sin(theta) * jnp.cos(phi)
    by0 = B * jnp.sin(theta) * jnp.sin(phi)

    orbital_dim = int(2 * params.S + 1)
    electron_dim = int(2 * params.S + 1)
    nuclear_dim = int(2 * params.Sn + 1)
    base_dims = (orbital_dim, electron_dim, nuclear_dim)
    expanded_dims = (2,) + base_dims
    base_dim = orbital_dim * electron_dim * nuclear_dim

    def _as_qarray(data, dims=expanded_dims):
        return jqt.Qarray.create(data, dims=dims)

    def _block_data_qarray(H_gnd_data, H_exc_data, dims, exc_shift=0.0, rwa_gnd=False):
        ground_dim = int(H_gnd_data.shape[0])
        excited_dim = int(H_exc_data.shape[0])
        dtype = jnp.result_type(H_gnd_data, H_exc_data, exc_shift)

        zero_ge = jnp.zeros((ground_dim, excited_dim), dtype=dtype)
        zero_eg = jnp.zeros((excited_dim, ground_dim), dtype=dtype)
        eye_exc = jnp.eye(excited_dim, dtype=dtype)

        H_gnd_data = H_gnd_data.astype(dtype)
        H_exc_data = H_exc_data.astype(dtype) + exc_shift * eye_exc
        if rwa_gnd:
            avg_gnd_energy = jnp.mean(jnp.diag(H_gnd_data))
            H_gnd_data = H_gnd_data - avg_gnd_energy * jnp.eye(ground_dim, dtype=dtype)
            H_exc_data = H_exc_data - avg_gnd_energy * eye_exc

        top = jnp.concatenate([H_gnd_data, zero_ge], axis=1)
        bottom = jnp.concatenate([zero_eg, H_exc_data], axis=1)
        return _as_qarray(jnp.concatenate([top, bottom], axis=0), dims=dims)

    def _offdiag_data_qarray(lower_left, upper_right, dims):
        """Build [[0, upper_right], [lower_left, 0]] from dense blocks."""
        excited_dim = int(lower_left.shape[0])
        ground_dim = int(upper_right.shape[0])
        dtype = jnp.result_type(lower_left, upper_right)

        zero_ground = jnp.zeros((ground_dim, ground_dim), dtype=dtype)
        zero_excited = jnp.zeros((excited_dim, excited_dim), dtype=dtype)

        top = jnp.concatenate([zero_ground, upper_right.astype(dtype)], axis=1)
        bottom = jnp.concatenate([lower_left.astype(dtype), zero_excited], axis=1)
        return _as_qarray(jnp.concatenate([top, bottom], axis=0), dims=dims)

    def _block_qarray(H_gnd, H_exc, exc_shift=0.0):
        return _block_data_qarray(
            H_gnd.to_dense().data,
            H_exc.to_dense().data,
            expanded_dims,
            exc_shift=exc_shift,
        )

    def _offdiag_qarray(lower_left, upper_right):
        return _offdiag_data_qarray(lower_left, upper_right, expanded_dims)

    def _state_pair_indices(states_to_include, num_gnd_states, num_exc_states):
        """Normalize None or tuple[int, ...] into matching pair indices.

        ``(0, 1, 2)`` means ground indices ``(0, 1, 2)`` and excited indices
        ``(0, 1, 2)``.
        """
        if states_to_include is None:
            return None

        if not isinstance(states_to_include, tuple):
            raise TypeError(
                "included_states must be None or a tuple of ints, "
                "for example included_states=(0, 1, 2)."
            )

        if len(states_to_include) == 0:
            raise ValueError(
                "included_states must be None or a non-empty tuple of ints."
            )

        indices = []
        for value in states_to_include:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    "included_states must contain only Python int values, "
                    "for example included_states=(0, 1, 2)."
                )
            indices.append(int(value))

        if len(set(indices)) != len(indices):
            raise ValueError("included_states contains duplicate pair indices.")

        bad_gnd = [idx for idx in indices if idx < 0 or idx >= num_gnd_states]
        bad_exc = [idx for idx in indices if idx < 0 or idx >= num_exc_states]

        if bad_gnd:
            raise ValueError(
                f"Ground-state pair indices out of range [0, {num_gnd_states}): "
                f"{bad_gnd}."
            )
        if bad_exc:
            raise ValueError(
                f"Excited-state pair indices out of range [0, {num_exc_states}): "
                f"{bad_exc}."
            )

        return tuple(indices)

    def _project_operator(operator_data, row_states, col_states):
        return jnp.einsum(
            "ai,ij,bj->ab",
            jnp.conj(row_states),
            operator_data,
            col_states,
        )

    H_gnd_static = H_single(
        bx0,
        by0,
        bz0,
        params.rg,
        params.q,
        A_gnd,
        Ax_gnd,
        Ay_gnd,
        params.L,
        alpha,
        beta,
        upsilon_gnd,
    )

    H_exc_static = H_single(
        bx0,
        by0,
        bz0,
        params.rg,
        params.q_exc,
        A_exc,
        Ax_exc,
        Ay_exc,
        params.L_exc,
        alpha_exc,
        beta_exc,
        upsilon_exc,
    )

    bx_drive = (
        B
        * jnp.sin(B_drive_orientation[0])
        * jnp.cos(B_drive_orientation[1])
    )
    by_drive = (
        B
        * jnp.sin(B_drive_orientation[0])
        * jnp.sin(B_drive_orientation[1])
    )
    bz_drive = B * jnp.cos(B_drive_orientation[0])

    H_gnd_drive = H_b(
        bx_drive,
        by_drive,
        bz_drive,
        params.rg,
        params.q,
    )
    H_exc_drive = H_b(
        bx_drive,
        by_drive,
        bz_drive,
        params.rg,
        params.q_exc,
    )

    eta = jnp.asarray(
        [
            jnp.sin(resonant_pump_polarization[0])
            * jnp.cos(resonant_pump_polarization[1]),
            jnp.sin(resonant_pump_polarization[0])
            * jnp.sin(resonant_pump_polarization[1]),
            jnp.cos(resonant_pump_polarization[0]),
        ]
    )

    p_eta = 0.0 * p[0]
    for j in range(3):
        p_eta = p_eta + eta[j] * p[j]

    p_eta_data = p_eta.to_dense().data
    p_eta_dag_data = jnp.conj(p_eta_data.T)

    _, U_gnd_decay_states = jqt.eigenstates(H_gnd_static)
    _, U_exc_decay_states = jqt.eigenstates(H_exc_static)

    gnd_states = _state_matrix(U_gnd_decay_states)
    exc_states = _state_matrix(U_exc_decay_states)

    spontaneous_rates = jnp.zeros(
        (int(exc_states.shape[0]), int(gnd_states.shape[0]))
    )
    for p_op in p:
        spontaneous_rates = spontaneous_rates + jnp.abs(
            jnp.einsum(
                "li,ij,kj->lk",
                jnp.conj(exc_states),
                p_op.to_dense().data,
                gnd_states,
            )
        ) ** 2

    cyclicity = calculate_cyclicity(spontaneous_rates)

    pair_indices = _state_pair_indices(
        included_states,
        int(gnd_states.shape[0]),
        int(exc_states.shape[0]),
    )

    if pair_indices is None:
        H0 = _block_qarray(
            H_gnd_static,
            H_exc_static,
            exc_shift=excited_ground_split,
        )
        Hb = _block_qarray(
            H_gnd_drive,
            H_exc_drive,
            exc_shift=0.0,
        )
        Hs_optical = [_offdiag_qarray(zero_base, p_eta_dag_data), _offdiag_qarray(p_eta_data, zero_base)]

        total_decay_rate = 1.0 / excited_state_lifetime
        c_ops_list = []
        zero_base = jnp.zeros((base_dim, base_dim))

        for l in range(int(exc_states.shape[0])):
            for k in range(int(gnd_states.shape[0])):
                jump_base = jnp.outer(gnd_states[k], jnp.conj(exc_states[l]))
                jump_rate = total_decay_rate * cyclicity[l, k]
                jump_ge = jnp.sqrt(jump_rate) * jump_base
                c_ops_list.append(_offdiag_qarray(zero_base, jump_ge))

    else:
        pair_indices_arr = jnp.asarray(pair_indices, dtype=jnp.int32)

        kept_gnd_states = gnd_states[pair_indices_arr]
        kept_exc_states = exc_states[pair_indices_arr]

        reduced_dim = int(len(pair_indices))
        reduced_dims = (2, reduced_dim)
        zero_reduced = jnp.zeros((reduced_dim, reduced_dim))

        H_gnd_static_red = _project_operator(
            H_gnd_static.to_dense().data,
            kept_gnd_states,
            kept_gnd_states,
        )
        H_exc_static_red = _project_operator(
            H_exc_static.to_dense().data,
            kept_exc_states,
            kept_exc_states,
        )
        H0 = _block_data_qarray(
            H_gnd_static_red,
            H_exc_static_red,
            reduced_dims,
            exc_shift=excited_ground_split,
            rwa_gnd=True,
        )

        H_gnd_drive_red = _project_operator(
            H_gnd_drive.to_dense().data,
            kept_gnd_states,
            kept_gnd_states,
        )
        H_exc_drive_red = _project_operator(
            H_exc_drive.to_dense().data,
            kept_exc_states,
            kept_exc_states,
        )
        Hb = _block_data_qarray(
            H_gnd_drive_red,
            H_exc_drive_red,
            reduced_dims,
            exc_shift=0.0,
            rwa_gnd=False,
        )

        p_ge_red = _project_operator(
            p_eta_data,
            kept_exc_states,
            kept_gnd_states,
        )
        p_eg_red = _project_operator(
            p_eta_dag_data,
            kept_gnd_states,
            kept_exc_states,
        )
        Hs_optical = [_offdiag_data_qarray(zero_reduced, p_eg_red, reduced_dims), _offdiag_data_qarray(p_ge_red, zero_reduced, reduced_dims)]

        cyclicity_red = cyclicity[
            pair_indices_arr[:, None],
            pair_indices_arr[None, :],
        ]

        total_decay_rate = 1.0 / excited_state_lifetime
        c_ops_list = []

        for l in range(reduced_dim):
            for k in range(reduced_dim):
                jump_rate = total_decay_rate * cyclicity_red[l, k]
                jump_ge = zero_reduced.at[k, l].set(jnp.sqrt(jump_rate))
                c_ops_list.append(
                    _offdiag_data_qarray(
                        zero_reduced,
                        jump_ge,
                        reduced_dims,
                    )
                )

    c_ops = jqt.Qarray.from_list(c_ops_list)

    return H0, Hb, Hs_optical, c_ops


@jax.jit(static_argnames=["included_states"])
def calculate_spontaneous_cyclicity(
    B,
    theta,
    phi,
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
    upsilon_gnd=0.0,
    upsilon_exc=0.0,
    included_states=None,
):
    """Calculate spontaneous-emission branching ratios.

    This is the spontaneous-decay analog of ``PLE_transitions``. Unlike PLE,
    this does not use a laser polarization ``eta``. Instead, it sums the
    decay strength over the three dipole operators ``p_x, p_y, p_z``:

        Gamma[l, k] ∝ sum_i |<exc_l|p_i|gnd_k>|^2

    and then normalizes each excited-state row so that

        sum_k cyclicity[l, k] = 1

    for every excited state with nonzero total dipole strength.

    Parameters
    ----------
    B : scalar
        Magnetic-field magnitude in Hamiltonian units.
    theta : scalar
        Polar angle of the magnetic field in radians.
    phi : scalar
        Azimuthal angle of the magnetic field in radians.
    alpha, beta : scalar, optional
        Ground-state strain/Jahn-Teller parameters.
    alpha_exc, beta_exc : scalar, optional
        Excited-state strain/Jahn-Teller parameters.
    A_gnd, Ax_gnd, Ay_gnd : array_like, shape (3, 3), optional
        Ground-state hyperfine tensors. Missing tensors are replaced with zeros.
    A_exc, Ax_exc, Ay_exc : array_like, shape (3, 3), optional
        Excited-state hyperfine tensors. Missing tensors are replaced with zeros.
    upsilon_gnd, upsilon_exc : scalar, optional
        Ground/excited iso-orbital coupling strengths.
    included_states : None or tuple of int, optional
        If ``None``, return the full excited-by-ground cyclicity matrix.

        If a tuple, for example ``(0, 1, 2)``, return only the submatrix
        connecting those matched excited and ground eigenstate indices.

    Returns
    -------
    spontaneous_rates : jax.Array
        Unnormalized spontaneous-emission strengths. Shape is
        ``(num_exc_states, num_gnd_states)`` if ``included_states is None``,
        otherwise ``(len(included_states), len(included_states))``.
    cyclicity : jax.Array
        Row-normalized branching ratios with the same shape as
        ``spontaneous_rates``.
    """
    H_single, _, p, _ = create_hamiltonian_nuclear()

    zero3 = jnp.zeros((3, 3))
    if A_gnd is None:
        A_gnd = zero3
    if Ax_gnd is None:
        Ax_gnd = zero3
    if Ay_gnd is None:
        Ay_gnd = zero3
    if A_exc is None:
        A_exc = zero3
    if Ax_exc is None:
        Ax_exc = zero3
    if Ay_exc is None:
        Ay_exc = zero3

    bz0 = B * jnp.cos(theta)
    bx0 = B * jnp.sin(theta) * jnp.cos(phi)
    by0 = B * jnp.sin(theta) * jnp.sin(phi)

    H_gnd_static = H_single(
        bx0,
        by0,
        bz0,
        params.rg,
        params.q,
        A_gnd,
        Ax_gnd,
        Ay_gnd,
        params.L,
        alpha,
        beta,
        upsilon_gnd,
    )

    H_exc_static = H_single(
        bx0,
        by0,
        bz0,
        params.rg,
        params.q_exc,
        A_exc,
        Ax_exc,
        Ay_exc,
        params.L_exc,
        alpha_exc,
        beta_exc,
        upsilon_exc,
    )

    _, U_gnd_states = jqt.eigenstates(H_gnd_static)
    _, U_exc_states = jqt.eigenstates(H_exc_static)

    gnd_states = _state_matrix(U_gnd_states)      # (num_gnd_states, dim)
    exc_states = _state_matrix(U_exc_states)      # (num_exc_states, dim)

    spontaneous_rates = jnp.zeros(
        (int(exc_states.shape[0]), int(gnd_states.shape[0]))
    )

    for p_op in p:
        p_data = p_op.to_dense().data
        matrix_elements = jnp.einsum(
            "li,ij,kj->lk",
            jnp.conj(exc_states),
            p_data,
            gnd_states,
        )
        spontaneous_rates = spontaneous_rates + jnp.abs(matrix_elements) ** 2

    cyclicity = calculate_cyclicity(spontaneous_rates)

    if included_states is not None:
        pair_indices = jnp.asarray(included_states, dtype=jnp.int32)
        spontaneous_rates = spontaneous_rates[
            pair_indices[:, None],
            pair_indices[None, :],
        ]

        # Renormalize after truncation so each kept excited state decays with
        # total probability 1 inside the reduced model.
        cyclicity = calculate_cyclicity(spontaneous_rates)

    return cyclicity