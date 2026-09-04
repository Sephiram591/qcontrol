"""QuTiP implementation of the SnV117 dynamic-Jahn--Teller model.

The model includes spin-orbit, iso-orbital, strain, electron Zeeman, nuclear
Zeeman, orbital Zeeman, asymmetric-Ham, and DJT hyperfine interactions.

Notes
-----
The single-manifold tensor-product order is
``orbital x electron x nuclear``. All coupling parameters are expressed in
GHz, while magnetic-field components use the original frequency-like code
units.
"""

import numpy as np
import qutip as qt
import parameters_DJT as params


def create_hamiltonian_nuclear(manifold='ground'):
    """Create the SnV117 single-manifold Hamiltonian builders.
    
    Parameters
    ----------
    manifold : {'ground', 'excited'}, optional
        Electronic manifold whose physical parameters are used as defaults.
    
    Returns
    -------
    H : callable
        Full Hamiltonian builder. ``H(bx, by, bz, ...)`` returns a QuTiP
        operator in the single-manifold Hilbert space.
    Href : callable
        Reference Hamiltonian builder containing spin-orbit coupling and
        strain only.
    p : list of qutip.Qobj
        Dipole operators ``[p_x, p_y, p_z]``.
    J2 : qutip.Qobj
        Total electron-plus-nuclear angular-momentum-squared operator.
    
    Raises
    ------
    ValueError
        If ``manifold`` is neither ``'ground'`` nor ``'excited'``.
    
    Notes
    -----
    The tensor-product order is ``orbital x electron x nuclear``. The orbital
    and electron spaces are both two-dimensional for the SnV117 model.
    """
    # Resolve manifold-based defaults
    if manifold == 'ground':
        _rg = params.rg
        _q = params.q
        _L = params.L
        _Aperp = params.Aperp_gnd
        _Apar = params.Apar_gnd
        _A1 = params.A1_gnd
        _A2 = params.A2_gnd
        _delta_f = params.delta_f_gnd
    elif manifold == 'excited':
        _rg = params.rg
        _q = params.q_exc
        _L = params.L_exc
        _Aperp = params.Aperp_exc
        _Apar = params.Apar_exc
        _A1 = params.A1_exc
        _A2 = params.A2_exc
        _delta_f = params.delta_f_exc
    else:
        raise ValueError(f"manifold must be 'ground' or 'excited', got '{manifold}'")

    # Electron spin (S = 1/2)
    S = params.S
    Sn = params.Sn
    X = qt.jmat(S, 'x')
    Y = qt.jmat(S, 'y')
    Z = qt.jmat(S, 'z')
    I = qt.qeye(int(2*S + 1))  # Identity matrix (2x2, used for both orbital and electron spaces)
    
    # Nuclear spin operators
    Xn = qt.jmat(Sn, 'x')
    Yn = qt.jmat(Sn, 'y')
    Zn = qt.jmat(Sn, 'z')
    In = qt.qeye(int(2*Sn + 1))
    
    # Raising/lowering operators for electron spin
    Sp = qt.jmat(S, '+')   # S+
    Sm = qt.jmat(S, '-')   # S-
    
    # Raising/lowering operators for nuclear spin
    Inp = qt.jmat(Sn, '+')  # I+
    Inm = qt.jmat(Sn, '-')  # I-
    
    # Orbital raising/lowering operators: σ± = |e±⟩⟨e∓|
    sigma_p = qt.jmat(S, '+')  # σ+ in orbital space
    sigma_m = qt.jmat(S, '-')  # σ- in orbital space
    
    # Total angular momentum squared operator
    # J^2 = (S_e*(S_e+1) + S_n*(S_n+1))*I_orb ⊗ I_e ⊗ I_n + 2*I_orb ⊗ (S_e ⊗ S_n)
    # Structure: orbital ⊗ electron ⊗ nuclear
    J2 = ((S*(S + 1) + Sn*(Sn + 1)) * qt.tensor(I, I, In) + 
          2 * (qt.tensor(I, X, Xn) + qt.tensor(I, Y, Yn) + qt.tensor(I, Z, Zn)))
    
    # Magnetic field on electron
    # Structure: orbital ⊗ electron ⊗ nuclear
    # Units: magnetic field is in units of g_e * mu_B (electron gyromagnetic ratio)
    Hbxe = lambda bx: bx * qt.tensor(I, X, In)
    Hbye = lambda by: by * qt.tensor(I, Y, In)
    Hbze = lambda bz: bz * qt.tensor(I, Z, In)
    
    # Magnetic field on nucleus (rg = ratio of nuclear/electron gyromagnetic ratios)
    Hbxn = lambda bx, rg: rg * bx * qt.tensor(I, I, Xn)
    Hbyn = lambda by, rg: rg * by * qt.tensor(I, I, Yn)
    Hbzn = lambda bz, rg: rg * bz * qt.tensor(I, I, Zn)
    
    # Magnetic field on orbital degree of freedom
    # Term: f·μB·Bz·L̂z with f = p·gL (called q here) and L̂z = 2·Z (eigenvalues ±1).
    # Units: bz = gS·μB·Bz, so the exact coefficient is (2·q/gS)·bz·Z
    # (the tensor Z supplies L̂z/2; 2/gS converts bz back to μB·Bz exactly).
    Hbzo = lambda bz, q: (2 * q / params.gS) * bz * qt.tensor(Z, I, In)
    
    # Zeeman zz correction (4th term in eq. 9, Thiering & Gali 2018, PhysRevX.8.021063)
    # +2δf·μB·Sz·Bz corrects g_zz due to asymmetric Ham reduction (p_{3/2} ≠ p_{1/2}).
    # SIGN: eq. (9) of the 2018 paper PRINTS a minus sign, but that contradicts the
    # paper's own per-branch Hamiltonian (eq. C1 with defs C3/C8/C9), which requires
    #   g_zz(E_3/2) = gS + 2·gL·p_{3/2},   g_zz(E_1/2) = gS - 2·gL·p_{1/2},
    # i.e. a PLUS sign, and contradicts their statement that this term ENHANCES g_zz.
    # The corrected form appears in Tóth, Gali & Thiering (arXiv:2512.05704, eq. 18):
    #   +μB·(f·Bz·Lz + 2·δf·Bz·Sz).
    # With the minus sign each branch gets the OTHER branch's Ham factor (this was the
    # "abnormal spectrum" that led to delta_f being zeroed out; huge for the excited
    # state where δf = 0.237). In code units (bz = gS·μB·Bz): coefficient +(2·δf/gS).
    Hbze_corr = lambda bz, delta_f: +(2 * delta_f / params.gS) * bz * qt.tensor(I, Z, In)
    
    # Hyperfine coupling (DJT form), VERBATIM from eq. (2) of Tóth, Gali & Thiering,
    # PRB 112, 155201 (2025), DOI 10.1103/fq19-lfmv (= arXiv:2408.10407v3):
    #   H_HF = [½A⊥(S+I- + S-I+) + 2A∥·Sz·Iz] ⊗ I_orb
    #          - A1[(Sz I+ + S+ Iz)σ+ + (Sz I- + S- Iz)σ-]
    #          + (A2/2)[S- I- σ+ + S+ I+ σ-],      σ± = |e±⟩⟨e∓|
    # Definitions (their eq. 4): A∥ = ½A^x_zz, A⊥ = ½(A^x_xx + A^x_yy), A1 = q·A^x_xz,
    # A2 = q·(A^x_yy - A^x_xx) — the factor 2 on A∥ is the paper's own convention
    # (A∥ is HALF the physical zz coupling), so do not "fix" it away.
    # The σ± pairings look like they violate angular momentum conservation but do not:
    # σ∓ flips the real orbital momentum twice (through the deep a_1g/a_2u orbitals),
    # so conservation holds mod 3 (C3 symmetry) — see the discussion below their eq. (2).
    # Observables: zero-field splitting within each SOC branch = A∥; the PLE-visible
    # splitting A_PLE = A∥_exc - A∥_gnd = -473 MHz matches the measured -484(8) MHz.
    Hhf = lambda Aperp, Apar, A1, A2: (
        # Static part: [½A⊥(S+I- + S-I+) + 2A∥ Sz Iz] ⊗ I_orb
        (Aperp / 2) * (qt.tensor(I, Sp, Inm) + qt.tensor(I, Sm, Inp)) +
        2 * Apar * qt.tensor(I, Z, Zn) +
        # Dynamic part: -A1 [(Sz I+ + S+ Iz)σ+ + (Sz I- + S- Iz)σ-]
        -A1 * (qt.tensor(sigma_p, Z, Inp) + qt.tensor(sigma_p, Sp, Zn) +
               qt.tensor(sigma_m, Z, Inm) + qt.tensor(sigma_m, Sm, Zn)) +
        # Dynamic part: (A2/2) [S- I- σ+ + S+ I+ σ-]
        (A2 / 2) * (qt.tensor(sigma_p, Sm, Inm) + qt.tensor(sigma_m, Sp, Inp))
    )

    # SOC (Spin-Orbit Coupling): H = -λ·L̂z·Ŝz (Thiering & Gali 2018, eq. 2; the
    # minus sign encodes the HOLE character of the e_g/e_u states). With λ > 0 the
    # ALIGNED doublet E_3/2 = {|e+ ↑⟩, |e- ↓⟩} lies lowest in BOTH manifolds, so the
    # lower doublet has g_zz = gS + 2f > 2, as measured (e.g. Meesala et al., PRB 97,
    # 205444: lower-doublet slope 30.8 GHz/T > 28 GHz/T for SiV). Do NOT flip this
    # sign: that swaps the doublet ordering relative to every other orbital-diagonal
    # term (orbital Zeeman, δf correction, A1/A2 pairings) and contradicts experiment.
    # Factor of 2: orbital Z is a spin-1/2 operator (±1/2) while L̂z has eigenvalues ±1.
    Hsoc = lambda L: -2 * L * qt.tensor(Z, Z, In)
    
    # IOC (Iso-Orbital Coupling, also called upsilon)
    # Factor of 2 since each Z has a factor of 1/2
    # TODO: this is not in general true for Sn != 1/2
    Hioc = lambda u: 2 * u * qt.tensor(Z, I, Zn)
    
    # Strain/Jahn-Teller terms
    # GAUGE NOTE: with the dipole set below fixed to Hepp's convention
    # (px = σx, py = σy, pz = 2·I), the e± phase gauge is pinned, and in that gauge
    # Meesala's strain Hamiltonian (PRB 97, 205444, eq. 3) reads
    # +ε_Egx·σx - ε_Egy·σy. The signs coded here are the opposite, i.e. this
    # code's (alpha, beta) = (-ε_Egx, -ε_Egy) in Meesala's convention. Eigenvalues
    # only depend on alpha² + beta², but when eta mixes z with x/y polarization the
    # px/pz interference makes sign(alpha) OBSERVABLE in line intensities — when
    # fitting experimental intensities, try both signs of (alpha, beta).
    Hegx = lambda alpha: -2 * alpha * qt.tensor(X, I, In)
    Hegy = lambda beta: 2 * beta * qt.tensor(Y, I, In)
    
    # Dipole moment operators [px, py, pz] in eg+/eg- basis
    # After transformation from egx/egy to eg+/eg- basis:
    # Z (in egx/egy) => -X (in eg+/eg-)
    # -X (in egx/egy) => -Y (in eg+/eg-)
    # I (in egx/egy) => I (in eg+/eg-)
    
    # Directly define the transformed operators
    p_orbital = [
        2*X,  # px
        2*Y,  # py
        2*I    # pz
    ]

    # Tensor with electron and nuclear spaces
    # Structure: orbital ⊗ electron ⊗ nuclear
    p = [qt.tensor(p_op, I, In) for p_op in p_orbital]
    # Keep as list of QuTiP operators instead of numpy array
    
    # Reference Hamiltonian (SOC + strain only, defaults from manifold)
    Href = lambda alpha=0, beta=0, L=_L: Hsoc(L) + Hegx(alpha) + Hegy(beta)
    
    # Total Hamiltonian (defaults from manifold; bx, by, bz have no defaults)
    def H(bx, by, bz, alpha=0, beta=0, rg=_rg, q=_q, Aperp=_Aperp, Apar=_Apar,
          L=_L, upsilon=0, A1=_A1, A2=_A2, delta_f=_delta_f):
        """Evaluate the complete single-manifold Hamiltonian.
        
        Parameters
        ----------
        bx, by, bz : scalar
            Magnetic-field components in the original frequency-like code
            units.
        alpha, beta : scalar, optional
            Strain/Jahn--Teller parameters.
        rg : scalar, optional
            Nuclear-to-electron Zeeman ratio.
        q : scalar, optional
            Orbital magnetic-field susceptibility.
        Aperp, Apar : scalar, optional
            Perpendicular and parallel DJT hyperfine couplings.
        L : scalar, optional
            Spin-orbit coupling.
        upsilon : scalar, optional
            Iso-orbital coupling.
        A1, A2 : scalar, optional
            Orbital-off-diagonal DJT hyperfine couplings.
        delta_f : scalar, optional
            Asymmetric-Ham electron-Zeeman correction.
        
        Returns
        -------
        qutip.Qobj
            Total Hamiltonian operator.
        """
        return (Href(alpha=alpha, beta=beta, L=L) + 
                (Hbxe(bx) + Hbxn(bx, rg)) + 
                (Hbye(by) + Hbyn(by, rg)) + 
                (Hbze(bz) + Hbzn(bz, rg) + Hbzo(bz, q) + Hbze_corr(bz, delta_f)) + 
                Hhf(Aperp, Apar, A1, A2) + 
                Hioc(upsilon))
    
    return H, Href, p, J2


def solve_hamiltonian(B, theta, phi, manifold='ground', alpha=0, beta=0, **kwargs):
    """Diagonalize the Hamiltonian over a magnetic-field sweep.
    
    Parameters
    ----------
    B : scalar or array_like
        Magnetic-field magnitude or one-dimensional sweep.
    theta : scalar
        Polar field angle in radians.
    phi : scalar
        Azimuthal field angle in radians.
    manifold : {'ground', 'excited'}, optional
        Electronic manifold.
    alpha, beta : scalar, optional
        Strain/Jahn--Teller parameters.
    **kwargs
        Overrides forwarded to the full Hamiltonian builder, such as ``q``,
        ``Aperp``, ``Apar``, ``A1``, ``A2``, ``L``, ``delta_f``, ``upsilon``,
        or ``rg``.
    
    Returns
    -------
    E : numpy.ndarray, shape (num_fields, num_states)
        Eigenvalues.
    Eref : numpy.ndarray, shape (num_states,)
        Reference-Hamiltonian eigenvalues.
    U : numpy.ndarray, shape (num_fields, dim, num_states)
        Eigenvectors arranged as columns.
    U_states : list of list of qutip.Qobj
        ``U_states[b][s]`` is state ``s`` at field index ``b``.
    alignment : numpy.ndarray, shape (num_fields, num_states)
        Expectation values of ``J2``.
    p : list of qutip.Qobj
        Dipole operators ``[p_x, p_y, p_z]``.
    """
    H, Href, p, J2 = create_hamiltonian_nuclear(manifold)
    
    # Convert B to array for iteration
    B = np.atleast_1d(B)
    
    # Calculate magnetic field components for each B value
    bz_vals = B * np.cos(theta)
    bx_vals = B * np.sin(theta) * np.cos(phi)
    by_vals = B * np.sin(theta) * np.sin(phi)
    
    # Solve Hamiltonian for each B value
    E = []
    U = []
    U_states = []
    alignment = []
    
    for i in range(len(B)):
        H_qobj = H(bx_vals[i], by_vals[i], bz_vals[i], alpha=alpha, beta=beta, **kwargs)
        eigvals, eigvecs = H_qobj.eigenstates()
        E.append(eigvals)
        U_states.append(eigvecs)
        
        U_matrix = np.column_stack([vec.full().flatten() for vec in eigvecs])
        U.append(U_matrix)
        
        align_i = []
        for vec in eigvecs:
            align_i.append(qt.expect(J2, vec))
        alignment.append(align_i)
    
    E = np.array(E)
    U = np.array(U)
    alignment = np.array(alignment)
    
    # Reference Hamiltonian (B=0): forward L override if provided
    ref_kwargs = {k: kwargs[k] for k in ('L',) if k in kwargs}
    Href_qobj = Href(alpha=alpha, beta=beta, **ref_kwargs)
    Eref = np.array(Href_qobj.eigenstates()[0])
    
    return E, Eref, U, U_states, alignment, p


def calculate_cyclicity(transition):
    """Normalize transition rates into branching ratios.
    
    Parameters
    ----------
    transition : array_like, shape (num_excited, num_ground)
        Transition or emission rates. Row ``l`` contains all decay channels
        from excited state ``l``.
    
    Returns
    -------
    numpy.ndarray, shape (num_excited, num_ground)
        Row-normalized branching ratios. Rows with zero total rate are
        returned as zeros.
    """
    num_exc, num_gnd = transition.shape
    cyclicity = np.zeros((num_exc, num_gnd))
    
    for l in range(num_exc):
        # Total decay rate from excited state l to all ground states
        total_rate = np.sum(transition[l, :])
        
        if total_rate > 0:
            # Branching ratio to each ground state
            cyclicity[l, :] = transition[l, :] / total_rate

    return cyclicity


def calculate_cyclicity_spinflip(B, theta, phi, alpha=0, beta=0, alpha_exc=0, beta_exc=0,
                                 gnd_kwargs=None, exc_kwargs=None, cap=1e6):
    """Calculate eigenstate-keyed optical-cycling cyclicity.
    
    For a driven transition from ground state ``k`` to excited state ``l``,
    the returned cyclicity is
    
    ``emission[l, k] / (total_l - emission_folded[l, k])``.
    
    Decay into the upper orbital ground branch is folded onto the lower branch
    using overlaps of reduced electron-nuclear density matrices. This models
    fast orbital relaxation that acts as the identity on spin.
    
    Parameters
    ----------
    B : scalar or array_like
        Magnetic-field magnitude or sweep.
    theta, phi : scalar
        Field direction in radians.
    alpha, beta : scalar, optional
        Ground-manifold strain.
    alpha_exc, beta_exc : scalar, optional
        Excited-manifold strain.
    gnd_kwargs, exc_kwargs : dict or None, optional
        Ground- and excited-manifold Hamiltonian overrides.
    cap : float, optional
        Soft upper bound used when the pump-out rate underflows.
    
    Returns
    -------
    E : numpy.ndarray
        Ground-manifold eigenvalues.
    E_exc : numpy.ndarray
        Excited-manifold eigenvalues.
    emission : numpy.ndarray
        Polarization-summed spontaneous-emission rates.
    cyclicity : numpy.ndarray
        Mean collected photons before pump-out for each addressed line.
    spin_g, spin_e : numpy.ndarray
        Signs of the electron-spin projection along the magnetic field.
    emission_folded : numpy.ndarray
        Decay rates after folding the upper orbital branch onto the lower
        branch.
    """
    Ns = int(2*params.Sn + 1)
    B = np.atleast_1d(B)
    nstates = 4 * Ns

    # Unit vector along the magnetic field
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)

    # Electron-spin projection along B, in the orbital (x) electron (x) nuclear space.
    # x2 so eigenvalues are ~ +/-1 (well-defined sign in the Zeeman-dominated regime).
    I2 = qt.qeye(2)
    In = qt.qeye(Ns)
    Sn_field = 2 * (nx * qt.tensor(I2, qt.jmat(0.5, 'x'), In) +
                    ny * qt.tensor(I2, qt.jmat(0.5, 'y'), In) +
                    nz * qt.tensor(I2, qt.jmat(0.5, 'z'), In))

    # Solve both manifolds (p operators are state-independent, taken from ground call)
    E, _, _, U_states, _, p = solve_hamiltonian(
        B, theta, phi, manifold='ground', alpha=alpha, beta=beta, **(gnd_kwargs or {}))
    E_exc, _, _, U_exc_states, _, _ = solve_hamiltonian(
        B, theta, phi, manifold='excited', alpha=alpha_exc, beta=beta_exc, **(exc_kwargs or {}))

    n_low = nstates // 2  # lower orbital branch = energy-sorted states 0 .. 2*Ns-1
                          # (SOC branch gap >> Zeeman/hyperfine, so ordering is stable)

    emission = np.zeros((len(B), nstates, nstates))
    emission_folded = np.zeros((len(B), nstates, n_low))
    cyclicity = np.zeros((len(B), nstates, nstates))
    spin_g = np.zeros((len(B), nstates))
    spin_e = np.zeros((len(B), nstates))

    for i in range(len(B)):
        gnd = U_states[i]
        exc = U_exc_states[i]

        # Rough electron-spin-projection sign of every eigenstate along B. This is
        # ONLY a coarse diagnostic label for printing: under the hyperfine + DJT
        # mixing neither electron nor nuclear spin is a good quantum number (the
        # eigenstates are entangled superpositions), so this sign does NOT enter the
        # cyclicity below -- pump-out is defined purely from the eigenstates.
        sg = np.array([np.sign(qt.expect(Sn_field, s)) for s in gnd])
        se = np.array([np.sign(qt.expect(Sn_field, s)) for s in exc])
        spin_g[i] = sg
        spin_e[i] = se

        # Polarization-summed (incoherent) emission rates |<exc_l| p_j |gnd_k>|^2.
        # bra*op*ket returns a 1x1 Qobj (qutip 4) or a plain complex scalar
        # (qutip 5) depending on version -- handle both robustly.
        for l in range(nstates):
            for k in range(nstates):
                rate = 0.0
                for pj in p:
                    me = exc[l].dag() * pj * gnd[k]
                    val = me.full()[0, 0] if hasattr(me, 'full') else complex(me)
                    rate += abs(val)**2
                emission[i, l, k] = rate

        # Fold upper-branch decays onto the lower branch FIRST (needed for the
        # cyclicity below): population reaching the upper orbital ground branch
        # relaxes down by fast single-phonon spontaneous orbital relaxation, which
        # acts as the identity on the total spin state. Redistribute each upper
        # state's population among the lower states by the TOTAL-SPIN overlap of the
        # eigenstates -- no electron/nuclear separation, since neither is a good
        # quantum number under the hyperfine + DJT mixing.
        rho_spin = [s.ptrace([1, 2]) for s in gnd]  # full (unresolved) spin density matrix
        T = np.zeros((nstates - n_low, n_low))
        for u in range(n_low, nstates):
            for k in range(n_low):
                T[u - n_low, k] = np.real((rho_spin[u] * rho_spin[k]).tr())
            T[u - n_low, :] /= T[u - n_low, :].sum()  # conserve population
        emission_folded[i] = emission[i, :, :n_low] + emission[i, :, n_low:] @ T

        # True per-line readout cyclicity: expected number of photons collected on
        # the addressed optical line (exc l -> gnd k) before the emitter is optically
        # pumped into ANY OTHER ground eigenstate -- a different optical frequency the
        # readout laser no longer drives. No spin label enters; "pump-out" is simply
        # "the population does not return to the addressed eigenstate k after the fast
        # orbital relaxation". Per absorb/emit cycle starting in k:
        #   collected photons / cycle = emission[l, k]        / total_l   (direct l->k line)
        #   prob. the cycle survives  = emission_folded[l, k] / total_l   (population back in k)
        # so, summing the resulting geometric series,
        #   photons before pump-out = emission[l, k] / (total_l - emission_folded[l, k]).
        # Only lower-branch states (k < n_low) are addressable readout lines; the
        # upper-branch ground columns are not driven and keep cyclicity = 0.
        for l in range(nstates):
            total_l = emission[i, l, :].sum()
            floor = total_l / cap if total_l > 0 else 1.0  # soft cap when pump-out underflows
            for k in range(n_low):
                pumpout = total_l - emission_folded[i, l, k]  # all decay not returning to k
                cyclicity[i, l, k] = emission[i, l, k] / max(pumpout, floor)

    return E, E_exc, emission, cyclicity, spin_g, spin_e, emission_folded


def PLE_transitions(B, theta, phi, eta, alpha=0, beta=0, alpha_exc=0, beta_exc=0,
                     gnd_kwargs=None, exc_kwargs=None):
    """Calculate polarization-resolved PLE transition intensities.
    
    Excitation is calculated from the coherent polarization projection
    ``|<exc_l|eta dot p|gnd_k>|**2``. Spontaneous-emission branching ratios
    are calculated from the incoherent sum over the three dipole channels.
    
    Parameters
    ----------
    B : scalar or array_like
        Magnetic-field magnitude or sweep.
    theta, phi : scalar
        Field direction in radians.
    eta : array_like, shape (3,)
        Optical polarization coefficients.
    alpha, beta : scalar, optional
        Ground-manifold strain.
    alpha_exc, beta_exc : scalar, optional
        Excited-manifold strain.
    gnd_kwargs, exc_kwargs : dict or None, optional
        Ground- and excited-manifold Hamiltonian overrides.
    
    Returns
    -------
    E, Eref, U, alignment : numpy.ndarray
        Ground-manifold eigensystem quantities.
    E_exc, Eref_exc, U_exc, alignment_exc : numpy.ndarray
        Excited-manifold eigensystem quantities.
    transition : numpy.ndarray
        Coherent polarization-projected excitation rates.
    cyclicity : numpy.ndarray
        Polarization-summed spontaneous-emission branching ratios.
    """
    Ns = int(2*params.Sn + 1)
    
    # Convert B to array for iteration
    B = np.atleast_1d(B)
    
    # Solve ground-state Hamiltonian (p operators are state-independent, take from either)
    E, Eref, U, U_states, alignment, p = solve_hamiltonian(
        B, theta, phi, manifold='ground', alpha=alpha, beta=beta, **(gnd_kwargs or {})
    )
    
    # Solve excited-state Hamiltonian
    E_exc, Eref_exc, U_exc, U_exc_states, alignment_exc, _ = solve_hamiltonian(
        B, theta, phi, manifold='excited', alpha=alpha_exc, beta=beta_exc, **(exc_kwargs or {})
    )
    
    # Calculate transition dipole moments using QuTiP
    # For each B field value, calculate transition matrix elements
    # - transition: |<exc| eta·p |gnd>|² — COHERENT sum over the laser polarization
    #   components (correct for excitation by a single linearly polarized beam)
    # - emission: Σ_j |<exc| p_j |gnd>|² — INCOHERENT sum over all three dipole
    #   components (spontaneous emission goes into orthogonal vacuum modes, so the
    #   amplitudes add incoherently and do not depend on the laser polarization eta)
    transition = np.zeros((len(B), 4*Ns, 4*Ns))
    emission = np.zeros((len(B), 4*Ns, 4*Ns))

    for i in range(len(B)):
        # Use the already calculated eigenstates
        gnd_states = U_states[i]
        exc_states = U_exc_states[i]

        for l, exc_state in enumerate(exc_states):
            for k, gnd_state in enumerate(gnd_states):
                amp = 0.0 + 0.0j  # coherent eta-projected excitation amplitude
                rate = 0.0        # incoherent polarization-summed emission rate
                for j, p_op in enumerate(p):
                    matrix_element_result = exc_state.dag() * p_op * gnd_state
                    if hasattr(matrix_element_result, 'data'):
                        matrix_element = matrix_element_result.data.toarray()[0, 0]
                    else:
                        matrix_element = complex(matrix_element_result)
                    amp += eta[j] * matrix_element
                    rate += np.abs(matrix_element)**2

                transition[i, l, k] = np.abs(amp)**2
                emission[i, l, k] = rate

    # Calculate cyclicity (branching ratios) from the EMISSION matrix: the decay
    # of |exc_l> is polarization-summed and laser-independent. (Previously this
    # used the eta-projected excitation rates, which dropped the y-dipole channel
    # and coherently interfered x/z decay amplitudes — unphysical for emission.)
    cyclicity = np.zeros((len(B), 4*Ns, 4*Ns))
    for i in range(len(B)):
        cyclicity[i] = calculate_cyclicity(emission[i])
    
    return E, Eref, U, alignment, E_exc, Eref_exc, U_exc, alignment_exc, transition, cyclicity


def _folded_cyclicity_weight(cyclicity,
                             cyclicity_min=None, cyclicity_weight=False,
                             cyclicity_half=1.0, cyclicity_softness=4.0):
    """Convert folded cyclicity into a per-line PLE brightness weight.
    
    Parameters
    ----------
    cyclicity : array_like, shape (num_fields, num_excited, num_ground)
        Eigenstate-keyed cyclicity from
        :func:`calculate_cyclicity_spinflip`.
    cyclicity_min : float or None, optional
        Soft visibility threshold. ``None`` disables the threshold.
    cyclicity_weight : bool, optional
        Apply the smooth factor ``C / (C + cyclicity_half)``.
    cyclicity_half : float, optional
        Cyclicity at which the smooth factor is one half.
    cyclicity_softness : float, optional
        Hill exponent used by ``cyclicity_min``.
    
    Returns
    -------
    weight : numpy.ndarray
        Multiplicative brightness factor with the full transition-matrix
        shape.
    cyclicity_lower : numpy.ndarray
        Lower-branch cyclicity values.
    """
    nB, nstates, _ = cyclicity.shape
    n_low = nstates // 2                 # only the lower orbital branch is addressable
    C = cyclicity[:, :, :n_low]          # eigenstate per-line cyclicity, lower ground states

    gate = np.ones((nB, nstates, n_low))
    if cyclicity_weight:
        gate = gate * (C / (C + cyclicity_half))
    if cyclicity_min is not None:
        n = cyclicity_softness
        gate = gate * (C**n / (C**n + cyclicity_min**n))

    weight = np.ones((nB, nstates, nstates))
    weight[:, :, :n_low] = gate
    return weight, C


def PLE_spectrum(f_meas, B, theta, phi, eta, intensity=1.0, lw=0.080,
                 alpha=0, beta=0, alpha_exc=0, beta_exc=0,
                 gnd_kwargs=None, exc_kwargs=None,
                 cyclicity_min=None, cyclicity_weight=False,
                 cyclicity_half=1.0, cyclicity_softness=4.0,
                 cyclicity_cap=1e6, return_cyclicity=False):
    """Calculate a Lorentzian-broadened PLE spectrum.
    
    Frequencies are referenced to the strained, hyperfine-free C transition.
    Optical-cycling brightness is disabled unless one of the cyclicity
    controls is requested.
    
    Parameters
    ----------
    f_meas : array_like
        Frequency grid.
    B : scalar or array_like
        Magnetic-field magnitude or sweep.
    theta, phi : scalar
        Field direction in radians.
    eta : array_like, shape (3,)
        Optical polarization coefficients.
    intensity : float, optional
        Overall spectrum scale.
    lw : float, optional
        Lorentzian full width at half maximum.
    alpha, beta : scalar, optional
        Ground-manifold strain.
    alpha_exc, beta_exc : scalar, optional
        Excited-manifold strain.
    gnd_kwargs, exc_kwargs : dict or None, optional
        Ground- and excited-manifold Hamiltonian overrides.
    cyclicity_min : float or None, optional
        Soft lower visibility threshold for folded cyclicity.
    cyclicity_weight : bool, optional
        Apply ``C / (C + cyclicity_half)`` brightness weighting.
    cyclicity_half : float, optional
        Half-brightness cyclicity.
    cyclicity_softness : float, optional
        Hill exponent for ``cyclicity_min``.
    cyclicity_cap : float, optional
        Underflow cap used by :func:`calculate_cyclicity_spinflip`.
    return_cyclicity : bool, optional
        Return ``(spectrum, cyclicity)`` instead of only the spectrum.
    
    Returns
    -------
    spectrum : numpy.ndarray
        PLE spectrum. A one-field input is squeezed to one dimension.
    cyclicity : numpy.ndarray, optional
        Lower-branch folded cyclicity, returned only when
        ``return_cyclicity=True``.
    """
    peak = lambda f, f0, a, sigma: a * (sigma/2)**2 / ((f - f0)**2 + (sigma/2)**2)  # Lorentzian
    Ns = int(2*params.Sn + 1)
    
    E, Eref, _, _, E_exc, Eref_exc, _, _, transition, _ = PLE_transitions(
        B, theta, phi, eta, alpha=alpha, beta=beta, alpha_exc=alpha_exc, beta_exc=beta_exc,
        gnd_kwargs=gnd_kwargs, exc_kwargs=exc_kwargs
    )

    # Calculate PLE spectrum
    B_size = len(E)
    num_states = 4 * Ns  # Total number of states (orbital * electron * nuclear)

    # Optical-cycling brightness weight (fix 1 fold + fix 2 visibility). Computed only
    # when a knob is on, so the default path stays identical to the bare-rate model and
    # existing callers are unaffected. Re-solving via calculate_cyclicity_spinflip keeps
    # PLE_transitions' 10-value signature (widely unpacked positionally) untouched.
    weight = None
    cyclicity = None
    if (cyclicity_min is not None) or cyclicity_weight or return_cyclicity:
        # Use the SAME eigenstate-keyed per-line cyclicity as SnV117_cyclicity_sphere
        # (calculate_cyclicity_spinflip's 4th return), so the PLE brightness gate and
        # the cyclicity plots share one definition. cyclicity_cap sets the underflow
        # ceiling inside that function.
        _, _, _, cyc_eig, _, _, _ = calculate_cyclicity_spinflip(
            B, theta, phi, alpha=alpha, beta=beta,
            alpha_exc=alpha_exc, beta_exc=beta_exc,
            gnd_kwargs=gnd_kwargs, exc_kwargs=exc_kwargs, cap=cyclicity_cap,
        )
        weight, cyclicity = _folded_cyclicity_weight(
            cyc_eig,
            cyclicity_min=cyclicity_min, cyclicity_weight=cyclicity_weight,
            cyclicity_half=cyclicity_half, cyclicity_softness=cyclicity_softness,
        )

    # Initialize PLE spectrum array
    PLE = np.zeros((B_size, len(f_meas)))

    # Calculate spectrum for each B field value
    for j in range(B_size):
        # Sum over all ground and excited state transitions
        for k in range(num_states):
            for l in range(num_states):
                # Transition frequency for the C transition
                f_transition = (E_exc[j, l] - Eref_exc[0]) - (E[j, k] - Eref[0])
                # Brightness = excitation rate, optionally dimmed by folded cyclicity
                amp = transition[j, l, k]
                if weight is not None:
                    amp = amp * weight[j, l, k]
                # Add Lorentzian peak for this transition
                PLE[j] += peak(f_meas, f_transition, amp, lw)

    # Apply overall intensity scaling
    PLE *= intensity

    # If single B value, return 1D array
    if B_size == 1:
        PLE = PLE[0]
        if cyclicity is not None:
            cyclicity = cyclicity[0]
    if return_cyclicity:
        return PLE, cyclicity
    return PLE
