"""
Hamiltonian model for SnV117 center in diamond using QuTiP.

This module provides the Hamiltonian for SnV117 (spin 1/2 electron, spin 1/2 nuclear)
with the following interactions:
- SOC (Spin-Orbit Coupling)
- IOC (Iso-Orbital Coupling)
- Strain (Jahn-Teller)
- Magnetic field on electron
- Magnetic field on nucleus
- Magnetic field on orbital degree of freedom
- Hyperfine coupling (spin 1/2 * spin 1/2)
"""

import numpy as np
import qutip as qt
import qcontrol.snv120.parameters as params


def create_hamiltonian_nuclear():
    """
    Create a Hamiltonian for SnV120 center in diamond using QuTiP.
    
    SnV117 has:
    - Electron spin S = 1/2
    - Nuclear spin Sn = 1/2 (from params)
    - Orbital degree of freedom (2D)
    
    Tensor product structure: orbital ⊗ electron ⊗ nuclear
    
    Returns:
        tuple: (H, Href, p, J2)
            - H: Total Hamiltonian function
            - Href: Reference Hamiltonian (SOC + strain only)
            - p: Dipole moment operators [px, py, pz]
            - J2: Total angular momentum squared operator
    """
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
    # Units: Since B is in units of g_e * mu_B, we need to divide by g_e to get orbital contribution
    # The orbital magnetic moment is mu_B * L, while electron is g_e * mu_B * S
    # Therefore, the orbital term needs a factor of 1/g_e ≈ 1/2
    Hbzo = lambda bz, q: (q / 2) * bz * qt.tensor(Z, I, In)
    
    S_ops = (X, Y, Z)
    N_ops = (Xn, Yn, Zn)
    # In the e+, e- basis:
    # sigma_z from the ex, ey basis becomes -sigma_x
    # sigma_x from the ex, ey basis becomes -sigma_y
    sigma_z_xy_in_pm = -2 * X
    sigma_x_xy_in_pm = -2 * Y
    def hf_block(B, orbital_op):
        H = 0 * qt.tensor(orbital_op, S_ops[0], N_ops[0])
        for i in range(3):
            for j in range(3):
                H += B[i, j] * qt.tensor(orbital_op, S_ops[i], N_ops[j])
        return H
    def Hhf(A, Ax, Ay):
        return (
            hf_block(A,  I)
            + hf_block(Ax, sigma_z_xy_in_pm)
            + hf_block(Ay, sigma_x_xy_in_pm)
        )
    # SOC (Spin-Orbit Coupling)
    # Factor of 2 since each Z has a factor of 1/2
    Hsoc = lambda L: 2 * L * qt.tensor(Z, Z, In)
    
    # IOC (Iso-Orbital Coupling, also called upsilon)
    # Factor of 2 since each Z has a factor of 1/2
    # TODO: this is not in general true for Sn != 1/2
    Hioc = lambda u: 2 * u * qt.tensor(Z, I, Zn)
    
    # Strain/Jahn-Teller terms # TODO: removing the "-"" sign because my calculation does not have it.
    Hegx = lambda alpha: -2 * alpha * qt.tensor(X, I, In)
    Hegy = lambda beta: 2 * beta * qt.tensor(Y, I, In)
    
    # Dipole moment operators [px, py, pz] in eg+/eg- basis
    # After transformation from egx/egy to eg+/eg- basis:
    # 2*Z (in egx/egy) => -X (in eg+/eg-)
    # -2*X (in egx/egy) => -Y (in eg+/eg-)
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
    
    # Reference Hamiltonian (SOC + strain only)
    Href = lambda L, alpha, beta: Hsoc(L) + Hegx(alpha) + Hegy(beta)
    
    # Total Hamiltonian
    def H(bx, by, bz, rg, q, A, Ax, Ay, L, alpha, beta, upsilon=0):
        """
        Total Hamiltonian for SnV117.
        
        Parameters:
            bx, by, bz: Magnetic field components (GHz)
            rg: Ratio of nuclear/electron gyromagnetic ratios
            q: Orbital magnetic field susceptibility
            A: Hyperfine coupling tensor (GHz)
            Ax: Hyperfine coupling tensor in x direction (GHz)
            Ay: Hyperfine coupling tensor in y direction (GHz)
            L: Spin-orbit coupling strength (GHz)
            alpha: Strain parameter in x direction (GHz)
            beta: Strain parameter in y direction (GHz)
            upsilon: Iso-orbital coupling strength (GHz), default=0
            
        Returns:
            Qobj: Hamiltonian operator
        """
        return (Href(L, alpha, beta) + 
                (Hbxe(bx) + Hbxn(bx, rg)) + 
                (Hbye(by) + Hbyn(by, rg)) + 
                (Hbze(bz) + Hbzn(bz, rg) + Hbzo(bz, q)) + 
                Hhf(A, Ax, Ay) + 
                Hioc(upsilon))

    return H, Href, p, J2


def solve_hamiltonian(B, theta, phi, q, A, Ax, Ay, L, alpha, beta):
    """
    Solve the Hamiltonian for a sweep of B-field strength.
    
    Parameters:
        B: Magnetic field strength array (GHz)
        theta: Polar angle (rad)
        phi: Azimuthal angle (rad)
        q: Orbital magnetic field susceptibility
        A: Hyperfine coupling tensor (GHz)
        Ax: Hyperfine coupling tensor in x direction (GHz)
        Ay: Hyperfine coupling tensor in y direction (GHz)
        L: Spin-orbit coupling strength (GHz)
        alpha: Strain parameter in x direction (GHz)
        beta: Strain parameter in y direction (GHz)
        
    Returns:
        tuple: (E, Eref, U, U_states, alignment)
            - E: Eigenvalues array (len(B) x num_states)
            - Eref: Reference eigenvalues (no B-field)
            - U: Eigenvector matrices (len(B) x dim x num_states)
            - U_states: List of QuTiP state vectors for each B value
            - alignment: J² expectation values (len(B) x num_states)
    """
    H, Href, p, J2 = create_hamiltonian_nuclear()
    
    # Convert B to array for iteration
    B = np.atleast_1d(B)
    
    # Calculate magnetic field components for each B value
    bz_vals = B * np.cos(theta)
    bx_vals = B * np.sin(theta) * np.cos(phi)
    by_vals = B * np.sin(theta) * np.sin(phi)
    
    # Solve Hamiltonian for each B value
    E = []
    U = []
    U_states = []  # Store QuTiP state vectors for later use
    alignment = []
    
    for i in range(len(B)):
        H_qobj = H(bx_vals[i], by_vals[i], bz_vals[i], params.rg, q, A, Ax, Ay, L, alpha, beta)
        # Use QuTiP's eigenstates method which returns (eigenvalues, eigenvectors)
        eigvals, eigvecs = H_qobj.eigenstates()
        E.append(eigvals)
        U_states.append(eigvecs)  # Store QuTiP states
        
        # Convert eigenvectors to matrix form (columns are eigenvectors)
        U_matrix = np.column_stack([vec.full().flatten() for vec in eigvecs])
        U.append(U_matrix)
        
        # Calculate alignment: <ψ|J²|ψ> for each eigenstate
        align_i = []
        for vec in eigvecs:
            align_i.append(qt.expect(J2, vec))
        alignment.append(align_i)
    
    E = np.array(E)
    U = np.array(U)
    alignment = np.array(alignment)
    
    # Reference Hamiltonian (B=0)
    Href_qobj = Href(L, alpha, 0)
    Eref, Uref_vecs = Href_qobj.eigenstates()
    Eref = np.array(Eref)
    
    return E, Eref, U, U_states, alignment


def calculate_cyclicity(transition):
    """
    Calculate the cyclicity of each transition using the branching ratio:
        cyclicity[l, k] = |⟨k|p|l⟩|² / Σ_k' |⟨k'|p|l⟩|²
    This gives the probability of returning to ground state k after being 
    excited to state l.
    
    Parameters:
        transition: 2D array of transition rates |⟨exc_l|p·η|gnd_k⟩|²
                    shape (num_exc_states, num_gnd_states)
        
    Returns:
        cyclicity: 2D array of shape (num_exc_states, num_gnd_states)
                   cyclicity[l, k] = branching ratio from exc state l to gnd state k
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


def PLE_transitions(B, theta, phi, eta, alpha=0, beta=0, alpha_exc=0, beta_exc=0, A_gnd=None, Ax_gnd=None, Ay_gnd=None, A_exc=None, Ax_exc=None, Ay_exc=None):
    """
    Calculate PLE transition intensities for a sweep of B-field strength using QuTiP.
    
    Parameters:
        B: Magnetic field strength (GHz)
        theta: Polar angle (rad)
        phi: Azimuthal angle (rad)
        eta: Polarization vector [px, py, pz] (unitless) e.g. [1,0,1] for +x, 0 for y, +z
        alpha: Strain parameter in x direction (GHz)
        beta: Strain parameter in y direction (GHz)
        alpha_exc: Strain parameter in x direction of excited state (GHz)
        beta_exc: Strain parameter in y direction of excited state (GHz)
        A_gnd: Hyperfine coupling tensor of ground state (GHz)
        Ax_gnd: Hyperfine coupling tensor in x direction of ground state (GHz)
        Ay_gnd: Hyperfine coupling tensor in y direction of ground state (GHz)
        A_exc: Hyperfine coupling tensor of excited state (GHz)
        Ax_exc: Hyperfine coupling tensor in x direction of excited state (GHz)
        Ay_exc: Hyperfine coupling tensor in y direction of excited state (GHz)
    Returns:
        tuple: (E, Eref, U, alignment, E_exc, Eref_exc, U_exc, alignment_exc, transition, cyclicity)
            - cyclicity: Spin preservation probability for each transition (len(B) x num_exc x num_gnd)
    """
    H, Href, p, J2 = create_hamiltonian_nuclear()
    Ns = int(2*params.Sn + 1)
    if A_gnd is None:
        A_gnd = np.zeros((3,3))
    if Ax_gnd is None:
        Ax_gnd = np.zeros((3,3))
    if Ay_gnd is None:
        Ay_gnd = np.zeros((3,3))
    if A_exc is None:
        A_exc = np.zeros((3,3))
    if Ax_exc is None:
        Ax_exc = np.zeros((3,3))
    if Ay_exc is None:
        Ay_exc = np.zeros((3,3))
    # Convert B to array for iteration
    B = np.atleast_1d(B)
    
    # Solve ground-state Hamiltonian
    E, Eref, U, U_states, alignment = solve_hamiltonian(
        B, theta, phi, 
        params.q, A_gnd, Ax_gnd, Ay_gnd, params.L, 
        alpha, beta
    )
    
    # Solve excited-state Hamiltonian
    E_exc, Eref_exc, U_exc, U_exc_states, alignment_exc = solve_hamiltonian(
        B, theta, phi,
        params.q_exc, A_exc, Ax_exc, Ay_exc, params.L_exc,
        alpha_exc, beta_exc
    )
    
    # Calculate transition dipole moments using QuTiP
    # For each B field value, calculate transition matrix elements
    transition = np.zeros((len(B), 4*Ns, 4*Ns))
    
    for i in range(len(B)):
        # Use the already calculated eigenstates
        gnd_states = U_states[i]
        exc_states = U_exc_states[i]
        
        # Calculate transition matrix elements: |<exc|p·eta|gnd>|²
        for l, exc_state in enumerate(exc_states):
            for k, gnd_state in enumerate(gnd_states):
                dipole_elements = [] # Calculate <exc|p_i|gnd> for each polarization component
                for j, p_op in enumerate(p): # dot product of eta and p_op
                    if eta[j] != 0:  # Only calculate if polarization component is non-zero
                        matrix_element_result = exc_state.dag() * p_op * gnd_state
                        if hasattr(matrix_element_result, 'data'):
                            matrix_element = matrix_element_result.data.toarray()[0, 0]
                        else:
                            matrix_element = complex(matrix_element_result)
                        dipole_elements.append(eta[j] * matrix_element)
                
                # complete the dot product and take absolute square
                transition[i, l, k] = np.abs(np.sum(dipole_elements))**2
    
    # Calculate cyclicity from transition matrix (branching ratios)
    cyclicity = np.zeros((len(B), 4*Ns, 4*Ns))
    for i in range(len(B)):
        cyclicity[i] = calculate_cyclicity(transition[i])
    
    return E, Eref, U, alignment, E_exc, Eref_exc, U_exc, alignment_exc, transition, cyclicity


def PLE_spectrum(f_meas, B, theta, phi, eta, intensity=1.0, lw=0.080, 
                 alpha=0, beta=0, alpha_exc=0, beta_exc=0, A_gnd=None, Ax_gnd=None, Ay_gnd=None, A_exc=None, Ax_exc=None, Ay_exc=None):
    """
    Calculate PLE spectrum as sum of Lorentzians with same linewidth, 
    intensity modulated by transition intensity using QuTiP.

    Parameters:
        f_meas: Frequency array
        B: Magnetic field strength (GHz)
        theta: Polar angle (rad)
        phi: Azimuthal angle (rad)
        eta: Polarization vector [px, py, pz] (unitless) e.g. [1,0,1] for +x, 0 for y, +z
        intensity: Intensity of the PLE spectrum
        lw: Linewidth of the Lorentzian
        alpha: Strain parameter in x direction (GHz)
        beta: Strain parameter in y direction (GHz)
        alpha_exc: Strain parameter in x direction of excited state (GHz)
        beta_exc: Strain parameter in y direction of excited state (GHz)
        A_gnd: Hyperfine coupling tensor of ground state (GHz)
        Ax_gnd: Hyperfine coupling tensor in x direction of ground state (GHz)
        Ay_gnd: Hyperfine coupling tensor in y direction of ground state (GHz)
        A_exc: Hyperfine coupling tensor of excited state (GHz)
        Ax_exc: Hyperfine coupling tensor in x direction of excited state (GHz)
        Ay_exc: Hyperfine coupling tensor in y direction of excited state (GHz)
    """
    peak = lambda f, f0, a, sigma: a * (sigma/2)**2 / ((f - f0)**2 + (sigma/2)**2)  # Lorentzian
    Ns = int(2*params.Sn + 1)
    E, Eref, _, _, E_exc, Eref_exc, _, _, transition, _ = PLE_transitions(
        B, theta, phi, eta, alpha=alpha, beta=beta, alpha_exc=alpha_exc, beta_exc=beta_exc, A_gnd=A_gnd, Ax_gnd=Ax_gnd, Ay_gnd=Ay_gnd, A_exc=A_exc, Ax_exc=Ax_exc, Ay_exc=Ay_exc
    )
    
    # Calculate PLE spectrum
    B_size = len(E)
    num_states = 4 * Ns  # Total number of states (orbital * electron * nuclear)
    
    # Initialize PLE spectrum array
    PLE = np.zeros((B_size, len(f_meas)))
    
    # Calculate spectrum for each B field value
    for j in range(B_size):
        # Sum over all ground and excited state transitions
        for k in range(num_states):
            for l in range(num_states):
                # Transition frequency for the C transition
                f_transition = (E_exc[j, l] - Eref_exc[0]) - (E[j, k] - Eref[0])
                # print(f"f_transition: {f_transition}")
                # print(f"transition intensity: {transition[j, l, k]}")
                # Add Lorentzian peak for this transition
                PLE[j] += peak(f_meas, f_transition, transition[j, l, k], lw)
    
    # Apply overall intensity scaling
    PLE *= intensity
    
    # If single B value, return 1D array
    if B_size == 1:
        return PLE[0]
    return PLE
