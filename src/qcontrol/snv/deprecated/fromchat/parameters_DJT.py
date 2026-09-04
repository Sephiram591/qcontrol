"""
Parameters for SnV117 center in diamond (for use with DJT Hamiltonian).

References:
  - Ham reduction & orbital parameters (gL, p, delta_p):
    Thiering & Gali 2018, Table IV, PhysRevX.8.021063
  - Spin-orbit coupling (L, L_exc):
    Thiering & Gali 2018, PhysRevX.8.021063
  - DJT hyperfine parameters (A1, A2, A_parallel, A_perpendicular):
    Thiering & Gali 2025, Phys. Rev. B, DOI: 10.1103/fq19-lfmv
    "Magneto-optical properties of group-IV vacancy centers in diamond
     upon hydrostatic pressure"
    NOTE: These are NOT the same as the Harris et al. 2023 (PRX Quantum 4, 040301)
    values. Harris uses Afc/Add (orbital-averaged form), while the DJT decomposition
    separates the hyperfine into orbital-independent (A_par, A_perp) and
    orbital-nuclear coupling (A1, A2) components. The numerical values differ
    because they represent different decompositions of the hyperfine tensor.

Hyperfine parameters use the DJT form with A1, A2, A_parallel, A_perpendicular.
"""

# Spin values
S = 1/2  # Electron spin
Sn = 1/2  # Nuclear spin (117Sn)

# Electron g-factor
gS = 2.0023

# Ham reduction & orbital parameters (Table IV, Thiering & Gali 2018, PhysRevX.8.021063)
# Ground state SnV(2Eg)
gL_gnd = 0.328       # Stevens orbital reduction factor
p_32_gnd = 0.513     # Ham reduction factor for E_{3/2} state
p_12_gnd = 0.429     # Ham reduction factor for E_{1/2} state
p_gnd = 0.471        # average: (p_{3/2} + p_{1/2}) / 2
delta_p_gnd = 0.042  # asymmetry: (p_{3/2} - p_{1/2}) / 2
# Zeeman zz correction δf = δp * gL (paper: 0.014). Physical, keep enabled;
# the "abnormal spectrum" it used to cause came from a sign error in Hbze_corr
# (now fixed: the term is +2δf·μB·Bz·Sz, giving g_zz(E_3/2) = gS + 2·gL·p_{3/2}).
delta_f_gnd = delta_p_gnd * gL_gnd  # = 0.0138

# Excited state SnV(2Eu)
# gL_exc = 0.782
gL_exc = 0.88 # Asher revised for accuracy
p_32_exc = 0.429
p_12_exc = -0.178    # negative: E_{1/2} vibronic state dominated by SOC-favored components
p_exc = 0.125
delta_p_exc = 0.303
delta_f_exc = delta_p_exc * gL_exc  # = 0.237 (paper: 0.238) — large, dominates ES g_zz

# Orbital magnetic field susceptibility: f = p * gL
q = gL_gnd * p_gnd       # ground state
q_exc = gL_exc * p_exc   # excited state

# Spin-orbit coupling
L = 830.0  # [GHz] spin-orbit coupling ground state
L_exc = 3000.0  # [GHz] spin-orbit coupling excited state

# Hyperfine Properties
# Ratio of electron to proton mass (mu_N/mu_B)
rmep = 5.44617021e-4
# Nuclear/electron Zeeman ratio in code units (bz = gS*mu_B*B):
# H_n = -g_n*mu_N*B·I with g_n(117Sn) = -2.00208 (mu = -1.00104 mu_N, I = 1/2),
# so the coefficient is -g_n*mu_N/(gS*mu_B) = +2.00208*rmep/gS — positive, i.e.
# the same sign as the electron term, because the 117Sn moment is negative.
rg = 2.00208 * rmep / gS

# Hyperfine parameters for 117Sn (S_n = 1/2), DJT form.
# Values and SIGNS verbatim from Table III of Tóth, Gali & Thiering, PRB 112,
# 155201 (2025), DOI 10.1103/fq19-lfmv (= arXiv:2408.10407v3, published version;
# note the arXiv v1 used a different parametrization: A∥SzIz with A∥ = 976 MHz —
# same net Hamiltonian). Used with their eq. (2) implemented verbatim in
# hamiltonian_DJT.Hhf (note its 2A∥SzIz: A∥ is defined as ½A_zz, their eq. 4).
# Observable check: A_PLE = A∥_exc - A∥_gnd = -473 MHz vs measured -484(8) MHz
# (Harris et al. 2023).

# Ground state (2Eg)
A1_gnd = 1.1 / 1000.0        # [GHz] dynamic (orbital-off-diagonal) hyperfine A1
A2_gnd = 1.9 / 1000.0        # [GHz] dynamic (orbital-off-diagonal) hyperfine A2
Apar_gnd = 488.0 / 1000.0    # [GHz] parallel hyperfine coupling A∥ (= ½A_zz)
Aperp_gnd = 1029.7 / 1000.0  # [GHz] perpendicular hyperfine coupling A⊥

# Excited state (2Eu)
A1_exc = 0.1 / 1000.0        # [GHz] dynamic hyperfine A1
A2_exc = -0.43 / 1000.0      # [GHz] dynamic hyperfine A2
Apar_exc = 15.0 / 1000.0     # [GHz] parallel hyperfine coupling A∥ (= ½A_zz)
Aperp_exc = 32.3 / 1000.0    # [GHz] perpendicular hyperfine coupling A⊥
