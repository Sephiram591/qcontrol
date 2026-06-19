"""
Parameters for SnV117 center in diamond (for use with QuTiP Hamiltonian).

All parameters are for the 117SnV- defect center.
"""

# Spin values
S = 1/2  # Electron spin
Sn = 1/2  # Nuclear spin (117Sn)

# Orbital magnetic field susceptibility
q = 0.3 # [] ground state orbital magnetic field susceptibility
q_exc = 0.073  # [] excited state orbital magnetic field susceptibility

# Spin-orbit coupling
L = 830.0  # [GHz] spin-orbit coupling ground state
L_exc = 3000.0  # [GHz] spin-orbit coupling excited state

# Hyperfine Properties
# Ratio of electron to proton mass
rmep = 5.44617021e-4
# Ratio of nuclear/electron gyromagnetic ratio (assuming g~2 for electrons)
rg = 2.00208 * rmep / 2



# Test hyperfine parameters from experiment
# Afc_gnd = 1300/1000.0  # [GHz] isotropic hyperfine coupling
# Add_gnd = -50/1000.0   # [GHz] anisotropic hyperfine coupling
# Afc_exc = 300/1000.0  # [GHz] isotropic hyperfine coupling
# Add_exc = 100/1000.0   # [GHz] anisotropic hyperfine coupling


# Values below is from DFT from paper PRX QUANTUM 4, 040301 (2023)
# Ground state hyperfine parameters (from DFT)
Afc_gnd = 1275.04/1000.0  # [GHz] isotropic hyperfine coupling
Add_gnd = -24.47/1000.0   # [GHz] anisotropic hyperfine coupling
# Asher: I noted a typo in the hyperfine paper. The Add_gnd should be divided by 2.
Apar_gnd = (Afc_gnd + Add_gnd) # [GHz] parallel hyperfine coupling
Aperp_gnd = (Afc_gnd - Add_gnd / 2)  # [GHz] perpendicular hyperfine coupling

# Excited state hyperfine parameters (from DFT) [GHz]
Afc_exc = 386.74/1000.0  # [GHz] isotropic hyperfine coupling
Add_exc = 230.43/1000.0   # [GHz] anisotropic hyperfine coupling
Apar_exc = (Afc_exc + Add_exc) # [GHz] parallel hyperfine coupling
Aperp_exc = (Afc_exc - Add_exc / 2)  # [GHz] perpendicular hyperfine coupling

#old parameters from the previous version of the code
# Afc_gnd = 661.9/1000.0  # [GHz] isotropic hyperfine coupling
# Add_gnd = -5.53/1000.0   # [GHz] anisotropic hyperfine coupling
# Apar_exc = 672.39 / 1000.0   # [GHz] parallel hyperfine coupling
# Aperp_exc = 295.82 / 1000.0  # [GHz] perpendicular hyperfine coupling
