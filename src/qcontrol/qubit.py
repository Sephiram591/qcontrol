import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.signal import chirp
from scipy.special import hyperu, laguerre, genlaguerre, hyp1f1
from scipy.constants import hbar
from scipy.constants import k as kB
from tqdm import tqdm
import dill
from scipy.integrate import quad
from qcontrol.tools import *

class ColorCenterQubit:
    def __init__(
        self,
        optical_fc =           2e9,
        optical_df =           0.3e9,
        optical_lifetime =     1/6e-9,           # Emitter lifetime
        mw_fc =                1.3e9,
        t2_star =              1.6e-6,         # T2* relaxation time
        branching_ratio =      0.001,   # Branching ratio
        dark_count_rate =      10e3,
        coupling_efficiency =  0.005,
        debye_waller_factor =  0.7,
        non_radiative_factor = 0.05,
        
        sigmaz_noise_fn =     None,

        ####################################### HEOM Settings #########################################
        gamma_DL =             None, # HEOM Bath bandwidth
        lambda_DL =            None, # HEOM Bath coupling strength
        bath_temp =            1.3, # HEOM Bath temperature
        Nk =                   3, # HEOM Bath number of exponentials
        max_depth =            3, # HEOM Solver maximum depth. NOTE- Should be larger than Nk

        ####################################### Trajectory purity settings ############################
        memory_tau=0, 
        purity_decay_fn=gaussian_decay, 
        purity_inverse_fn=inverse_gaussian_decay, 
        memory_delta_fn=delta_exponential_decay
    ):
        # System constants
        self.optical_fc = optical_fc
        self.optical_df = optical_df
        self.optical_lifetime = optical_lifetime
        # self.t2_star = t2_star
        self.branching_ratio = branching_ratio
        self.dark_count_rate = dark_count_rate
        self.coupling_efficiency = coupling_efficiency
        self.debye_waller_factor = debye_waller_factor
        self.non_radiative_factor = non_radiative_factor
        self.readout_settings = None
        self.sigmaz_noise_fn = sigmaz_noise_fn
        self.t2_star = t2_star

        # Derived constants
        self.f0 = self.optical_fc - self.optical_df/2
        self.f1 = self.optical_fc + self.optical_df/2

        self.w0 = 2 * np.pi * self.f0
        self.w1 = 2 * np.pi * self.f1

        self.mw_fc = mw_fc
        self.w_mw = 2 * np.pi * self.mw_fc

        self.current_dephasing_trajectory = np.array([0+0j, 0+0j, 0+0j])
        self.current_forgotten_trajectory = 0
        
        ####################################### HEOM Settings #########################################
        self.bath_temp = bath_temp
        self.gamma_DL = gamma_DL
        self.lambda_DL = lambda_DL
        # if not gamma_DL:
        #     self.gamma_DL = 1/t2_star
        # if not lambda_DL:
        #     self.lambda_DL = self.gamma_DL*hbar/(2*kB*self.bath_temp*self.t2_star)
        self.Nk = Nk
        self.max_depth = max_depth
        self.psi0 = qt.basis(4, 0).proj()
        
        ###################################### Trajectory Purity Settings #############################
        self.memory_tau = memory_tau
        self.purity_decay_fn = purity_decay_fn
        self.purity_inverse_fn = purity_inverse_fn
        self.memory_delta_fn = memory_delta_fn

    def _init_hamiltonian(self, include_excitation=True):
        # Basis states
        if include_excitation:
            self.g0 = qt.basis(4, 0)
            self.g1 = qt.basis(4, 1)
            self.e0 = qt.basis(4, 2)
            self.e1 = qt.basis(4, 3)

            # Expectation operators
            self.e0_proj = self.e0 * self.e0.dag()
            self.e1_proj = self.e1 * self.e1.dag()

            # Projectors and transition operators
            self.sig_g0e0 = self.g0 * self.e0.dag() + self.e0 * self.g0.dag()
            self.sig_g1e1 = self.g1 * self.e1.dag() + self.e1 * self.g1.dag()
            self.sig_g0g1 = self.g0 * self.g1.dag() + self.g1 * self.g0.dag()
            self.sigma_z_g = self.g0*self.g0.dag() - self.g1*self.g1.dag()

            # If psi0 does not have e0 and e1, expand it to include them
            # Check if psi0 is 2-dimensional (i.e., only g0 and g1)
            if qt.isket(self.psi0):
                M2 = qt.ket2dm(self.psi0).full()                       # 2×2 NumPy array
                M4 = np.zeros((4,4), dtype=complex)      # allocate 4×4
                M4[:2, :2] = M2                          # put your block back in the top‑left
                self.psi0 = qt.Qobj(M4, dims=[[4],[4]])  # wrap as a 4×4 Qobj

            self.H0 = (self.w0-self.w_mw/2) * self.e0 * self.e0.dag() + (self.w1+self.w_mw/2) * self.e1 * self.e1.dag() + self.w_mw/2 * (self.g1 * self.g1.dag() - self.g0 * self.g0.dag())

            # Collapse operators
            self.c_e0 = np.sqrt(self.optical_lifetime * (1 - self.branching_ratio)) * self.g0 * self.e0.dag()
            self.c_e1 = np.sqrt(self.optical_lifetime * (1 - self.branching_ratio)) * self.g1 * self.e1.dag()
            self.c_branching_e0 = np.sqrt(self.optical_lifetime * self.branching_ratio) * self.g1 * self.e0.dag()
            self.c_branching_e1 = np.sqrt(self.optical_lifetime * self.branching_ratio) * self.g0 * self.e1.dag()
            self.c_ops = [self.c_e0, self.c_e1, self.c_branching_e0, self.c_branching_e1]
        else:
            # print("Reducing to 2-dimensional Hilbert space")
            self.g0 = qt.basis(2, 0)
            self.g1 = qt.basis(2, 1)
            # If psi0 has e0 and e1, trace them out
            # Check if the Hilbert space is 4-dimensional (i.e., includes e0 and e1)
            if self.psi0.dims[0][0] == 4:
                if self.memory_tau:
                    dm_purity = self.psi0.purity()
                    # Mix the purity of the density matrix with the purity of the dephasing trajectory
                    total_purity = dm_purity * self.purity_decay_fn(self.current_forgotten_trajectory+np.linalg.norm(self.current_dephasing_trajectory))
                    total_trajectory = self.purity_inverse_fn(total_purity)
                    self.current_forgotten_trajectory = total_trajectory - np.linalg.norm(self.current_dephasing_trajectory)
                M4 = self.psi0.full()         # 4×4 array
                M2 = qt.Qobj(M4[:2, :2], dims=[[2],[2]])               # keep only rows 0–1 and cols 0–1
                if self.gamma_DL:
                    self.psi0 = M2
                else:   
                    self.psi0 = qt.Qobj(M2.eigenstates()[0], dims=[[2],[1]])  # Now a ket
            
            self.H0 = self.w_mw/2 * (self.g1 * self.g1.dag() - self.g0 * self.g0.dag())
            self.c_ops = []

        self.sig_g0g1 = self.g0 * self.g1.dag() + self.g1 * self.g0.dag()
        self.sigma_z_g = self.g0*self.g0.dag() - self.g1*self.g1.dag()
        self.g0_proj = self.g0 * self.g0.dag()
        self.g1_proj = self.g1 * self.g1.dag()
        self.g0_proj_g1 = self.g0 * self.g1.dag()
        self.g1_proj_g0 = self.g1 * self.g0.dag()
        
        self.e_ops = [self.g0_proj, self.g1_proj, self.g0_proj_g1, self.g1_proj_g0]
        if include_excitation:
            self.e_ops += [self.e0_proj, self.e1_proj]

        if self.gamma_DL:
            self.bath = qt.solver.heom.DrudeLorentzBath(self.sigma_z_g, lam=self.lambda_DL, gamma=self.gamma_DL, T=self.bath_temp, Nk=self.Nk)
            self.bath_env = qt.DrudeLorentzEnvironment(lam=self.lambda_DL, gamma=self.gamma_DL, T=self.bath_temp, Nk=self.Nk)
            self.bath_env_approx = self.bath_env.approximate(method="pade", Nk=self.Nk)
            # wlist = np.linspace(0, 2e8, 1000)
            # J = self.bath_env.spectral_density(wlist)
            # J_approx = self.bath_env_approx.spectral_density(wlist)


    def apply_pulse_sequence(self, tlist, pulse_dict, plot=False, use_coherence_trajectory=False, purity_downsampling=10, custom_options={}):
        """
        pulse_dict: {
            'optical': f(t) or array,
            'microwave': f(t) or array,
        }
        """
        # Hamiltonian
        H = []
        # H = [self.H0]
        if 'optical' in pulse_dict:
            self._init_hamiltonian(include_excitation=True)
            H.append([self.sig_g0e0, pulse_dict['optical']])
            H.append([self.sig_g1e1, pulse_dict['optical']])
            c_ops = self.c_ops
        else:
            self._init_hamiltonian(include_excitation=False)
            c_ops = []
        H.append(self.H0)

        if 'mw' in pulse_dict:
            H.append([self.sig_g0g1, pulse_dict['mw']])
        if self.sigmaz_noise_fn is not None:
            sigmaz_noise = self.sigmaz_noise_fn(tlist)
            # plt.figure(figsize=(8,4))
            # plt.plot(tlist, sigmaz_noise)
            # plt.xlabel("Time (us)")
            # plt.ylabel("Noise")
            # plt.tight_layout()
            # plt.show()
            H.append([self.sigma_z_g, sigmaz_noise])


        
        options = {'store_states': True,'store_final_state': True, 'progress_bar': False}
        options.update(custom_options)
        ####################################### HEOM solver #######################################
        if self.gamma_DL:
            H_td = qt.QobjEvo(H, tlist=tlist)
            liouvillian = qt.liouvillian(H_td, self.c_ops)
            heom_solver = qt.solver.heom.HEOMSolver(H=liouvillian,
                                        bath=(self.bath_env_approx, self.sigma_z_g),
                                        max_depth=self.max_depth, options=options)
            result = heom_solver.run(self.psi0, tlist, e_ops=self.e_ops)
        ######################################### Regular solver #######################################
        else:
            result = qt.mesolve(H, self.psi0, tlist, c_ops=c_ops, e_ops=self.e_ops, options=options)
            self.psi0 = result.final_state
        
        tlist_downsampled = tlist[::purity_downsampling]
        # if self.memory_tau and use_coherence_trajectory:
        #     coherence_history = analyze_state_purity(tlist_downsampled, result.states[::purity_downsampling], 1/self.t2_star, self.memory_tau, purity_decay_fn=self.purity_decay_fn, purity_inverse_fn=self.purity_inverse_fn, memory_delta_fn=self.memory_delta_fn, plot=False, f_rot=self.mw_fc)
        # else:
        #     coherence_history = None
        if plot:
            plt.figure(figsize=(8,4))
            plt.plot(tlist*1e6, result.expect[0], label='|g0>')
            plt.plot(tlist*1e6, result.expect[1], label='|g1>')
            plt.plot(tlist*1e6, np.abs(result.expect[2]), label='|g0g1>')
            plt.plot(tlist*1e6, np.abs(result.expect[3]), label='|g1g0>')
            if 'optical' in pulse_dict:
                plt.plot(tlist*1e6, result.expect[4], label='|e0>')
                plt.plot(tlist*1e6, result.expect[5], label='|e1>')

            # if coherence_history:
            #     plt.plot(tlist_downsampled*1e6, coherence_history['purity'], label='purity')
            #     plt.plot(tlist_downsampled*1e6, self.purity_decay_fn(coherence_history['forgotten_trajectory']), label='maximum purity')

            plt.xlabel("Time (us)")
            plt.ylabel("Population")
            plt.legend()
            plt.tight_layout()
            plt.show()
        return result, None

    def process_result(self, result, bin_dt, bin_res, reps, use_dark_counts=True, cps=False):
        total_emitted = self.optical_lifetime * (1 - self.branching_ratio) *(result.expect[4] + result.expect[5])
        zpl_photon_rate = total_emitted * self.coupling_efficiency * self.debye_waller_factor * self.non_radiative_factor
        psb_photon_rate = total_emitted * self.coupling_efficiency * (1-self.debye_waller_factor) * self.non_radiative_factor
        zpl_photon_rate = zpl_photon_rate[::bin_res]
        psb_photon_rate = psb_photon_rate[::bin_res]
        zpl_counts = zpl_photon_rate*reps*bin_dt
        psb_counts = psb_photon_rate*reps*bin_dt
        if use_dark_counts:
            zpl_dark_counts = np.random.poisson(self.dark_count_rate*reps*bin_dt, size=len(zpl_counts))
            psb_dark_counts = np.random.poisson(self.dark_count_rate*reps*bin_dt, size=len(zpl_counts))
            zpl_counts += zpl_dark_counts
            psb_counts += psb_dark_counts
        if cps:
            psb_counts = psb_counts/reps/bin_dt
            zpl_counts = zpl_counts/reps/bin_dt
        return zpl_counts, psb_counts

    def initialize_readout(self, amps, length=200e-6, bin_dt=164.0625e-9, bin_res=25, plot=False):
        '''Create a cyclicity propagator for the qubit. Use the result to determine the auto readout.
        inputs:
            amps: amplitude of the cyclicity pulse
            length: length of the cyclicity pulse
            bin_dt: time resolution of the binning
            bin_res: number of points per bin
            plot: whether to plot the results
        '''
        self._init_hamiltonian(include_excitation=True)

        f0_drive = lambda t, args: amps*np.cos(2*np.pi*self.f0*t)
        f1_drive = lambda t, args: amps*np.cos(2*np.pi*self.f1*t)
        H_f0 = [self.H0]
        H_f0.append([self.sig_g0e0, f0_drive])
        H_f0.append([self.sig_g1e1, f0_drive])

        H_f1 = [self.H0]
        H_f1.append([self.sig_g0e0, f1_drive])
        H_f1.append([self.sig_g1e1, f1_drive])

        tlist = np.arange(0, length, bin_dt/bin_res)
        prop_f0 = qt.propagator(H_f0, tlist, c_ops = self.c_ops)
        prop_f1 = qt.propagator(H_f1, tlist, c_ops = self.c_ops)
        self.readout_settings = {
            'prop_f0': prop_f0,
            'prop_f1': prop_f1,
            'tlist': tlist,
            'bin_dt': bin_dt,
            'bin_res': bin_res,
            'amps': amps,
            'length': length,
        }
        

    def auto_readout(self, drive, readout_length, reps=1, use_dark_counts=True, cps=False, plot=False):
        '''Apply the readout propagator to the qubit and measure the population.
        inputs:
            drive: 'f0' or 'f1'
            readout_length: length of the readout pulse
            reps: number of repetitions of the readout pulse
            use_dark_counts: whether to use dark counts
            cps: whether to return counts per second
            plot: whether to plot the results
        outputs:
            zpl_counts: total zpl counts
            psb_counts: total psb counts
        '''
        if self.readout_settings is None:
            raise ValueError("Readout settings not initialized. Please run initialize_readout first.")
        self._init_hamiltonian(include_excitation=True)
        max_t_i = np.argmin(np.abs(self.readout_settings['tlist']-readout_length))
        result = qt.Result(e_ops=self.e_ops, options={'progress_bar': True, 'store_states': True, 'store_final_state': True})
        if drive =='f0':
            for i in range(max_t_i):
                state = qt.vector_to_operator(self.readout_settings['prop_f0'][i] * qt.operator_to_vector(self.psi0))
                t= self.readout_settings['tlist'][i]
                result.add(t, state)
        elif drive == 'f1':
            for i in range(max_t_i):
                state = qt.vector_to_operator(self.readout_settings['prop_f1'][i] * qt.operator_to_vector(self.psi0))
                t= self.readout_settings['tlist'][i]
                result.add(t, state)
        else:
            raise ValueError(f"Invalid drive: {drive}")
        self.psi0 = result.states[-1]
        zpl_counts, psb_counts = self.process_result(result, self.readout_settings['bin_dt'], self.readout_settings['bin_res'], reps, use_dark_counts=False, cps=cps)
        if plot:
            plt.plot(self.readout_settings['tlist'][:max_t_i:self.readout_settings['bin_res']]*1e6, psb_counts)
            plt.xlabel('Time (us)')
            plt.ylabel('Photon Counts')
            plt.grid(True)
            plt.show()
        
        psb_dark_counts = np.random.poisson(self.dark_count_rate*reps*readout_length, size=1)
        zpl_dark_counts = np.random.poisson(self.dark_count_rate*reps*readout_length, size=1)
        return zpl_counts+zpl_dark_counts, psb_counts+psb_dark_counts

    def frequency_sweep(self, fcen, fwidth, amps=3e8, pulse_length=8e-6, reps=1e5,
                             bin_dt=164.0625e-9, bin_res=25, use_dark_counts=True, plot=False, cps=False):
        '''Create a chirped drive pulse and apply it to the qubit
        inputs:
            fcen: center frequency of the chirped pulse
            fwidth: width of the chirped pulse
            amps: amplitude of the chirped pulse
            pulse_length: length of the chirped pulse
            reps: number of repetitions of the chirped pulse
            bin_dt: time resolution of the binning
            bin_res: number of points per bin
            use_dark_counts: whether to use dark counts
            plot: whether to plot the results
            cps: whether to return counts per second
        outputs:
            tbins: time bins
            f_drive: drive frequency
            zpl_counts: zpl counts
            psb_counts: psb counts
        '''
        chirped_drive = lambda t, args: amps*chirp(t, fcen-fwidth/2, pulse_length, fcen+fwidth/2, method='linear')
        tlist = np.arange(0, pulse_length, bin_dt/bin_res)
        f_drive = fcen-fwidth/2 + fwidth*tlist/pulse_length

        result = self.apply_pulse_sequence(tlist, {'optical': chirped_drive}, plot=plot)
        zpl_counts, psb_counts = self.process_result(result, bin_dt, bin_res, reps, use_dark_counts, cps)
        
        tbins = tlist[::bin_res]
        f_drive = f_drive[::bin_res]
        if plot:
            if fwidth:
                plt.plot(f_drive/1e9, psb_counts)
                plt.xlabel('Drive Frequency (GHz)')
                plt.ylabel('Photon Counts')
                plt.grid(True)
                plt.show()
            else:
                plt.plot(tbins*1e6, psb_counts)
                plt.xlabel('Time (us)')
                plt.ylabel('Photon Counts')
                plt.grid(True)
                plt.show()

        return tbins, f_drive, zpl_counts, psb_counts
