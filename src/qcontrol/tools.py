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

def get_bloch_angles(states, tlist, f_rot=0, plot=False):
    # Check if states have 4 levels
    thetas = []
    phis = []
    global_phases = []
    # Loop over states
    for i, state in enumerate(states):
        # Project full state to qubit subspace
        rho_qubit = state
        if f_rot:
            U = (-1j * 2*np.pi*f_rot * tlist[i] / 2 * qt.sigmaz()).expm()
            rho_qubit = U * rho_qubit

        # Extract Bloch vector
        x = qt.expect(qt.sigmax(), rho_qubit)
        y = qt.expect(qt.sigmay(), rho_qubit)
        z = qt.expect(qt.sigmaz(), rho_qubit)
        thetas.append(np.arccos(z))
        phis.append(np.arctan2(y, x))
        global_phases.append(np.angle(rho_qubit.full()[0,0]))
    if plot:
        plt.figure()
        plt.plot(tlist, thetas, label='theta')
        plt.plot(tlist, phis, label='phi')
        plt.plot(tlist, global_phases, label='global phase')
        plt.legend()
        plt.show()
    return np.array(thetas), np.array(phis), np.array(global_phases)

def get_bloch_vector(states, tlist, f_rot=0, plot=False, plot_abs=False):
    # Check if states have 4 levels
    bloch_vectors = []
    global_phases = []
    # Loop over states
    for i, state in enumerate(states):
        # Project full state to qubit subspace
        rho_qubit = state
        if f_rot:
            U = (-1j * 2*np.pi*f_rot * tlist[i] / 2 * qt.sigmaz()).expm()
            rho_qubit = U * rho_qubit

        # Extract Bloch vector
        x = qt.expect(qt.sigmax(), rho_qubit)
        y = qt.expect(qt.sigmay(), rho_qubit)
        z = qt.expect(qt.sigmaz(), rho_qubit)
        bloch_vectors.append(np.array([x, y, z]))
        global_phases.append(np.angle(rho_qubit.full()[0,0]))
    bloch_vectors = np.array(bloch_vectors)
    global_phases = np.array(global_phases)
    if plot:
        if plot_abs:
            plt.figure()
            plt.plot(tlist, np.abs(bloch_vectors[:,0]), label='x')
            plt.plot(tlist, np.abs(bloch_vectors[:,1]), label='y')
            plt.plot(tlist, np.abs(bloch_vectors[:,2]), label='z')
            plt.plot(tlist, np.linalg.norm(bloch_vectors, axis=1), label='norm')
            # plt.plot(tlist, global_phases, label='global phase')
            plt.legend()
        else:
            plt.figure()
            plt.plot(tlist, bloch_vectors[:,0], label='x')
            plt.plot(tlist, bloch_vectors[:,1], label='y')
            plt.plot(tlist, bloch_vectors[:,2], label='z')
            plt.plot(tlist, np.linalg.norm(bloch_vectors, axis=1), label='norm')
            # plt.plot(tlist, global_phases, label='global phase')
            plt.legend()
    return bloch_vectors, global_phases

def __smooth_square_pulse(self, t, start, width, t_ramp, amplitude, style="sine"):
    """Smooth square pulse using sine ramping."""
    if style == "sine":
        pulse = np.zeros_like(t)
        # Rising edge
        idx_rise = (t >= start) & (t < start + t_ramp)
        pulse[idx_rise] = amplitude * 0.5 * (1 - np.cos(np.pi * (t[idx_rise] - start) / t_ramp))
        # Flat top
        idx_flat = (t >= start + t_ramp) & (t < start + width - t_ramp)
        pulse[idx_flat] = amplitude
        # Falling edge
        idx_fall = (t >= start + width - t_ramp) & (t < start + width)
        pulse[idx_fall] = amplitude * 0.5 * (1 + np.cos(np.pi * (t[idx_fall] - (start + width - t_ramp)) / t_ramp))
        # Elsewhere remains zero
        return pulse
    elif style == "sin2":
        pulse = np.zeros_like(t)
        t_start = start
        t_end = start + width

        t_rise = (t >= t_start) & (t < t_start + t_ramp)
        t_flat = (t >= t_start + t_ramp) & (t < t_end - t_ramp)
        t_fall = (t >= t_end - t_ramp) & (t < t_end)

        # Rise: sin² ramp
        pulse[t_rise] = amplitude * np.sin((np.pi / 2) * (t[t_rise] - t_start) / t_ramp) ** 2
        # Flat: constant amplitude
        pulse[t_flat] = amplitude
        # Fall: mirrored sin²
        pulse[t_fall] = amplitude * np.sin((np.pi / 2) * (1 - (t[t_fall] - (t_end - t_ramp)) / t_ramp)) ** 2

        return pulse

def gaussian_decay(x, tau=1):
    '''Returns a gaussian whose 1/e time is tau'''
    return np.exp(-x**2/tau**2)

def inverse_gaussian_decay(y, tau=1):
    '''Returns the x at which the gaussian decay is y'''
    return tau*np.sqrt(-np.log(y))

def exponential_decay(x, tau=1):
    '''Returns an exponential decay whose 1/e time is tau'''
    return np.exp(-x/tau)

def delta_exponential_decay(x, tau=1):
    '''Returns the derivative of the exponential decay'''
    return -1/tau*x

def analyze_state_purity(tlist, states, correlation_fn, auto_correlated_fn, plot=False, f_rot=0.0):
    '''Analyzes the coherence of a sequence of states given the rates of dephasing and memory loss'''
    dt = tlist[1] - tlist[0]
    g0 = qt.basis(2, 0)
    g1 = qt.basis(2, 1)
    # thetas, phis, global_phases = get_bloch_angles(states, tlist, f_rot)
    bloch_vectors, global_phases = get_bloch_vector(states, tlist, f_rot, plot=plot)
    history = {
        'purity': np.zeros(len(tlist)),
        'trajectory': np.zeros((len(tlist), 3), dtype=complex),
        'dephasing_mag': np.zeros(len(tlist)),
        'forgotten_trajectory': np.zeros(len(tlist)),
        'total_trajectory': np.zeros(len(tlist))
    }
    corr_0 = correlation_fn(0)[0]
    dt = np.diff(tlist, prepend=tlist[0]-tlist[1])
    corr_dt = correlation_fn(dt)
    plt.plot(tlist, corr_dt/corr_0)
    plt.show()

    total_forgotten_trajectory = 0
    dephasing_trajectory = np.array([0+0j,0+0j,0+0j])
    p21s = np.abs(qt.expect(g1*g0.dag(), states))
    for i, state in enumerate(states):
        if np.linalg.norm(dephasing_trajectory) > 0:
            remembered_ratio = 0.995 #corr_dt[i]/corr_0
            total_forgotten_trajectory += (1-remembered_ratio)*np.linalg.norm(dephasing_trajectory)
            dephasing_trajectory *= remembered_ratio

        bloch_vec = bloch_vectors[i]
        global_phase=global_phases[i]
        p21 = p21s[i]

        dephasing_mag = (2*p21)*dt[i]#*corr_0
        # print(f'dephasing mag: {dephasing_mag}, p21: {p21}, corr_0: {corr_0}, dt: {dt[i]}')

        dephasing_trajectory += dephasing_mag*bloch_vec*np.exp(2j*global_phase)

        total_trajectory = np.linalg.norm(dephasing_trajectory) + total_forgotten_trajectory

        history['trajectory'][i] = dephasing_trajectory
        history['dephasing_mag'][i] = dephasing_mag
        history['forgotten_trajectory'][i] = total_forgotten_trajectory
        history['total_trajectory'][i] = total_trajectory
    
    history['purity'][:] = np.exp(-2*auto_correlated_fn(history['total_trajectory']))*0.5
    if plot:
        plt.figure()
        plt.plot(tlist, history['purity'], label='purity')
        plt.plot(tlist, np.exp(-2*auto_correlated_fn(history['forgotten_trajectory']))/2, label='maximum purity')
        # plt.plot(tlist, np.abs(trajectory_history), label='dephasing trajectory')
        # plt.plot(tlist, total_trajectory_history, label='total trajectory')
        plt.legend()
        plt.show()
        # Make a 3d plot of the trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.real(history['trajectory'][:,0]), np.real(history['trajectory'][:,1]), np.real(history['trajectory'][:,2]), c=tlist, cmap='viridis')
        ax.scatter(np.imag(history['trajectory'][:,0]), np.imag(history['trajectory'][:,1]), np.imag(history['trajectory'][:,2]), c=tlist, cmap='magma')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    return history
