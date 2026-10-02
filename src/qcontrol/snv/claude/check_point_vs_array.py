import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from pulseseq.sequencing.waveform import AnalogPulse, Apodization, Shape
from qcontrol.snv.pulseseq_interconnect import synthesize_analog_pulse, make_analog_pulse_time_array
sp = 1 / 6.144e9
for apod in Apodization:
    for shape in Shape:
        p = AnalogPulse(length=100e-9, apodization=apod, apodization_length=10e-9, padding_length=20e-9,
                        shift=0, amplitude=0.7, frequency=1.94e9, frequency_chirp=0, shape=shape,
                        S21_correct=True, w_3db=2*np.pi*2e9, phase_offset=np.nan, name="")
        tau = make_analog_pulse_time_array(p, sp, 50e-9)
        full = synthesize_analog_pulse(p, tau, at_time=50e-9)
        point = jax.vmap(lambda t: synthesize_analog_pulse(p, t, at_time=50e-9, dt=sp))(tau)
        g = jax.grad(lambda a: jnp.sum(synthesize_analog_pulse(p, tau, at_time=a)))(50e-9)
        print(f"{apod!s:24s} {shape!s:16s} max|full-point|={float(jnp.max(jnp.abs(full-point))):.2e} "
              f"max|w|={float(jnp.max(jnp.abs(full))):.3f} grad_finite={bool(jnp.isfinite(g))}")
