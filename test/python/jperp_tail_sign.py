"""Signed, configuration-by-configuration oracle for the transverse measurements (one rank)."""

import re
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from triqs.operators import n
from triqs.utility import mpi
from triqs_ctseg import Solver
from triqs_ctseg.postprocessing import _assemble_density_tail_moments


assert mpi.size == 1  # visualize_config writes rank zero's trajectory
beta, U, h = 4.0, 1.0, 0.13
gf_struct = [('up', 1), ('down', 1)]
up, down = n('up', 0), n('down', 0)
times = np.linspace(0, beta, 32)  # 31 distinct periodic points
phase = 2*np.pi*times/beta
j0, a = 0.1, 0.55
j = j0*(1 + a*np.cos(phase))
# For this input the discrete circular autocorrelation is known at every node.
kernel = j0**2*(1 + 0.5*a**2*np.cos(phase))
cycles = 1600


def run(bins, record):
    s = Solver(beta=beta, gf_struct=gf_struct, n_tau=17, n_tau_bosonic=len(times))
    s.Jperp_tau.data[:, 0, 0] = j
    s.gf_struct, s.h_int = gf_struct, U*up*down
    s.h_loc0_mat = [np.array([[-h]]), np.array([[h]])]
    s.solve(h_int=s.h_int, h_loc0=-h*(up-down), n_tau_G=17, n_tau_chi2=bins,
            length_cycle=20, n_warmup_cycles=100, n_cycles=cycles, random_seed=23,
            measure_G_tau=False, measure_nn_static=True, measure_Sperp_asym_tau=record,
            visualize_config=record, verbosity=0)
    return s


with TemporaryDirectory() as directory, chdir(directory):
    s = run(len(times), True)
    configs = Path('configuration.txt').read_text().split('Configuration is')[1:]

# Delta=0, J>0: each spin line contributes exactly one minus sign. Recover
# signs independently from k, not from any signed measurement under test.
assert len(configs) == cycles
vertices = re.compile(r'S_minus : .*?n = (\d+).*?S_plus : .*?n = (\d+)')
ticks = 2**64
z = count = 0
ksum = 0.0
hist = np.zeros((2, len(times)))
dt = times[1]-times[0]
for config in configs:
    lines = vertices.findall(config)
    sign = (-1)**len(lines)
    z += sign
    count += sign*len(lines)
    for sm, sp in lines:
        lag = int(sm)-int(sp)
        for orientation, tick in enumerate((lag, -lag)):
            d = beta*((tick % ticks)/(ticks-1))
            jd = np.interp(d, times, j)
            hist[orientation, int(np.floor(d/dt + 0.5))] += 2*sign/jd
            if orientation == 0:
                ksum += sign*np.interp(d, times, kernel)/jd

assert 0 < z/cycles < 0.99, "fixture must visit configurations of both signs"
np.testing.assert_allclose(s.results.average_sign, z/cycles, rtol=0, atol=1e-11)
hist /= -beta*z*dt
hist[:, [0, -1]] *= 2
for gf, ref in zip((s.results.Sminus_Splus_tau, s.results.Splus_Sminus_tau), hist):
    np.testing.assert_allclose(gf.data[:, 0, 0], ref, rtol=0, atol=1e-11)

A, B = -count/(beta*z), -ksum/(beta*z)
# Same trajectory with another chi mesh and both histograms disabled.
no_hist = run(13, False)
assert no_hist.results.Sperp_tau is None and no_hist.results.Sminus_Splus_tau is None
assert no_hist.results.Splus_Sminus_tau is None
for solver in (s, no_hist):
    res = solver.results
    np.testing.assert_allclose([res.jperp_int_J_chi_xx, res.jperp_J_chi_xx_J], [A, B], rtol=0, atol=1e-11)
    moments = _assemble_density_tail_moments(solver)
    for spin, other in [('up', 'down'), ('down', 'up')]:
        n_other = res.densities[other][0]
        sigma1 = U**2*n_other*(1-n_other) + (j[0]+B)/2 - 2*U*A
        np.testing.assert_allclose(moments['Sigma_moments'][spin][1, 0, 0], sigma1, rtol=0, atol=1e-11)
print(f"average sign {z/cycles:.5f}; signed reference A={A:.10f}, B={B:.10f}")
