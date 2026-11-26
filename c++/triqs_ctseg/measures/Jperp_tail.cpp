// This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "./Jperp_tail.hpp"
#include <triqs/gfs/transform/fourier_common.hpp>

namespace triqs_ctseg::measures {

  Jperp_tail::Jperp_tail(gf<imtime> const &J, configuration_t const &config, results_t &results)
     : J{J}, config{config}, results{results}, kernel{J.mesh()} {
    // K_r = sum_i J_i J_{i+r} / N on the input mesh, not the measurement mesh.
    // Fold the two half-weight endpoint samples before taking the circular FFT.
    int n = J.mesh().size() - 1;
    nda::array<dcomplex, 2> time(n, 1), freq(n, 1);
    for (int i = 0; i < n; ++i) time(i, 0) = real(J.data()(i, 0, 0));
    time(0, 0) = 0.5 * real(J.data()(0, 0, 0) + J.data()(n, 0, 0));
    _fourier_base(time, freq, 1, &n, 1, FFTW_FORWARD);
    for (auto &z : freq) z = std::norm(z);
    _fourier_base(freq, time, 1, &n, 1, FFTW_BACKWARD);
    for (int i = 0; i < n; ++i) kernel.data()(i) = real(time(i, 0)) / (double(n) * n);
    kernel.data()(n) = kernel.data()(0);
  }

  void Jperp_tail::accumulate(double s) {
    Z += s;
    order += s * config.Jperp_order();
    for (auto const &line : config.Jperp_list) {
      double d = double(line.tau_Sminus - line.tau_Splus);
      weighted_lines += s * kernel(d) / real(J(d)(0, 0));
    }
  }

  void Jperp_tail::collect_results(mpi::communicator const &c) {
    double norm = -J.mesh().beta() * mpi::all_reduce(Z, c);
    results.jperp_int_J_chi_xx = mpi::all_reduce(order, c) / norm;
    results.jperp_J_chi_xx_J   = mpi::all_reduce(weighted_lines, c) / norm;
  }

} // namespace triqs_ctseg::measures
