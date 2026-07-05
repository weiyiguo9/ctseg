// This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later
// See LICENSE in the root of this distribution for details.

#pragma once

#include "../configuration.hpp"
#include "../work_data.hpp"

namespace triqs_ctseg::measures {

  inline double static_prefactor(work_data_t const &wdata, configuration_t const &config, int color, tau_t tau) {
    double value = 0;
    for (auto const &[c, sl] : itertools::enumerate(config.seglists)) {
      if (c != color) value += wdata.U(c, color) * n_tau(tau, sl);
    }
    return value;
  }

  inline double retarded_density_prefactor(work_data_t const &wdata, configuration_t const &config, int color,
                                           tau_t tau) {
    if (not wdata.has_Dt) return 0;

    double value = 0;
    for (auto const &[c, sl] : itertools::enumerate(config.seglists))
      value -= K_overlap(sl, tau, false, wdata.Kprime, c, color);
    return value;
  }

  inline double retarded_density_jump(work_data_t const &wdata, int color) {
    return wdata.has_Dt ? -2 * real(wdata.Kprime(0)(color, color)) : 0;
  }

  // The endpoint field used in F excludes the static screening shift A_ac n_c.
  // Restore it only for the source-induced field measured by dyn_corr.
  inline double retarded_density_source_field(work_data_t const &wdata, configuration_t const &config, int color,
                                              tau_t tau) {
    if (not wdata.has_Dt) return 0;

    // The bosonic field is periodic; use the occupied beta side at the boundary.
    if (tau == tau_t::zero()) tau = tau_t::beta();
    double value = retarded_density_prefactor(wdata, config, color, tau);
    for (auto const &[c, sl] : itertools::enumerate(config.seglists)) {
      double const A = real(wdata.Kprime.data()(wdata.Kprime.mesh().size() - 1, c, color)
                            - wdata.Kprime.data()(0, c, color));
      value += A * n_tau(tau, sl);
    }
    return value;
  }

  inline double retarded_density_prefactor_integral(work_data_t const &wdata, configuration_t const &config, int color,
                                                    tau_t tau_left, tau_t tau_right) {
    if (not wdata.has_Dt) return 0;

    double value = 0;
    for (auto const &[c, sl] : itertools::enumerate(config.seglists)) {
      auto Ks = slice_target_to_scalar(wdata.K, c, color);
      double const A = real(wdata.Kprime.data()(wdata.Kprime.mesh().size() - 1, c, color)
                            - wdata.Kprime.data()(0, c, color));
      for (auto const &seg : sl) {
        value += real(Ks(double(seg.tau_c - tau_right)) - Ks(double(seg.tau_c - tau_left))
                      - Ks(double(seg.tau_cdag - tau_right)) + Ks(double(seg.tau_cdag - tau_left)));
        value += A * overlap(seg, segment_t{tau_left, tau_right});
      }
    }
    return value;
  }

} // namespace triqs_ctseg::measures
