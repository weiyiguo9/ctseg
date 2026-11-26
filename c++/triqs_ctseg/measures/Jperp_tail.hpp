// This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once
#include "../configuration.hpp"
#include "../results.hpp"

namespace triqs_ctseg::measures {

  struct Jperp_tail {
    gf<imtime> const &J;
    configuration_t const &config;
    results_t &results;
    gf<imtime, scalar_real_valued> kernel;
    double Z = 0, order = 0, weighted_lines = 0;

    Jperp_tail(gf<imtime> const &J, configuration_t const &config, results_t &results);
    void accumulate(double s);
    void collect_results(mpi::communicator const &c);
  };

} // namespace triqs_ctseg::measures
