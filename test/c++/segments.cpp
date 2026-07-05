// Copyright (c) 2022--present, The Simons Foundation
// Copyright (c) 2022--present, Max Planck Institute for Polymer Research, Mainz, Germany
// This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later
// See LICENSE in the root of this distribution for details.

#include <iostream>

#include <cmath>
#include <triqs/test_tools/arrays.hpp>
#include <triqs_ctseg/tau_t.hpp>
#include <triqs_ctseg/configuration.hpp>
#include <triqs_ctseg/measures/dyn_density_prefactors.hpp>

using namespace triqs_ctseg;
using vs_t = std::vector<segment_t>;

double beta      = 10;
double precision = 1.e-13;

tau_t make_tau(double x) { return tau_t{x}; } //uint64_t((x / tau_t::get_beta()) * double(tau_t::n_max))};}
tau_t make_tau(tau_t x) { return x; }

segment_t S(auto x, auto y) { return {make_tau(x), make_tau(y)}; }

// ------------------------------

TEST(segment, overlap) {
  tau_t::set_beta(beta);
  EXPECT_NEAR(overlap(S(3, 2), S(2.5, 1.5)), 0.5, precision);
  // noncyclic, cyclic
  EXPECT_NEAR(overlap(S(3, 2), S(2.5, 10)), 0.5, precision);
  // cyclic, noncyclic
  EXPECT_NEAR(overlap(S(2.5, 10), S(3, 2)), 0.5, precision);
}

// ------------------------------

TEST(segment, flip) {
  tau_t::set_beta(beta);
  auto v  = vs_t{S(4, 3), S(2, 1)};
  auto vf = vs_t{S(3, 2), S(1, 4)};
  //std::cout  << flip(v) << std::endl;
  //std::cout  << vf << std::endl;
  EXPECT_EQ(flip(v), vf);
  EXPECT_EQ(flip(flip(v)), v);
}

// ------------------------------

TEST(segment, lower_bound) {
  tau_t::set_beta(beta);

  auto tau1 = make_tau(3);
  auto v    = vs_t{S(tau1, 2.5), S(2, 1)};

  EXPECT_EQ(lower_bound(v, tau1) - v.begin(), 0);

  EXPECT_EQ(lower_bound(v, tau_t{2.0}) - v.begin(), 1);
  EXPECT_EQ(lower_bound(v, tau_t{2.1}) - v.begin(), 1);
  EXPECT_EQ(lower_bound(v, tau_t{1.9}) - v.begin(), 2);
}

// ------------------------------

// FIXME : more case, with cyclic ...
TEST(segment, is_insertable) {
  tau_t::set_beta(beta);

  auto v = vs_t{S(3, 2.5), S(2, 1)};

  EXPECT_TRUE(is_insertable_into(S(2.4, 2.3), v));
  EXPECT_FALSE(is_insertable_into(S(2.1, 2.3), v));
  EXPECT_FALSE(is_insertable_into(S(2.5, 2.3), v));
  EXPECT_FALSE(is_insertable_into(S(2.1, 2), v));

  EXPECT_TRUE(is_insertable_into(S(0.5, 5), v));
}

TEST(segment, retarded_density_source_field) {
  tau_t::set_beta(beta);

  constr_params_t construction;
  construction.beta          = beta;
  construction.gf_struct     = {{"up", 1}, {"down", 1}};
  construction.n_tau         = 11;
  construction.n_tau_bosonic = 11;

  solve_params_t solve;
  solve.h_int  = triqs::operators::n("up", 0) * triqs::operators::n("down", 0);
  solve.h_loc0 = -0.5 * (triqs::operators::n("up", 0) + triqs::operators::n("down", 0));
  params_t params{construction, solve};

  inputs_t inputs;
  inputs.Delta  = block_gf<imtime>({beta, Fermion, construction.n_tau}, construction.gf_struct);
  inputs.D0t    = make_block2_gf<imtime>({beta, Boson, construction.n_tau_bosonic}, construction.gf_struct);
  inputs.Jperpt = gf<imtime>({beta, Boson, construction.n_tau_bosonic}, {1, 1});
  inputs.Delta()  = 1;
  inputs.D0t()    = 0.2;
  inputs.Jperpt() = 0;

  work_data_t wdata{params, inputs, mpi::communicator{}};
  configuration_t config{2};
  for (auto const &seg : {S(8, 4), S(2, 8), segment_t::full_line()}) {
    config.seglists[0] = {seg};
    double const field = 0.2 * double(seg.length());
    for (int color = 0; color < 2; ++color) {
      for (double t : {0.0, 2.0, 4.0, 5.0, 8.0, 10.0})
        EXPECT_NEAR(measures::retarded_density_source_field(wdata, config, color, make_tau(t)), field, precision);
      EXPECT_NEAR(measures::retarded_density_prefactor_integral(wdata, config, color, make_tau(9), make_tau(3)),
                  field * 6, precision);
      EXPECT_NEAR(measures::retarded_density_prefactor_integral(wdata, config, color, tau_t::beta(), tau_t::zero()),
                  field * beta, precision);
    }
  }
}

// TEST OVERLAP
//

MAKE_MAIN;
