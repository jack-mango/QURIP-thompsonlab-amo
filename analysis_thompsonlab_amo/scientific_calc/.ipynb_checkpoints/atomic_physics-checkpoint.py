import numpy as np


bohr_magneton = 9.274009994e-24 / 1e4  # J/G
planck_const = 6.62607015e-34  # J.s


def calc_gJ(J, L, S):
    return 1 + ((J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1)))


def calc_gF(F, I, J, L, S):
    return calc_gJ(J, L, S) * ((F * (F + 1) - I * (I + 1) + J * (J + 1)) / (2 * F * (F + 1)))


def yb174_zeeman_splitting(magnetic_field, J, L, S):
    lande_factor = calc_gJ(J, L, S)
    return lande_factor * bohr_magneton * magnetic_field / planck_const


def yb171_zeeman_splitting(magnetic_field, F, J, L, S):
    I = 1 / 2
    lande_factor = calc_gF(F, I, J, L, S)
    return lande_factor * bohr_magneton * magnetic_field / planck_const
