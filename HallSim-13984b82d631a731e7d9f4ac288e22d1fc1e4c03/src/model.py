# HallSim physics model
#
# Hall thruster similarity-parameter model (Lafleur & Chabert, 2025).
# Pure numerical core: no Streamlit or other UI code lives here, so this
# module can be imported and driven independently of the web app.
#
# Naming convention: a star superscript (e.g. v_star) marks a characteristic
# variable; a bar (e.g. Gamma_bar) marks a normalised variable.

from dataclasses import dataclass, replace
import pathlib

import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as optimize

# ============================================================
# Physical constants
# ============================================================
AMU = 1.66053906660e-27     # [kg] atomic mass unit
K_B = 1.380649e-23          # [J/K] Boltzmann constant
E_CHARGE = 1.602176634e-19  # [C] Elementary charge
M_E = 9.10938356e-31        # [kg] Electron rest mass

# Table 1 default operating / material parameters (xenon on boron nitride)
B_MAX = 20e-3          # [T] peak magnetic field
C_MAG = 4              # [-] magnetic field shape parameter
T_G = 700              # [K] neutral gas temperature
M_XE = 131.293 * AMU   # [kg] xenon ion mass

# Channel-wall material (boron nitride), Eq. (12)
A_SEE = 0.207   # [eV^-b]
B_SEE = 0.549   # [-]

# Table 7: characteristic rate coefficients for xenon, evaluated at T_e = epsilon_iz
EPSILON_IZ = 12.13   # [eV] xenon ionization threshold energy

N_ZBAR_DEFAULT = 200   # default number of output points along the channel

# ============================================================
# Rate-coefficient data (xenon), loaded once from the bundled asset
# ============================================================
ASSET_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets"


def _load_rate_coefficients():
    data = pd.read_excel(ASSET_DIR / 'rate_coefficients.xlsx', header=0, skiprows=[0])
    data.columns = ['idx', 'T_e_data',
                     'K_iz_Xe', 'K_m_Xe', 'epsilon_c_Xe',
                     'K_iz_Kr', 'K_m_Kr', 'epsilon_c_Kr',
                     'K_iz_Ar', 'K_m_Ar', 'epsilon_c_Ar']
    return data


def _cubic_interp(data, column):
    return interpolate.interp1d(data['T_e_data'], data[column],
                                 kind='cubic', bounds_error=False,
                                 fill_value='extrapolate')


_rate_coefficients_data = _load_rate_coefficients()

# Cubic interpolants for the xenon rate coefficients and collisional energy cost
K_iz_func = _cubic_interp(_rate_coefficients_data, 'K_iz_Xe')
K_m_func = _cubic_interp(_rate_coefficients_data, 'K_m_Xe')
epsilon_c_func = _cubic_interp(_rate_coefficients_data, 'epsilon_c_Xe')

K_iz_star = float(K_iz_func(EPSILON_IZ))   # ~5.45e-14 m^3/s
K_m_star = float(K_m_func(EPSILON_IZ))     # ~2.37e-13 m^3/s


# ============================================================
# Run parameters
# ============================================================
@dataclass
class ThrusterParams:
    r_1: float           # [m] inner channel radius
    r_2: float           # [m] outer channel radius
    L_ch: float          # [m] channel length
    Q_m: float            # [kg/s] mass flow rate
    I_d: float            # [A] discharge current
    verification_mode: bool = False
    N_zbar: int = N_ZBAR_DEFAULT


def discharge_current_from_Ibar(I_bar_d, Q_m):
    """Normalised discharge current, Eq. (27): I_bar_d = M*I_d/(e*Q_m)."""
    return I_bar_d * E_CHARGE * Q_m / M_XE


def Ibar_from_discharge_current(I_d, Q_m):
    """Inverse of discharge_current_from_Ibar, Eq. (27): I_bar_d = M*I_d/(e*Q_m)."""
    return I_d * M_XE / (E_CHARGE * Q_m)


def _mode_settings(verification_mode):
    """Mode-dependent physics switches and anode boundary conditions."""
    if verification_mode:
        # Section 3.1: anomalous transport and wall losses off (Table 1)
        return dict(h_R=0, delta_an=0, f_m_pinned=1, f_eps_pinned=2,
                    Gamma_bar_0=0.01, G_bar_0=0.001)   # Table 1: f_m=1, f_eps=2

    # Full model (Sections 3.2 / 4)
    Gamma_bar_0 = 0.005   # Section 2.5: physical anode momentum flux boundary condition
    G_bar_0 = Gamma_bar_0 * np.sqrt(K_B * T_G / (E_CHARGE * EPSILON_IZ))
    return dict(h_R=0.4, delta_an=3.5e-3, f_m_pinned=None, f_eps_pinned=None,
                Gamma_bar_0=Gamma_bar_0, G_bar_0=G_bar_0)


def f_ce_profile(z, L_ch):
    """Normalised magnetic field profile, Eq. (45)."""
    z_bar = z / L_ch
    return np.exp(-C_MAG * (z_bar - 1) ** 2)


def rate_fractions(T_e, verification_mode, f_m_pinned=None, f_eps_pinned=None):
    """Normalised rate-coefficient fractions, Eqs. (30), (31), (34)."""
    f_iz = float(K_iz_func(T_e)) / K_iz_star   # Eq. (30)

    if verification_mode:
        f_m, f_eps = f_m_pinned, f_eps_pinned
    else:
        f_m = float(K_m_func(T_e)) / K_m_star             # Eq. (31)
        f_eps = float(epsilon_c_func(T_e)) / EPSILON_IZ   # Eq. (34)

    return f_iz, f_m, f_eps


def sigma_from_Te(T_e):
    """Secondary electron emission coefficient, Eqs. (12)-(14)."""
    sigma_see = A_SEE * T_e ** B_SEE            # Eq. (12)
    sigma_scl = 1 - 8.3 * np.sqrt(M_E / M_XE)   # Eq. (13)
    return min(sigma_see, sigma_scl)            # Eq. (14)


def solve_model(params: ThrusterParams) -> dict:
    """Integrate the anode (z_bar=0) to exit (z_bar=1) ODE system for `params`."""
    mode = _mode_settings(params.verification_mode)
    h_R, delta_an = mode["h_R"], mode["delta_an"]
    f_m_pinned, f_eps_pinned = mode["f_m_pinned"], mode["f_eps_pinned"]
    Gamma_bar_0, G_bar_0 = mode["Gamma_bar_0"], mode["G_bar_0"]

    r_1, r_2, L_ch = params.r_1, params.r_2, params.L_ch
    Q_m, I_d = params.Q_m, params.I_d

    # Derived parameters (z-independent).
    # Note: omega_ce, nu_an, u_B, nu_m are NOT constants - they depend on z (via B).
    A_ch = np.pi * (r_2**2 - r_1**2)         # channel cross-section area [m^2]
    delta_r = r_2 - r_1                       # channel width [m]
    v_g = np.sqrt(K_B * T_G / M_XE)           # neutral thermal speed [m/s]
    v_star = np.sqrt(E_CHARGE * EPSILON_IZ / M_XE)   # characteristic ion speed [m/s]
    omega_max = E_CHARGE * B_MAX / M_E        # peak electron cyclotron freq [rad/s]
    Gamma_m = Q_m / (M_XE * A_ch)             # max ion flux (full ionisation), Eq. (15)
    Gamma_d = I_d / (E_CHARGE * A_ch)         # discharge flux, text after Eq. (17)

    Gamma_0 = Gamma_bar_0 * Gamma_d
    G_0 = G_bar_0 * v_star * Gamma_d

    Te_guess = [EPSILON_IZ]   # warm-start cache for fsolve, local to this solve

    def power_balance(T_e, Gamma, G, f_ce):
        T_e = max(float(np.atleast_1d(T_e)[0]), 0)

        rate_fractions(T_e, params.verification_mode, f_m_pinned, f_eps_pinned)
        sigma = sigma_from_Te(T_e)

        omega_ce = omega_max * f_ce             # Eq. (29)
        u_B = np.sqrt(E_CHARGE * T_e / M_XE)    # Bohm velocity
        K_iz = K_iz_func(T_e)
        K_m = K_m_star
        epsilon_c = 2 * EPSILON_IZ

        # Eq. (20)
        LHS = (M_E * v_g * omega_ce**2 * G**2 * (Gamma_d - Gamma)**2) / (
            E_CHARGE * Gamma**4 * (K_m * (Gamma_m - Gamma) + delta_an * v_g * omega_ce))

        RHS = (((Gamma_m - Gamma) * K_iz * epsilon_c) / v_g
               + (2 * h_R * u_B * T_e / delta_r) * (2 / (1 - sigma)
                  + np.log((1 - sigma) * np.sqrt(M_XE / (2 * np.pi * M_E)))))

        return LHS - RHS

    def diff_eqns(z, y):
        Gamma, G = y[0], y[1]
        f_ce = f_ce_profile(z, L_ch)

        guess = Te_guess[0]
        T_e = optimize.fsolve(power_balance, guess, args=(Gamma, G, f_ce))[0]
        Te_guess[0] = T_e

        omega_ce = omega_max * f_ce
        u_B = np.sqrt(E_CHARGE * T_e / M_XE)   # Bohm velocity
        K_iz = K_iz_func(T_e)
        K_m = K_m_star

        dy_dx = np.zeros(2)

        # Xe flux - Eq. (18)
        dy_dx[0] = ((Gamma**2 * (Gamma_m - Gamma) * K_iz) / (v_g * G)
                    - (2 * h_R * Gamma**2 * u_B) / (G * delta_r))

        # Xe momentum flux - Eq. (19)
        dy_dx[1] = ((M_E * v_g * omega_ce**2 * (Gamma_d - Gamma)) /
                    (M_XE * (K_m * (Gamma_m - Gamma) + delta_an * v_g * omega_ce))
                    - (2 * h_R * Gamma * u_B) / delta_r)

        return dy_dx

    z_bar_array = np.linspace(0, 1, params.N_zbar)
    z_array = z_bar_array * L_ch
    y_0 = [Gamma_0, G_0]

    sol = integrate.solve_ivp(
        diff_eqns, t_span=[0, L_ch],
        y0=y_0, method='DOP853', t_eval=z_array, rtol=1e-10, atol=1e-6)

    Gamma_array = sol.y[0]
    G_array = sol.y[1]

    G_bar_array = G_array / (v_star * Gamma_d)

    f_ce_array = f_ce_profile(z_array, L_ch)

    Te_guess[0] = EPSILON_IZ   # reset warm start before the diagnostic re-solve pass

    T_e_array = []
    for i in range(len(z_array)):
        sol_i = optimize.fsolve(power_balance, Te_guess[0],
                                 args=(Gamma_array[i], G_array[i], f_ce_array[i]))[0]
        Te_guess[0] = sol_i
        T_e_array.append(sol_i)
    T_e_array = np.array(T_e_array)

    T_bar_e_array = T_e_array / EPSILON_IZ   # Eq. (28) [eV]

    f_m_array = np.array([
        rate_fractions(T_e, params.verification_mode, f_m_pinned, f_eps_pinned)[1]
        for T_e in T_e_array])

    v_i_array = G_array / Gamma_array             # ion velocity [m/s]
    n_array = Gamma_array / v_i_array              # plasma density [m^-3]
    n_g_array = (Gamma_m - Gamma_array) / v_g       # neutral density, Eq. (15)

    # --- Axial electric field, Eqs. (16)-(17) ---
    omega_ce_array = omega_max * f_ce_array
    nu_an_array = delta_an * omega_ce_array
    K_m_array = K_m_star * f_m_array
    mu_array = E_CHARGE * (n_g_array * K_m_array + nu_an_array) / (M_E * omega_ce_array**2)
    E_array = (Gamma_d - Gamma_array) / (n_array * mu_array)

    # --- Electrostatic potential (E = -dphi/dz, cathode reference phi(L)=0) ---
    E_func = interpolate.interp1d(z_array, E_array, kind='cubic',
                                   bounds_error=False, fill_value='extrapolate')

    def phi_drop(z_value):
        val, _ = integrate.quad(E_func, 0, z_value)
        return val

    phi_drop_array = np.array([phi_drop(z_value) for z_value in z_array])
    phi_array = phi_drop_array[-1] - phi_drop_array

    # --- Wall sheath potential drop / heat flux, Eqs. (9)-(11), (37) ---
    sigma_array = np.array([sigma_from_Te(T_bar_e) for T_bar_e in T_bar_e_array])

    u_B_array = np.sqrt(E_CHARGE * T_e_array / M_XE)
    nu_iw_array = 2 * h_R * u_B_array / delta_r
    nu_ew_array = nu_iw_array / (1 - sigma_array)
    eps_w_array = T_e_array * (2 + (1 - sigma_array)
                  * np.log((1 - sigma_array) * np.sqrt(M_XE / (2 * np.pi * M_E))))
    q_integrand = n_array * nu_ew_array * eps_w_array
    q_func = interpolate.interp1d(z_array, q_integrand, kind='cubic',
                                   bounds_error=False, fill_value='extrapolate')
    q_int, _ = integrate.quad(q_func, 0, L_ch)
    q_ave = E_CHARGE * delta_r / (2 * L_ch) * q_int   # [W m^-2], Eq. (37)

    # --- Scalar outputs ---
    phi_d_value = phi_array[0]                        # discharge voltage [V]
    utilisation = Gamma_array[-1] / Gamma_m           # fraction of injected gas that leaves as ions
    G_bar_L = G_bar_array[-1]                         # normalised momentum flux

    F = I_d * np.sqrt(M_XE * EPSILON_IZ / E_CHARGE) * G_bar_L   # thrust, Eq. (39)
    I_sp = F / (Q_m * 9.80665)                         # Eq. (40)
    P_d = I_d * phi_d_value                            # Eq. (41)
    eta_T = F**2 / (2.0 * Q_m * P_d)                    # Eq. (43)

    return {
        "z_bar": z_bar_array, "z": z_array,
        "Gamma": Gamma_array, "G_bar": G_array,
        "n": n_array, "n_g": n_g_array,
        "v_i": v_i_array, "E": E_array, "phi": phi_array,
        "T_bar_e": T_bar_e_array, "T_e": T_e_array,
        "phi_d": phi_d_value, "I_d": I_d, "utilisation": utilisation,
        "q_ave": q_ave, "F": F, "I_sp": I_sp, "P_d": P_d, "eta_T": eta_T,
    }


def solve_operating_point(params: ThrusterParams, Q_m_value, I_d_value) -> dict:
    """Re-solve at a given mass flow rate [kg/s] and discharge current [A]."""
    return solve_model(replace(params, Q_m=Q_m_value, I_d=I_d_value))


def solve_for_Q_m(params: ThrusterParams, Q_m_value) -> dict:
    """Re-solve the model at a given mass flow rate [kg/s], current unchanged."""
    return solve_model(replace(params, Q_m=Q_m_value))


def solve_for_voltage(params: ThrusterParams, Q_m_value, V_target_value, I_d_guess=None) -> dict:
    """Root-find I_d so the model's discharge voltage matches V_target_value."""
    def residual(I_d_value):
        return solve_operating_point(params, Q_m_value, I_d_value)["phi_d"] - V_target_value

    lo, hi = 0.2, 11.0
    if I_d_guess is not None:
        lo_try = max(0.2, I_d_guess * 0.4)
        hi_try = min(11.0, I_d_guess * 2.0)
        try:
            if residual(lo_try) * residual(hi_try) < 0:
                lo, hi = lo_try, hi_try
        except Exception:
            pass

    I_d_sol = optimize.brentq(residual, lo, hi, xtol=1e-3, rtol=1e-6, maxiter=60)
    return solve_operating_point(params, Q_m_value, I_d_sol)
