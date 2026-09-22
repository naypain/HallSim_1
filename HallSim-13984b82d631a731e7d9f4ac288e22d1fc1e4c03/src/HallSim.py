# HallSim

import streamlit as st

import time
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

st.title("HallSim")
st.write("An Interactive Web Tool for Hall Thruster Simulation and Design.")
st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExa3VydWk5NGpoc3h1ODczdm4xb2F5cnB4Z3cyYjgxODU1bjE3azgzZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/vdZFmH39wHGcVOBgh0/giphy.gif")
# preamble
# to begin, type 
# "streamlit run "C:\Users\npain\OneDrive - UNSW\Apps\_DATA\HallSim.py""
# into a new terminal then check your browser for the app.
# A variable with star supercript (ie. v_star) implies that is the 
# characteristic variable (characteristic velocity)
# A variable with bar (ie. Gamma_bar) means that is the normalized variable 
# (normalized Gamma)

# ============================================================
# Buttons for thruster geometries
# ============================================================

import streamlit as st

# Hall thruster dataset
hall_thrusters = [
    "SPT-100", "SPT-20", "SPT-25", "HET-100", "KHT-40", "KHT-50", "HEPS-200",
    "BHT-200", "KM-32", "SPT-50M", "SPT-30", "KM-37", "CAM200",
    "SPT-50", "A-3", "HEPS-500", "BHT-600", "SPT-70", "custom"
]

# Select a Hall thruster
selected_thruster_name = st.selectbox(
    "Select Hall thruster geometry:",
    hall_thrusters
)

# Thruster geometry database ===========================================================
# r_1: inner radius [m], r_2: outer radius [m], L_ch: channel length [m], Q_m: mass flow rate [kg/s]
THRUSTERS = {
    "SPT-100":  dict(r_1=35.0e-3, r_2=50.0e-3, L_ch=25.0e-3, Q_m=5.0e-6),
    "SPT-20":   dict(r_1=5.0e-3,  r_2=10.0e-3, L_ch=32.0e-3, Q_m=0.47e-6),
    "SPT-25":   dict(r_1=7.5e-3,  r_2=12.5e-3, L_ch=25.0e-3, Q_m=0.59e-6),  # L_ch placeholder
    "HET-100":  dict(r_1=9.0e-3,  r_2=14.5e-3, L_ch=14.5e-3, Q_m=0.50e-6),
    "KHT-40":   dict(r_1=11.0e-3, r_2=20.0e-3, L_ch=25.5e-3, Q_m=0.69e-6),
    "KHT-50":   dict(r_1=17.0e-3, r_2=25.0e-3, L_ch=25.0e-3, Q_m=0.88e-6),
    "HEPS-200": dict(r_1=17.0e-3, r_2=25.5e-3, L_ch=25.0e-3, Q_m=0.88e-6),
    "BHT-200":  dict(r_1=7.7e-3,  r_2=13.3e-3, L_ch=25.0e-3, Q_m=0.94e-6),  # L_ch placeholder
    "KM-32":    dict(r_1=12.5e-3, r_2=19.5e-3, L_ch=16.0e-3, Q_m=1.00e-6),
    "SPT-50M":  dict(r_1=14.0e-3, r_2=25.0e-3, L_ch=25.0e-3, Q_m=1.50e-6),
    "SPT-30":   dict(r_1=9.0e-3,  r_2=15.0e-3, L_ch=11.0e-3, Q_m=0.98e-6),
    "KM-37":    dict(r_1=14.0e-3, r_2=23.0e-3, L_ch=17.5e-3, Q_m=1.15e-6),
    "CAM200":   dict(r_1=15.5e-3, r_2=27.5e-3, L_ch=25.0e-3, Q_m=1.09e-6),  # L_ch placeholder
    "SPT-50":   dict(r_1=14.0e-3, r_2=25.0e-3, L_ch=25.0e-3, Q_m=1.18e-6),
    "A-3":      dict(r_1=17.0e-3, r_2=30.0e-3, L_ch=30.0e-3, Q_m=1.18e-6),
    "HEPS-500": dict(r_1=17.0e-3, r_2=32.5e-3, L_ch=25.0e-3, Q_m=1.67e-6),
    "BHT-600":  dict(r_1=20.0e-3, r_2=36.0e-3, L_ch=25.0e-3, Q_m=2.60e-6),  # L_ch placeholder
    "SPT-70":   dict(r_1=21.0e-3, r_2=35.0e-3, L_ch=25.0e-3, Q_m=2.56e-6),
}

# selectbox: e.g. selected_thruster_name = st.selectbox("Thruster", [*THRUSTERS, "custom"])

# Display selected geometry ============================================================
st.write("### Selected Thruster")
st.write(f"**{selected_thruster_name}**")

if selected_thruster_name in THRUSTERS:
    g = THRUSTERS[selected_thruster_name]
    r_1, r_2, L_ch, Q_m = g["r_1"], g["r_2"], g["L_ch"], g["Q_m"]

elif selected_thruster_name == "custom":
    # sliders work in mm and mg/s (readable), then convert to SI
    r_1  = st.slider("Inner radius [mm]",     0.0, 100.0, value=35.0) * 1e-3
    r_2  = st.slider("Outer radius [mm]",     0.0, 100.0, value=50.0) * 1e-3
    L_ch = st.slider("Channel length [mm]",   0.0, 100.0, value=25.0) * 1e-3
    Q_m  = st.slider("Mass flow rate [mg/s]", 0.0, 10.0,  value=5.14, step=0.01) * 1e-6
    if r_1 >= r_2: 
        st.error("Inner radius must be smaller than outer radius") 
        st.stop()

else:
    st.error(f"Unknown thruster: {selected_thruster_name}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Inner radius r₁",   f"{r_1*1e3:.1f} mm")
c2.metric("Outer radius r₂",   f"{r_2*1e3:.1f} mm")
c3.metric("Channel length L",  f"{L_ch*1e3:.1f} mm")
c4.metric("Mass flow rate ṁ",  f"{Q_m*1e6:.2f} mg/s")

# =============================================================
# Run configuration
# =============================================================
# VERIFICATION_MODE = True   # True -> Fig. 1 (Section 3.1); False -> full model
VERIFICATION_MODE = st.toggle("Verification Mode")

# if VERIFICATION_MODE:
#     st.write("Verification Mode is on!")

# Load rate coefficient data =============================================================

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

ASSET_DIR = BASE_DIR / "assets"

OUTPUT_DIR = BASE_DIR / "output"

# Load rate-coefficient table for Xe, Kr, Ar
rate_coefficients_data = pd.read_excel(ASSET_DIR / 'rate_coefficients.xlsx',
                                       header=0, skiprows=[0])
rate_coefficients_data.columns = ['idx', 'T_e_data',
                                  'K_iz_Xe', 'K_m_Xe', 'epsilon_c_Xe',
                                  'K_iz_Kr', 'K_m_Kr', 'epsilon_c_Kr',
                                  'K_iz_Ar', 'K_m_Ar', 'epsilon_c_Ar']

# Cubic interpolants for the xenon rate coefficients and collisional energy cost

K_iz_func = interpolate.interp1d(rate_coefficients_data['T_e_data'],
                                 rate_coefficients_data['K_iz_Xe'],
                                 kind='cubic', bounds_error=False,
                                 fill_value='extrapolate')

K_m_func = interpolate.interp1d(rate_coefficients_data['T_e_data'],
                                rate_coefficients_data['K_m_Xe'],
                                kind='cubic', bounds_error=False,
                                fill_value='extrapolate')

epsilon_c_func = interpolate.interp1d(rate_coefficients_data['T_e_data'],
                                      rate_coefficients_data['epsilon_c_Xe'],
                                      kind='cubic', bounds_error=False,
                                      fill_value='extrapolate')


# Constants ==============================================================================

amu = 1.66053906660e-27     # [kg] atomic mass unit
k_B = 1.380649e-23          # [J/K] Boltzmann constant
e   = 1.602176634e-19       # [C] Elementary charge
m_e = 9.10938356e-31        # [kg] Electron rest mass

# =============================================================
# Table 1 geometry of SPT-100 / operating parameters

# make a case loop dependent on 
# =============================================================
# r_1   = 35E-3          # [m] inner radius
# r_2   = 50E-3          # [m] outer radius
# L_ch  = 25E-3          #  [m] channel length
B_max = 20E-3          # [T] peak magnetic field
# Q_m   = 5E-6          # [kg s^-1] mass flow rate or Q_m
I_d   = 3              # [A] discharge current
C_mag = 4              # [-] magnetic field shape parameter
T_g   = 700            # [K] neutral gas temperature
M     = 131.293 * amu  # [kg]  (xenon)

# Normalised discharge current (paper eq. 27): I_bar_d = M*I_d/(e*Q_m)
I_bar_d = st.slider("Normalised discharge current Ī_d", 0.9, 1.6, 1.2, 0.05)
I_d = 3.0 if VERIFICATION_MODE else I_bar_d * e * Q_m / M   # [A]

# Channel-wall material (boron nitride), Eq. (12)
a = 0.207              # [eV^-b]
b = 0.549              # [-]

# Table 7 characteristic rate coefficients for Xenon (evaluated at T_e = epsilon_iz)

epsilon_iz = 12.13                          # [eV]  (xenon) ionization threshold energy
K_iz_star  = float(K_iz_func(epsilon_iz))   # ~5.45e-14 m^3/s
K_m_star   = float(K_m_func(epsilon_iz))    # ~2.37e-13 m^3/s

# Full model (Sections 3.2 / 4).

# =============================================================
# Mode-dependent physics switches and anode boundary conditions
# =============================================================
if VERIFICATION_MODE:
    # Section 3.1: anomalous transport and wall losses off (from Table 1)
    h_R          = 0
    delta_an     = 0
    f_m_pinned   = 1       # Table 1: f_m = 1
    f_eps_pinned = 2       # Table 1: f_eps = 2
    # Section 3.1 boundary conditions (chosen to match ref. [57]).
    Gamma_bar_0  = 0.01
    G_bar_0      = 0.001
else:
    # Full model (Sections 3.2 / 4).
    h_R          = 0.4     # Section 3.2 (last line rather than 0.4) or 0.3
    delta_an     = 3.5e-3  # representative anomalous transport coefficient (Section 4.1)
    f_m_pinned   = None    # not a set constant
    f_eps_pinned = None    # not a set constant
    # Section 2.5 boundary conditions: physical anode momentum flux.
    # G_bar_0 needs epsilon_iz (defined just below), so it is finalised there.
    Gamma_bar_0  = 0.005
    G_bar_0      = Gamma_bar_0 * np.sqrt(k_B * T_g / (e * epsilon_iz))

N_zbar = 200    # number of output points along the channel

# Derived parameters =====================================================================
# Note: omega_ce, nu_an, u_B, nu_m are NOT constants - they depend on z (via B)

A_ch      = np.pi * (r_2**2 - r_1**2)   # channel cross-section area [m^2]
delta_r   = r_2 - r_1                    # channel width [m]
v_g       = np.sqrt(k_B * T_g / M)       # neutral thermal speed [m/s]
v_star    = np.sqrt(e * epsilon_iz / M)  # characteristic ion speed [m/s], v* (2025)
omega_max = e * B_max / m_e              # peak electron cyclotron freq [rad/s]
Gamma_m   = Q_m / (M * A_ch)             # max ion flux (full ionisation), Eq. (15)
Gamma_d   = I_d / (e * A_ch)             # discharge flux, text after Eq. (17)
M_bar     = M / m_e                      # Eq. (33)

# Magnetic field profile =====================================================================
def f_ce_func(z):
    z_bar = z/L_ch
    f_ce = np.exp(-C_mag * (z_bar - 1)**2) # equation 45
    return f_ce

# Normalised rate coefficient fractions - Eqs. (30), (31), (34)
def rate_fractions(T_e):
    # T_e = T_bar_e * epsilon_iz                       # Eq. (28) [eV]
    f_iz = float(K_iz_func(T_e)) / K_iz_star         # Eq. (30)

    if VERIFICATION_MODE:
        f_m   = f_m_pinned                               # = 1
        f_eps = f_eps_pinned                             # = 2
    else:
        f_m   = float(K_m_func(T_e)) / K_m_star          # Eq. (31)
        f_eps = float(epsilon_c_func(T_e)) / epsilon_iz  # Eq. (34)
    
    return f_iz, f_m, f_eps

# Secondary electron emission coefficient - Eqs. (12)-(14)
def sigma_func(T_e):
    sigma_see = a * T_e**b                     # Eq. (12)
    sigma_scl = 1 - 8.3 * np.sqrt(m_e / M)     # Eq. (13)
    sigma = min(sigma_see, sigma_scl)          # Eq. (14)
    return sigma

def power_balance(T_e, Gamma, G, f_ce):
    
    T_e = max(float(np.atleast_1d(T_e)[0]), 0) # idk what this does yet
    
    
    
    f_iz, f_m, f_eps = rate_fractions(T_e)
    sigma = sigma_func(T_e)
    
    omega_ce = omega_max * f_ce             # Eq. (29)
    u_B = np.sqrt(e * T_e/M)                # Bohm velocity
    K_iz = K_iz_func(T_e)
    # K_m = K_m_func(T_e)
    # epsilon_c = epsilon_c_func(T_e)
    
    K_m = K_m_star
    epsilon_c = 2* epsilon_iz
    
    
    # Eq. (20) 
    LHS = (m_e * v_g * omega_ce**2 * G**2 * (Gamma_d - Gamma)**2) / (
    e * Gamma**4 * (K_m * (Gamma_m - Gamma) + delta_an * v_g * omega_ce))
    
    
    RHS = (((Gamma_m - Gamma) * K_iz * epsilon_c) / v_g
    + (2 * h_R * u_B * T_e / delta_r) * (2 / (1 - sigma) + np.log((1 - sigma) * np.sqrt(M / (2 * np.pi * m_e)))))
	
    return LHS - RHS

# Warm start cache for fsolve (mutable so it persists across calls).
_Te_guess = [epsilon_iz]    # ~12 eV, physical units (unlike script 1's normalised cache)

def diff_eqns(z, y):

	# Unpack state vector ----------------------------------------------------------------
    Gamma = y[0]
    G = y[1]
    f_ce = f_ce_func(z)

    guess = _Te_guess[0]
    solution = optimize.fsolve(power_balance, guess, args=(Gamma, G, f_ce))
    T_e = solution[0]
    _Te_guess[0] = T_e
    
    omega_ce = omega_max * f_ce
    u_B = np.sqrt(e * T_e/M)                # Bohm velocity
    K_iz = K_iz_func(T_e)
    # K_m = K_m_func(T_e) 
    
    K_m = K_m_star

	# Construct differential equations ---------------------------------------------------
      
    dy_dx = np.zeros(2)
    
	# Xe flux ----------------------------------------------------------------------------
	
    
    # Eq. (18) for flux   
    dy_dx[0] = ((Gamma**2 * (Gamma_m - Gamma) * K_iz) / (v_g * G)
    - (2 * h_R * Gamma**2 * u_B) / (G * delta_r))
        
    # Xe momentum flux - Eq. (19) -----------------------------------------------------
    dy_dx[1] = ((m_e * v_g * omega_ce**2 * (Gamma_d - Gamma)) /
    (M * (K_m * (Gamma_m - Gamma) + delta_an * v_g * omega_ce))
    - (2 * h_R * Gamma * u_B) / delta_r)
    
    return dy_dx


# # just after eq. (20)
# Gamma_bar_0 = 0.01
# G_bar_0 = 0.001

Gamma_0 = Gamma_bar_0 * Gamma_d              # good enough, leave for now           
G_0 = G_bar_0 * v_star * Gamma_d                   # momentum flux

y_0 = [Gamma_0, G_0]

# Full model solver
# =============================================================
def solve_model():


# --- Integrate anode (z_bar = 0) to exit (z_bar = 1) ---
    _Te_guess[0] = epsilon_iz   # reset warm start to original value
    
    z_bar_array = np.linspace(0, 1, N_zbar)
    z_array = z_bar_array * L_ch


    y_0 = [Gamma_0, G_0]
    
    sol = integrate.solve_ivp(
        diff_eqns, t_span=[0, L_ch], 
        y0=y_0, method='DOP853', t_eval=z_array, rtol=1e-10, atol=1e-6)
    
    Gamma_array = sol.y[0]
    G_array     = sol.y[1]
    
    G_bar_array = G_array/ (v_star * Gamma_d)
    
    f_ce_array = f_ce_func(z_array)
    
    _Te_guess[0] = epsilon_iz   # reset warm start before the diagnostic re-solve pass

    T_e_array = []
    for i in range(len(z_array)):
        sol_i = optimize.fsolve(power_balance, _Te_guess[0],
                                 args=(Gamma_array[i], G_array[i], f_ce_array[i]))[0]
        _Te_guess[0] = sol_i
        T_e_array.append(sol_i)
    T_e_array = np.array(T_e_array)
    
    
    T_bar_e_array = T_e_array / epsilon_iz         # Eq. (28) [eV]
    
    f_m_array = np.array([
        rate_fractions(T_bar_e)[1]
        for T_bar_e
        in T_e_array])
    
    v_i_array   = G_array / Gamma_array            # ion velocity [m/s]
    n_array     = Gamma_array / v_i_array          # plasma density [m^-3]
    n_g_array   = (Gamma_m - Gamma_array) / v_g    # neutral density, Eq.(15)

    # --- Axial electric field, Eqs. (16)-(17) ---
    omega_ce_array = omega_max * f_ce_array                                            # Eq. (29) = e B_max f_ce / m_e
    nu_an_array    = delta_an * omega_ce_array                                         # Below Eq. (6) anomalous frequency
    K_m_array      = K_m_star * f_m_array                                              # Eq. (31) actual K_m(T_e)
    mu_array       = e * (n_g_array * K_m_array + nu_an_array) / (m_e * omega_ce_array**2)  # Eq. (16)
    E_array        = (Gamma_d - Gamma_array) / (n_array * mu_array)                    # Eq. (17)

    # --- Electrostatic potential (E = -dphi/dz, cathode reference phi(L)=0) ---
    E_func = interpolate.interp1d(z_array, E_array, kind='cubic',
                                  bounds_error=False, fill_value='extrapolate')

    def phi_drop(z_value):   # Integration of the above E_func, which is an interpolation of E_array
        val, _ = integrate.quad(E_func, 0, z_value)
        return val

    phi_drop_array = np.array([
        phi_drop(z_value)
        for z_value
        in z_array])
    phi_array = phi_drop_array[-1] - phi_drop_array   # Array of voltage until exit

    # --- Wall sheath potential drop ---
    sigma_array = np.array([
        sigma_func(T_bar_e)
        for T_bar_e
        in T_bar_e_array])
    # dphi_sheath = T_e_array * np.log((1 - sigma_array) * np.sqrt(M / (2 * np.pi * m_e)))   # Besides Fig 10

    # --- Wall heat flux, Eq. (37) in physical units ---
    u_B_array   = np.sqrt(e * T_e_array / M)        # Below Eq. (11) Bohm velocity
    nu_iw_array = 2 * h_R * u_B_array / delta_r     # Eq. (10)
    nu_ew_array = nu_iw_array / (1 - sigma_array)   # Eq. (9)
    eps_w_array = T_e_array * (2 + (1 - sigma_array)
                  * np.log((1 - sigma_array) * np.sqrt(M / (2 * np.pi * m_e))))   # Eq. (11)
    q_integrand = n_array * nu_ew_array * eps_w_array
    q_func = interpolate.interp1d(z_array, q_integrand, kind='cubic',
                                  bounds_error=False, fill_value='extrapolate')
    q_int, _ = integrate.quad(q_func, 0, L_ch)
    q_ave = e * delta_r / (2 * L_ch) * q_int        # [W m^-2], Eq. (37)

    # --- Scalar outputs ---
    phi_d_value = phi_array[0]                       # discharge voltage [V]
    utilisation = Gamma_array[-1] / Gamma_m          # Fraction of injected gas that leaves as ions
    G_bar_L     = G_bar_array[-1]                    # Normalised momentum flux
    
    
    
    F           = I_d * np.sqrt(M * epsilon_iz / e) * G_bar_L   # thrust, Eq.(39)
    I_sp        = F / (Q_m * 9.80665)                # Eq. (40)
    P_d         = I_d * phi_d_value                  # Eq. (41)
    eta_T       = F**2 / (2.0 * Q_m * P_d)           # Eq. (43)

    return {
        "z_bar": z_bar_array, "z": z_array,
        "Gamma": Gamma_array, "G_bar": G_array,
        "Gamma": Gamma_array, "n": n_array, "n_g": n_g_array,
        "v_i": v_i_array, "E": E_array, "phi": phi_array,
        "T_bar_e": T_bar_e_array, "T_e": T_e_array,
        # "dphi_sheath": dphi_sheath,
        "phi_d": phi_d_value, "I_d": I_d, "utilisation": utilisation,
        "q_ave": q_ave, "F": F, "I_sp": I_sp, "P_d": P_d, "eta_T": eta_T,
    }


# =============================================================
# Run
# =============================================================
baseline = solve_model()

mode_str = "VERIFICATION (Section 3.1)" if VERIFICATION_MODE else "FULL MODEL"
st.write(f"Mode: {mode_str}")

# =============================================================
# Operating map: thrust over (Q_m, I_d)
# =============================================================
def solve_operating_point(Q_m_value, I_d_value):
    """Re-solve at a given mass flow rate [kg/s] and discharge current [A]."""
    global Q_m, Gamma_m, I_d, Gamma_d, Gamma_0, G_0
    Q_m     = Q_m_value
    I_d     = I_d_value
    Gamma_m = Q_m / (M * A_ch)            # Eq. (15)
    Gamma_d = I_d / (e * A_ch)            # after Eq. (17)
    Gamma_0 = Gamma_bar_0 * Gamma_d       # anode BCs scale with Gamma_d,
    G_0     = G_bar_0 * v_star * Gamma_d  # so they must track the swept I_d
    return solve_model()

# --- Plasma properties: value + peak location share a structure, so use a table ---
st.subheader("Plasma properties")
plasma_table = pd.DataFrame({
    "Quantity": [
        "Peak plasma density",
        "Anode neutral density",
        "Peak electron temperature",
        "Peak electric field",
    ],
    "Value": [
        f"{baseline['n'].max():.3e} m⁻³",
        f"{baseline['n_g'][0]:.3e} m⁻³",
        f"{baseline['T_e'].max():.2f} eV",
        f"{baseline['E'].max()/1e3:.2f} kV/m",
    ],
    "z/L": [
        f"{baseline['z_bar'][baseline['n'].argmax()]:.2f}",
        "0.00",
        f"{baseline['z_bar'][baseline['T_e'].argmax()]:.2f}",
        f"{baseline['z_bar'][baseline['E'].argmax()]:.2f}",
    ],
})




# --- Headline performance metrics ---
st.divider()
st.subheader("Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Thrust", f"{baseline['F']*1e3:.2f} mN")
c2.metric("Anode I_sp", f"{baseline['I_sp']:.1f} s")
c3.metric("Discharge power", f"{baseline['P_d']:.1f} W")
c4.metric("Anode efficiency", f"{baseline['eta_T']:.3f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Discharge voltage", f"{baseline['phi'][0]:.1f} V")
c6.metric("Propellant utilisation", f"{baseline['utilisation']:.3f}")
c7.metric("Wall heat flux", f"{baseline['q_ave']/1e3:.2f} kW/m^2")
c8.metric("Exit ion velocity", f"{baseline['v_i'][-1]/1e3:.2f} km/s")


st.dataframe(plasma_table, hide_index=True, use_container_width=True)


st.subheader("Thrust operating map")

if st.toggle("Run (Q_m, I_d) sweep"):
    Q_m_op, I_d_op = Q_m, I_d             # remember the operating point
    n_Q, n_I = 11, 11                     # 121 full solves — start small
    Q_m_values = np.linspace(0.5, 1.5, n_Q) * Q_m_op
    I_d_values = np.linspace(0.5, 1.5, n_I) * I_d_op

    F_grid = np.full((n_I, n_Q), np.nan)  # rows = I_d, cols = Q_m
    progress = st.progress(0.0)
    for j, i_d in enumerate(I_d_values):
        for i, q in enumerate(Q_m_values):
            try:
                F_grid[j, i] = solve_operating_point(q, i_d)["F"] * 1e3  # [mN]
            except Exception:
                pass                      # failed solve stays NaN (blank cell)
            progress.progress((j * n_Q + i + 1) / (n_I * n_Q))
    progress.empty()

    # restore the selected operating point for everything below
    solve_operating_point(Q_m_op, I_d_op)

    fig3, ax3 = plt.subplots(figsize=(7, 5.5))
    mesh = ax3.pcolormesh(Q_m_values * 1e6, I_d_values, F_grid,
                          shading='gouraud', cmap='viridis')
    fig3.colorbar(mesh, ax=ax3, label=r'Thrust $F$ (mN)')

    # iso-thrust contour lines make the gradient readable
    cs = ax3.contour(Q_m_values * 1e6, I_d_values, F_grid,
                     colors='white', linewidths=0.8, alpha=0.7)
    ax3.clabel(cs, fmt='%.0f', fontsize=8)

    # mark the selected thruster's operating point
    ax3.plot(Q_m_op * 1e6, I_d_op, 'wo', mec='black')
    ax3.annotate(selected_thruster_name, (Q_m_op * 1e6, I_d_op),
                 textcoords='offset points', xytext=(8, 6),
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))

    ax3.set_xlabel(r'$Q_m$ (mg/s)')
    ax3.set_ylabel(r'$I_d$ (A)')
    ax3.set_title(f'Thrust map — {selected_thruster_name}')
    st.pyplot(fig3)
    plt.close(fig3)



# st.write("\n========== Similarity parameters ==========")
# st.write(f"alpha   = {alpha:.4f}")
# st.write(f"beta    = {beta:.4f}")
# st.write(f"gamma   = {gamma:.4f}")
# st.write(f"lambda  = {lam:.4f}")
# st.write(f"I_bar_d = {I_bar_d:.4f}")
# st.write(f"(cross-check) beta/sqrt(f_eps) = {beta/np.sqrt(2.0):.4f}  [= 2024 alpha]")

# st.write("\n========== Normalised exit values ==========")
# st.write(f"G_bar(1)     = {baseline['G_bar'][-1]:.4f}")

# st.write("\n========== Physical quantities ==========")
# st.write(f"Peak plasma density    = {baseline['n'].max():.3e} m^-3"
#       f" at z/L = {baseline['z_bar'][baseline['n'].argmax()]:.2f}")
# st.write(f"Anode neutral density  = {baseline['n_g'][0]:.3e} m^-3")
# st.write(f"Exit ion velocity      = {baseline['v_i'][-1]/1e3:.2f} km/s")
# st.write(f"Propellant utilisation = {baseline['utilisation']:.3f}")
# st.write(f"Peak electron temp     = {baseline['T_e'].max():.2f} eV"
#       f" at z/L = {baseline['z_bar'][baseline['T_e'].argmax()]:.2f}")

# st.write("\n========== Field and potential ==========")
# st.write(f"Peak E field     = {baseline['E'].max()/1e3:.2f} kV/m"
#       f" at z/L = {baseline['z_bar'][baseline['E'].argmax()]:.2f}")
# st.write(f"Anode potential  = {baseline['phi'][0]:.1f} V  (discharge voltage)")
# st.write(f"Exit potential   = {baseline['phi'][-1]:.3e} V  (should be ~0)")

# st.write("\n========== Performance / thermal ==========")
# st.write(f"Thrust           = {baseline['F']*1e3:.2f} mN")
# st.write(f"Anode I_sp       = {baseline['I_sp']:.1f} s")
# st.write(f"Discharge power  = {baseline['P_d']:.1f} W")
# st.write(f"Anode efficiency = {baseline['eta_T']:.3f}")
# st.write(f"Wall heat flux   = {baseline['q_ave']/1e3:.2f} kW/m^2")


# =============================================================
# Parameter sweep: thrust vs mass flow rate
# =============================================================
st.divider()
st.subheader("Thrust vs mass flow rate")
def solve_for_Q_m(Q_m_value):
    """Re-solve the model at a given mass flow rate [kg/s]."""
    global Q_m, Gamma_m
    Q_m = Q_m_value
    Gamma_m = Q_m / (M * A_ch)   # Eq. (15) — Q_m only enters the ODEs through this
    return solve_model()



if st.toggle("Run Q_m sweep"):   # opt-in: each point is a full ODE solve
    Q_m_op = Q_m                                      # thruster's operating point
    Q_m_values = np.linspace(0.5, 1.5, 21) * Q_m_op   # sweep ±50% around it

    F_values = []
    progress = st.progress(0.0)
    for i, q in enumerate(Q_m_values):
        try:
            F_values.append(solve_for_Q_m(q)["F"] * 1e3)   # [mN]
        except Exception:
            F_values.append(np.nan)   # solver failed here; leaves a gap in the line
        progress.progress((i + 1) / len(Q_m_values))
    progress.empty()

    # restore the globals so anything below the sweep sees the selected thruster
    Q_m = Q_m_op
    Gamma_m = Q_m / (M * A_ch)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(Q_m_values * 1e6, F_values, 'b-')
    ax2.plot(Q_m_op * 1e6, baseline["F"] * 1e3, 'bo')
    ax2.annotate(selected_thruster_name, (Q_m_op * 1e6, baseline["F"] * 1e3),
                 textcoords="offset points", xytext=(8, -4))
    ax2.set_xlabel(r'$Q_m$ (mg/s)')
    ax2.set_ylabel(r'$F$ (mN)')
    ax2.set_title(f'Thrust vs mass flow rate — {selected_thruster_name}')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

# --- Solver sanity checks, tucked out of the way ---
with st.expander("Solver checks"):
    st.write(f"G_bar(1) = {baseline['G_bar'][-1]:.4f}")
    st.write(f"Exit potential = {baseline['phi'][-1]:.3e} V (should be ~0)")


# =============================================================
# Plot (matches Figure 1 layout)
# =============================================================

# Load verification data
figure_1_axial_profiles = pd.read_excel(ASSET_DIR / 'figure_1_axial_profiles.xlsx',
                                        header=0, skiprows=[0])
figure_1_axial_profiles.columns = ['nz1', 'n1', 'n_gz1', 'n_g1', 'z_v_i', 'v_i', 'z_E', 'E', 'z_phi', 'phi']


fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# top left graph
ax = axs[0, 0]
ax.plot(baseline["z_bar"], baseline["n_g"] * 1e-19, 'r-', label=r'model $n_g$')
ax.plot(baseline["z_bar"], baseline["n"]   * 1e-18, 'b-', label=r'model $n$')
ax.plot(figure_1_axial_profiles["n_gz1"], figure_1_axial_profiles["n_g1"], 'b:', label=r'paper $n_g$')
ax.plot(figure_1_axial_profiles["nz1"], figure_1_axial_profiles["n1"], 'r:', label=r'paper $n$')
ax.set_xlabel(r'$z / L_{ch}$')
ax.set_ylabel(r'$n_g$ ($10^{19}$ m$^{-3}$) or $n$ ($10^{18}$ m$^{-3}$)')
ax.set_title('(a) Densities'); ax.legend(); ax.grid(True)

# top right graph
axs[0, 1].plot(baseline["z_bar"], baseline["v_i"] * 1e-3, 'b-', label=r'model $v_i$')
axs[0, 1].plot(figure_1_axial_profiles["z_v_i"], figure_1_axial_profiles["v_i"], 'r:', label=r'paper $v_i$')
axs[0, 1].set(xlabel=r'$z / L_{ch}$', ylabel=r'$v_i$ ($10^3$ m/s)',
              title='(b) Ion velocity')
axs[0, 1].legend()
axs[0, 1].grid(True)

# bottom left graph
axs[1, 0].plot(baseline["z_bar"], baseline["E"] * 1e-3, 'b-', label=r'model $E$')
axs[1, 0].plot(figure_1_axial_profiles["z_E"], figure_1_axial_profiles["E"], 'r:', label=r'paper $E$')
axs[1, 0].set(xlabel=r'$z / L_{ch}$', ylabel=r'$E$ ($10^3$ V/m)',
              title='(c) Axial electric field')
axs[1, 0].legend()
axs[1, 0].grid(True)

# bottom right graph
axs[1, 1].plot(baseline["z_bar"], baseline["phi"], 'b-', label=r'model $\phi$')
axs[1, 1].plot(figure_1_axial_profiles["z_phi"], figure_1_axial_profiles["phi"], 'r:', label=r'paper $\phi$')
axs[1, 1].set(xlabel=r'$z / L_{ch}$', ylabel=r'$\phi$ (V)',
              title='(d) Electrostatic potential')
axs[1, 1].legend()
axs[1, 1].grid(True)

fig.suptitle(f'Xenon Axial Profiles - {mode_str}')
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# =============================================================
# Validation figure: I_d, F, I_sp vs mass flow rate at fixed V_d
# =============================================================
st.divider()
st.subheader("Model validation vs experimental data")
st.caption(
    "Discharge current, thrust and anode specific impulse vs. mass flow rate at a fixed "
    "discharge voltage, compared against the experimental SPT-100 data bundled in "
    "`assets/experimental_data.xlsx`. The model curve is solved at your currently selected "
    "geometry; select SPT-100 above for a like-for-like comparison."
)

if VERIFICATION_MODE:
    st.info("Switch off Verification Mode above to run the full-model validation sweep.")
else:
    VALIDATION_DATASETS = {
        "Xenon, 300 V": ("Xenon (300 V)", 300.0),
        "Xenon, 250 V": ("Xenon (250 V)", 250.0),
        "Xenon, 200 V": ("Xenon (200 V)", 200.0),
    }
    validation_choice = st.selectbox(
        "Experimental dataset (fixed discharge voltage)",
        list(VALIDATION_DATASETS.keys()), index=1,
    )
    exp_sheet_name, V_target = VALIDATION_DATASETS[validation_choice]

    exp_data = pd.read_excel(ASSET_DIR / 'experimental_data.xlsx',
                             sheet_name=exp_sheet_name, header=0)
    exp_Qm  = exp_data['Anode  Mass Flow Rate (mg/s)'].to_numpy()
    exp_Id  = exp_data['Discharge Current (A)'].to_numpy()
    exp_F   = exp_data['Thrust (mN)'].to_numpy()
    exp_Isp = (exp_F * 1e-3) / (exp_Qm * 1e-6 * 9.80665)

    n_pts = st.slider(
        "Sweep resolution (mass-flow points)", 5, 20, 12,
        help="Each point is a full ODE solve wrapped in a root-find for the discharge "
             "current that matches the target voltage — more points take longer."
    )

    run_validation = st.toggle(
        f"Run validation sweep at {V_target:.0f} V "
        f"(~{n_pts} full model solves, may take up to a minute)"
    )

    if run_validation:
        Q_m_op, I_d_op = Q_m, I_d   # remember the operating point, restore afterwards

        Qm_lo = max(0, float(exp_Qm.min()) * 0.85)
        Qm_hi = max(8.0, float(exp_Qm.max()) * 1.3)
        Qm_sweep_mg = np.linspace(Qm_lo, Qm_hi, n_pts)

        def solve_for_voltage(Q_m_value, V_target_value, I_d_guess):
            """Root-find I_d so the model's discharge voltage matches V_target_value."""
            def residual(I_d_value):
                return solve_operating_point(Q_m_value, I_d_value)["phi_d"] - V_target_value

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
            return solve_operating_point(Q_m_value, I_d_sol)

        model_Qm, model_Id, model_F, model_Isp = [], [], [], []
        I_d_guess = None
        progress = st.progress(0.0)
        for i, qm_mg in enumerate(Qm_sweep_mg):
            try:
                r = solve_for_voltage(qm_mg * 1e-6, V_target, I_d_guess)
                I_d_guess = r["I_d"]
                model_Qm.append(qm_mg)
                model_Id.append(r["I_d"])
                model_F.append(r["F"] * 1e3)
                model_Isp.append(r["I_sp"])
            except Exception:
                pass   # failed root-find at this point: skip it, leave a gap
            progress.progress((i + 1) / len(Qm_sweep_mg))
        progress.empty()

        # restore the globals so anything below sees the selected thruster's operating point
        solve_operating_point(Q_m_op, I_d_op)

        if len(model_Qm) < n_pts:
            st.warning(f"{n_pts - len(model_Qm)} of {n_pts} sweep points did not converge "
                       "and were skipped.")

        fig4, axs4 = plt.subplots(1, 3, figsize=(13, 4.2))

        panels = [
            (axs4[0], model_Id, exp_Id, r'Discharge Current, $I_d$ (A)'),
            (axs4[1], model_F,  exp_F,  r'Thrust, $F$ (mN)'),
            (axs4[2], model_Isp, exp_Isp, r'Anode Specific Impulse, $I_{sp}$ (s)'),
        ]
        for ax, model_y, exp_y, ylabel in panels:
            ax.plot(model_Qm, model_y, 'k-', label='Model')
            ax.scatter(exp_Qm, exp_y, facecolors='tab:blue', edgecolors='navy',
                      alpha=0.65, s=55, label='Experiment')
            # Automatically crop the axes tightly to the data
            # ax.axis('tight')
            ax.set_xlabel(r'Mass Flow Rate, $Q_m$ (mg/s)')
            ax.set_ylabel(ylabel)
            # ax.set_xlim(0, Qm_hi)
            # ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig4.suptitle(f'SPT-100 Performance vs Mass Flow Rate — {validation_choice}')
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)


# =============================================================
# Save for cross-comparison (same keys as the bare-bones script)
# =============================================================
np.savez(OUTPUT_DIR / 'numerical_results.npz',
         xi=baseline['z_bar'], n_g=baseline['n_g'], n=baseline['n'],
         v_i=baseline['v_i'], E=baseline['E'], phi=baseline['phi'],
         I_d=baseline['I_d'], V_d=baseline['phi_d'])
st.write("Saved to 'numerical_results.npz'")

import io, zipfile, hashlib

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

def fmt_size(n):
    """1234567 -> '1.2 MB' (decimal units, like Zenodo)."""
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"

@st.cache_data                        # read + hash once, not on every rerun
def load_files(folder: str):
    files = []
    for p in sorted(pathlib.Path(folder).glob("*.xlsx")):
        data = p.read_bytes()
        files.append(dict(name=p.name, data=data, size=len(data),
                          md5=hashlib.md5(data).hexdigest()))
    return files

@st.cache_data
def zip_files(folder: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in load_files(folder):
            zf.writestr(f["name"], f["data"])
    return buf.getvalue()


files = load_files(str(BASE_DIR))
total = sum(f["size"] for f in files)

st.subheader("Files")
with st.expander(f"Files ({fmt_size(total)})", expanded=True):
    # header row
    h1, h2, h3 = st.columns([4, 1.5, 1.3], vertical_alignment="center")
    h1.markdown("**Name**")
    h2.markdown("**Size**")
    h3.download_button("Download all", zip_files(str(BASE_DIR)),
                       file_name="files.zip", mime="application/zip",
                       icon=":material/download:", key="dl_all")

    # one row per file
    for f in files:
        st.divider()
        c1, c2, c3 = st.columns([4, 1.5, 1.3], vertical_alignment="center")
        c1.markdown(f"[{f['name']}](https://zenodo.org/records/<ID>/files/{f['name']})")
        c1.caption(f"md5:{f['md5']}")
        c2.write(fmt_size(f["size"]))
        c3.download_button("Download", f["data"], file_name=f["name"],
                           mime=XLSX_MIME, icon=":material/download:",
                           key=f"dl_{f['name']}")   # keys must be unique per button


from urllib.parse import quote_plus

REFERENCES = [
    dict(authors="Lafleur T and Chabert P", year=2025,
         title="Similarity parameters and scaling laws for Hall thrusters",
         journal="Plasma Sources Sci. Technol.", volume=34, pages="055005",
         doi="10.1088/1361-6595/add562",
         pdf="https://iopscience.iop.org/article/10.1088/1361-6595/add562/pdf",
         anchor="thrust-operating-map"),    # a subheader in your app (see note below)
    dict(authors="Goebel D M, Katz I and Mikellides I G", year=2023,
         title="Fundamentals of Electric Propulsion", publisher="Wiley"),
]

def format_citation(r):
    """IOP style: authors year title *journal* **vol** pages, or authors year *book* (publisher)."""
    if "journal" in r:
        return (f"{r['authors']} {r['year']} {r['title']} *{r['journal']}* "
                f"**{r['volume']}** {r['pages']}")
    return f"{r['authors']} {r['year']} *{r['title']}* ({r['publisher']})"

def reference_links(r):
    """The row of text links under each citation."""
    links = []
    if r.get("anchor"):
        links.append(f"[Go to reference in article](#{r['anchor']})")
    if r.get("doi"):
        links.append(f"[Crossref](https://doi.org/{r['doi']})")
    links.append("[Google Scholar](https://scholar.google.com/scholar?q="
                 f"{quote_plus(r['title'])})")
    return ("&nbsp;" * 4).join(links)


with st.expander("References"):
    for i, r in enumerate(REFERENCES, start=1):
        st.markdown(f"**[{i}]** {format_citation(r)}")
        links_col, pdf_col = st.columns([4, 1], vertical_alignment="center")
        links_col.markdown(reference_links(r))
        if r.get("pdf"):
            pdf_col.link_button("View PDF", r["pdf"],
                                icon=":material/visibility:", type="primary")
        st.write("")                     # spacing between entries
