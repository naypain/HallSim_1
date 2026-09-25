# HallSim
#
# An interactive Streamlit web tool for Hall thruster simulation and design.
# All the maths lives in model.py; thruster geometries live in thrusters.json.
# This file is only the web-app layer: widgets, plots, and file downloads.
#
# To run:
#   streamlit run src/HallSim.py
#
# Naming convention: a star superscript (e.g. v_star) marks a characteristic
# variable; a bar (e.g. Gamma_bar) marks a normalised variable.

import io
import json
import zipfile
import hashlib
import mimetypes
import pathlib
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from model import ThrusterParams, discharge_current_from_Ibar, Ibar_from_discharge_current
from model import solve_model, solve_operating_point, solve_for_Q_m, solve_for_voltage

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
ASSET_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "output"

st.title("HallSim")
st.write("An Interactive Web Tool for Hall Thruster Simulation and Design.")
st.image(str(ASSET_DIR / "SPT100_ai_scan_better.gif"))

# ============================================================
# Thruster geometry selection
# ============================================================
# thrusters.json: r_1: inner radius [m], r_2: outer radius [m],
#                 L_ch: channel length [m], Q_m: mass flow rate [kg/s]
with open(pathlib.Path(__file__).resolve().parent / "thrusters.json") as f:
    THRUSTERS = json.load(f)

selected_thruster_name = st.selectbox(
    "Select Hall thruster geometry:",
    [*THRUSTERS, "custom"]
)

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
c3.metric("Channel length L_ch",  f"{L_ch*1e3:.1f} mm")
c4.metric("Mass flow rate Qₘ",  f"{Q_m*1e6:.2f} mg/s")

# ============================================================
# Run configuration
# ============================================================
VERIFICATION_MODE = st.toggle("Verification Mode")

# Normalised discharge current (paper eq. 27): I_bar_d = M*I_d/(e*Q_m).
# [0.9, 1.6] is the range the underlying model is validated over; results
# drift from physical for real thrusters outside this band (see operating map below).
IBAR_D_VALID_RANGE = (1.1, 1.4)
I_bar_d = st.slider("Normalised discharge current Ī_d", *IBAR_D_VALID_RANGE, 1.2, 0.05)
I_d = 3.0 if VERIFICATION_MODE else discharge_current_from_Ibar(I_bar_d, Q_m)


params = ThrusterParams(r_1=r_1, r_2=r_2, L_ch=L_ch, Q_m=Q_m, I_d=I_d,
                         verification_mode=VERIFICATION_MODE)

# =============================================================
# Run
# =============================================================
baseline = solve_model(params)

mode_str = "VERIFICATION (Section 3.1)" if VERIFICATION_MODE else "FULL MODEL"
st.write(f"Mode: {mode_str}")

# =============================================================
# Operating map: thrust over (Q_m, I_d)
# =============================================================

# --- Plasma properties: value + peak location share a structure, so use a table ---
with st.expander("Plasma properties"):
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
    st.dataframe(plasma_table, hide_index=True, use_container_width=True)

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

st.subheader("Thrust operating map")
# st.caption(
#     "Thrust as a function of mass flow rate and discharge current, swept independently "
#     "±50% around the selected operating point. The dashed lines mark where the "
#     "normalised discharge current Ī_d leaves the "
#     f"[{IBAR_D_VALID_RANGE[0]}, {IBAR_D_VALID_RANGE[1]}] range the model is validated over "
#     "(see the slider above) — outside that band the model is extrapolating, and the "
#     "discharge voltage it predicts can become unrealistically high or low."
# )

if st.toggle("Run mass flow rate and discharge current sweep"):
    Q_m_op, I_d_op = params.Q_m, params.I_d   # the selected operating point
    n_Q, n_I = 11, 11                         # 121 full solves — start small
    Q_m_values = np.linspace(0.5, 1.5, n_Q) * Q_m_op
    I_d_values = np.linspace(0.5, 1.5, n_I) * I_d_op

    F_grid = np.full((n_I, n_Q), np.nan)  # rows = I_d, cols = Q_m
    progress = st.progress(0.0)
    for j, i_d in enumerate(I_d_values):
        for i, q in enumerate(Q_m_values):
            try:
                F_grid[j, i] = solve_operating_point(params, q, i_d)["F"] * 1e3  # [mN]
            except Exception:
                pass                      # failed solve stays NaN (blank cell)
            progress.progress((j * n_Q + i + 1) / (n_I * n_Q))
    progress.empty()

    # normalised discharge current at every grid point, to flag where the model is
    # being extrapolated beyond the range it's actually validated over
    Ibar_d_grid = Ibar_from_discharge_current(I_d_values[:, None], Q_m_values[None, :])

    fig3, ax3 = plt.subplots(figsize=(7, 5.5))
    mesh = ax3.pcolormesh(Q_m_values * 1e6, I_d_values, F_grid,
                          shading='gouraud', cmap='viridis')
    fig3.colorbar(mesh, ax=ax3, label=r'Thrust $F$ (mN)')

    # iso-thrust contour lines make the gradient readable
    cs = ax3.contour(Q_m_values * 1e6, I_d_values, F_grid,
                     colors='white', linewidths=0.8, alpha=0.7)
    ax3.clabel(cs, fmt='%.0f', fontsize=8)

    # boundary of the validated normalised-current range
    cs_ibar = ax3.contour(Q_m_values * 1e6, I_d_values, Ibar_d_grid,
                          levels=IBAR_D_VALID_RANGE, colors='red',
                          linewidths=1.2, linestyles='dashed')
    ax3.clabel(cs_ibar, fmt=r'$\bar{I}_d$=%.1f', fontsize=8)

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

# # --- Solver sanity checks, tucked out of the way ---
# with st.expander("Solver checks"):
#     st.write(f"G_bar(1) = {baseline['G_bar'][-1]:.4f}")
#     st.write(f"Exit potential = {baseline['phi'][-1]:.3e} V (should be ~0)")

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
st.subheader("Model validation for SPT100 and Xenon")
# st.caption(
#     "Discharge current, thrust and anode specific impulse vs. mass flow rate at a fixed "
#     "discharge voltage, compared against the experimental SPT-100 data bundled in "
#     "`assets/experimental_data.xlsx`. The model curve is solved at your currently selected "
#     "geometry; select SPT-100 above for a like-for-like comparison."
# )

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
        Qm_lo = max(0, float(exp_Qm.min()) * 0.85)
        Qm_hi = max(8.0, float(exp_Qm.max()) * 1.3)
        Qm_sweep_mg = np.linspace(Qm_lo, Qm_hi, n_pts)

        model_Qm, model_Id, model_F, model_Isp = [], [], [], []
        I_d_guess = None
        progress = st.progress(0.0)
        for i, qm_mg in enumerate(Qm_sweep_mg):
            try:
                r = solve_for_voltage(params, qm_mg * 1e-6, V_target, I_d_guess)
                I_d_guess = r["I_d"]
                model_Qm.append(qm_mg)
                model_Id.append(r["I_d"])
                model_F.append(r["F"] * 1e3)
                model_Isp.append(r["I_sp"])
            except Exception:
                pass   # failed root-find at this point: skip it, leave a gap
            progress.progress((i + 1) / len(Qm_sweep_mg))
        progress.empty()

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
            ax.set_xlabel(r'Mass Flow Rate, $Q_m$ (mg/s)')
            ax.set_ylabel(ylabel)
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
    for p in sorted(pathlib.Path(folder).iterdir()):
        if not p.is_file() or p.suffix.lower() == ".gif":
            continue
        data = p.read_bytes()
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        files.append(dict(name=p.name, data=data, size=len(data), mime=mime,
                          md5=hashlib.md5(data).hexdigest()))
    return files

@st.cache_data
def zip_files(folder: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in load_files(folder):
            zf.writestr(f["name"], f["data"])
    return buf.getvalue()

files = load_files(str(ASSET_DIR))
total = sum(f["size"] for f in files)

st.subheader("Files")
with st.expander(f"Files ({fmt_size(total)})", expanded=True):
    # header row
    h1, h2, h3 = st.columns([4, 1.5, 1.3], vertical_alignment="center")
    h1.markdown("**Name**")
    h2.markdown("**Size**")
    h3.download_button("Download all", zip_files(str(ASSET_DIR)),
                       file_name="assets.zip", mime="application/zip",
                       icon=":material/download:", key="dl_all")

    # one row per file
    for f in files:
        st.divider()
        c1, c2, c3 = st.columns([4, 1.5, 1.3], vertical_alignment="center")
        c1.markdown(f"**{f['name']}**")
        c1.caption(f"md5:{f['md5']}")
        c2.write(fmt_size(f["size"]))
        c3.download_button("Download", f["data"], file_name=f["name"],
                           mime=f["mime"], icon=":material/download:",
                           key=f"dl_{f['name']}")   # keys must be unique per button

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
