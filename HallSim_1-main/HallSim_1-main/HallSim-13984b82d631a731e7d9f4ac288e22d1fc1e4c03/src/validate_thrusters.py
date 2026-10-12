# Validate the geometries in thrusters.json against experimental data.
#
# Source: collected low-power (<1.35 kW) Hall-thruster dataset,
# doi:10.2514/1.B37424, Table 1 (assets/hall_thruster_data_doi_10.2514_1.B37424.xlsx).
#
# Two checks are run for every thruster common to both files:
#   1. Geometry check   - do r_1, r_2, L_ch, Q_m in thrusters.json match the
#                          source table (after unit conversion)?
#   2. Performance check - using that geometry, root-find the discharge
#                          current that reproduces the reported anode voltage
#                          (U_d), then compare the model's thrust, specific
#                          impulse and discharge power against the reported
#                          experimental values.
#
# Run with:
#   python src/validate_thrusters.py

import json
import pathlib
import time

import pandas as pd

from model import ThrusterParams, solve_for_voltage

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
ASSET_DIR = BASE_DIR / "assets"
EXPERIMENTAL_DATA_PATH = ASSET_DIR / "hall_thruster_data_doi_10.2514_1.B37424.xlsx"
THRUSTERS_PATH = pathlib.Path(__file__).resolve().parent / "thrusters.json"

GEOMETRY_TOLERANCE = 0.01   # 1% - flags transcription/unit-conversion mistakes


def load_experimental_table() -> pd.DataFrame:
    df = pd.read_excel(EXPERIMENTAL_DATA_PATH, sheet_name="Table 1", header=2)
    df = df.dropna(subset=["index"])   # drop the trailing footnote row
    df["name"] = df["Thruster"].str.replace(r"\s*\[[0-9,]+\]\s*$", "", regex=True).str.strip()
    return df.set_index("name")


def load_thrusters() -> dict:
    with open(THRUSTERS_PATH) as f:
        return json.load(f)


def check_geometry(name, geom, exp) -> list:
    """Compare thrusters.json's geometry/mass-flow to the source table. Returns mismatches."""
    checks = [
        ("r_1", geom["r_1"], exp["inner radius, r_1"] * 1e-3),
        ("r_2", geom["r_2"], exp["outer radius, r_2"] * 1e-3),
        ("Q_m", geom["Q_m"], exp["ṁ_a, mg/s"] * 1e-6),
    ]
    if pd.notna(exp["L, mm (channel length)"]):
        checks.append(("L_ch", geom["L_ch"], exp["L, mm (channel length)"] * 1e-3))

    mismatches = []
    for field, json_value, table_value in checks:
        if abs(json_value - table_value) > GEOMETRY_TOLERANCE * abs(table_value):
            mismatches.append(f"{field}: json={json_value:.4g} vs table={table_value:.4g}")
    return mismatches


def validate() -> pd.DataFrame:
    experimental = load_experimental_table()
    thrusters = load_thrusters()

    rows = []
    for name, geom in thrusters.items():
        if name not in experimental.index:
            continue   # e.g. "custom" is a UI-only option, not a real geometry
        exp = experimental.loc[name]

        geometry_mismatches = check_geometry(name, geom, exp)

        params = ThrusterParams(r_1=geom["r_1"], r_2=geom["r_2"], L_ch=geom["L_ch"],
                                 Q_m=geom["Q_m"], I_d=3.0, verification_mode=False)
        U_d_target = exp["U_d, V (anode voltage)"]
        I_d_guess = exp["P, W"] / U_d_target   # P = I_d * U_d -- a good starting bracket

        t0 = time.time()
        print(f"  solving {name} ...", flush=True)
        try:
            result = solve_for_voltage(params, geom["Q_m"], U_d_target, I_d_guess=I_d_guess)
        except Exception as exc:
            print(f"    failed after {time.time() - t0:.1f}s: {exc}", flush=True)
            rows.append(dict(thruster=name, geometry_issues="; ".join(geometry_mismatches),
                             solve_error=str(exc)))
            continue
        print(f"    done in {time.time() - t0:.1f}s", flush=True)

        F_model_mN, F_exp_mN = result["F"] * 1e3, exp["T, mN"]
        Isp_model_s, Isp_exp_s = result["I_sp"], exp["I_sps,a, s"]
        P_model_W, P_exp_W = result["P_d"], exp["P, W"]

        rows.append(dict(
            thruster=name,
            geometry_issues="; ".join(geometry_mismatches),
            F_model_mN=F_model_mN, F_exp_mN=F_exp_mN,
            F_err_pct=100 * (F_model_mN - F_exp_mN) / F_exp_mN,
            Isp_model_s=Isp_model_s, Isp_exp_s=Isp_exp_s,
            Isp_err_pct=100 * (Isp_model_s - Isp_exp_s) / Isp_exp_s,
            P_model_W=P_model_W, P_exp_W=P_exp_W,
            P_err_pct=100 * (P_model_W - P_exp_W) / P_exp_W,
        ))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = validate()

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    geometry_issues = results[results["geometry_issues"].astype(bool)]
    if not geometry_issues.empty:
        print("=== Geometry mismatches (thrusters.json vs source table) ===")
        print(geometry_issues[["thruster", "geometry_issues"]].to_string(index=False))
        print()

    print("=== Performance validation (model vs experiment) ===")
    display_cols = ["thruster", "F_model_mN", "F_exp_mN", "F_err_pct",
                     "Isp_model_s", "Isp_exp_s", "Isp_err_pct",
                     "P_model_W", "P_exp_W", "P_err_pct"]
    print(results[display_cols].round(2).to_string(index=False))

    print()
    print("Mean absolute error:  "
          f"F={results['F_err_pct'].abs().mean():.1f}%  "
          f"Isp={results['Isp_err_pct'].abs().mean():.1f}%  "
          f"P={results['P_err_pct'].abs().mean():.1f}%")
