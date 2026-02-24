from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def cagr(start: float, end: float, n_years: int) -> float:
    start = float(start)
    end = float(end)
    return (end / start) ** (1 / n_years) - 1


def build_revenue_forecast_from_core(core: dict) -> tuple[pd.DataFrame, pd.Series]:
    cfg = core["revenue"]
    base = cfg["base_segment_revenue_billion"]
    target = cfg["target_revenue_billion_period_10"]
    periods = np.arange(1, 11, dtype=int)
    auto_growth = np.asarray(cfg["automotive_growth_path_period_1_to_10"], dtype=float)

    auto = []
    prev_auto = float(base["automotive"])
    for i, g in enumerate(auto_growth, start=1):
        if i < 10:
            prev_auto = prev_auto * (1 + g)
            auto.append(prev_auto)
        else:
            auto.append(float(target["automotive"]))
    auto = np.asarray(auto, dtype=float)

    energy_cagr = cagr(base["energy"], target["energy"], 10)
    software_cagr = cagr(base["software"], target["software"], 10)
    energy = np.asarray([base["energy"] * (1 + energy_cagr) ** p for p in periods], dtype=float)
    software = np.asarray([base["software"] * (1 + software_cagr) ** p for p in periods], dtype=float)

    robot = np.zeros(10, dtype=float)
    launch_period = int(cfg["robotaxi"]["launch_period"])
    launch_rev = float(cfg["robotaxi"]["launch_revenue_billion"])
    robot_target = float(target["robotaxi"])
    robot_cagr = cagr(launch_rev, robot_target, 6)
    robot[launch_period - 1] = launch_rev
    for idx in range(launch_period, 10):
        robot[idx] = robot[idx - 1] * (1 + robot_cagr)
    robot[-1] = robot_target

    rev_tbl = pd.DataFrame(
        {
            "Forecast Period": periods,
            "Total Revenue": auto + energy + software + robot,
            "Automotive": auto,
            "growth rate": auto_growth,
            "Energy": energy,
            "Software": software,
            "Robotaxi": robot,
        }
    )
    base_row = pd.Series(
        {
            "Forecast Period": f"Base Year ({cfg['base_year']})",
            "Total Revenue": float(sum(base.values())),
            "Automotive": float(base["automotive"]),
            "growth rate": np.nan,
            "Energy": float(base["energy"]),
            "Software": float(base["software"]),
            "Robotaxi": float(base["robotaxi"]),
        }
    )
    return rev_tbl, base_row


def build_gross_margin_from_core(
    core: dict, rev_tbl: pd.DataFrame, software_revenue_mode: str = "workbook_link"
) -> pd.DataFrame:
    gm = core["gross_margin"]
    n = len(rev_tbl)

    # Automotive margin path replicates the workbook's recursive formulas exactly.
    f3 = float(gm["automotive_margin_base"])
    f8 = float(gm["automotive_margin_target"])
    f9 = f8
    f10 = f8
    f4 = f3
    f5 = f4 + (f8 - f3) / 4
    f6 = f5 + (f9 - f4) / 4
    f7 = f6 + (f10 - f5) / 4
    auto_margin = np.array([f4, f5, f6, f7, f8, f8, f8, f8, f8, f8], dtype=float)

    i3 = float(gm["energy_margin_base"])
    i13 = float(gm["energy_margin_target"])
    energy_margin = np.zeros(n, dtype=float)
    energy_margin[0] = i3 + (i13 - i3) / 10
    for i in range(1, 9):
        energy_margin[i] = energy_margin[i - 1] + (i13 - i3) / 9
    energy_margin[9] = i13

    l3 = float(gm["software_margin_base"])
    l13 = float(gm["software_margin_target"])
    software_margin = np.zeros(n, dtype=float)
    software_margin[0] = l3 + (l13 - l3) / 10
    for i in range(1, 9):
        software_margin[i] = software_margin[i - 1] + (l13 - l3) / 10
    software_margin[9] = l13

    o7 = float(gm["robotaxi_margin_launch"])
    o13 = float(gm["robotaxi_margin_target"])
    robot_margin = np.zeros(n, dtype=float)
    launch_period = int(core["revenue"]["robotaxi"]["launch_period"])
    robot_margin[launch_period - 1] = o7
    for i in range(launch_period, 9):
        robot_margin[i] = robot_margin[i - 1] + (o13 - o7) / 6
    robot_margin[9] = o13

    out = pd.DataFrame({"Forecast Period": rev_tbl["Forecast Period"].to_numpy(dtype=int)})
    out["Total Revenue"] = rev_tbl["Total Revenue"].to_numpy(dtype=float)
    out["Auto revenue"] = rev_tbl["Automotive"].to_numpy(dtype=float)
    out["Energy revenue"] = rev_tbl["Energy"].to_numpy(dtype=float)
    out["Software revenue"] = rev_tbl["Software"].to_numpy(dtype=float)
    out["Robotaxi revenue"] = rev_tbl["Robotaxi"].to_numpy(dtype=float)

    out["Auto GM"] = auto_margin
    out["Energy GM"] = energy_margin
    out["Software GM"] = software_margin
    out["Robotaxi GM"] = np.where(out["Robotaxi revenue"].to_numpy(dtype=float) > 0, robot_margin, np.nan)

    if software_revenue_mode == "workbook_link":
        software_rev_used = out["Energy revenue"].to_numpy(dtype=float)
        software_gp_col = "Software GP (workbook logic)"
    elif software_revenue_mode == "forecast_software":
        software_rev_used = out["Software revenue"].to_numpy(dtype=float)
        software_gp_col = "Software GP"
    else:
        raise ValueError("software_revenue_mode must be 'workbook_link' or 'forecast_software'")

    out["Software revenue used for GP"] = software_rev_used
    out["Auto GP"] = out["Auto revenue"] * out["Auto GM"]
    out["Energy GP"] = out["Energy revenue"] * out["Energy GM"]
    out[software_gp_col] = software_rev_used * out["Software GM"]
    out["Robotaxi GP"] = out["Robotaxi revenue"].to_numpy(dtype=float) * np.nan_to_num(
        out["Robotaxi GM"].to_numpy(dtype=float), nan=0.0
    )
    out["Total gross profit"] = out["Auto GP"] + out["Energy GP"] + out[software_gp_col] + out["Robotaxi GP"]
    out["Total gross margin"] = out["Total gross profit"] / out["Total Revenue"]
    return out


def build_operating_expense_from_core(
    core: dict, rev_tbl: pd.DataFrame, gross_margin_tbl: pd.DataFrame
) -> pd.DataFrame:
    cfg = core["operating_expense"]
    rd_growth = np.asarray(cfg["rd_growth_path_period_1_to_10"], dtype=float)
    n = len(rd_growth)

    rd = np.zeros(n, dtype=float)
    prev = float(cfg["rd_base_billion"])
    for i, g in enumerate(rd_growth):
        prev = prev * (1 + g)
        rd[i] = prev

    sga = np.full(n, float(cfg["sga_constant_billion"]), dtype=float)
    other = np.full(n, float(cfg["other_constant_billion"]), dtype=float)
    total_opex = rd + sga + other

    total_gp = gross_margin_tbl["Total gross profit"].to_numpy(dtype=float)
    revenue = rev_tbl["Total Revenue"].to_numpy(dtype=float)
    ebit = total_gp - total_opex
    op_margin = ebit / revenue

    return pd.DataFrame(
        {
            "Forecast Period": rev_tbl["Forecast Period"].to_numpy(dtype=int),
            "Total Revenue": revenue,
            "Total gross profit": total_gp,
            "R&D": rd,
            "R&D growth": rd_growth,
            "SG&A": sga,
            "Restructuring and other": other,
            "Total operating expense": total_opex,
            "Operating margin": op_margin,
            "Operating profit (EBIT)": ebit,
        }
    )


def build_reinvestment_from_core(core: dict, rev_tbl: pd.DataFrame) -> pd.DataFrame:
    stc = np.asarray(core["reinvestment"]["sales_to_capital_path_period_1_to_10"], dtype=float)
    base_total_rev = float(sum(core["revenue"]["base_segment_revenue_billion"].values()))
    revenue = rev_tbl["Total Revenue"].to_numpy(dtype=float)
    delta_revenue = np.r_[revenue[0] - base_total_rev, np.diff(revenue)]
    return pd.DataFrame(
        {
            "Forecast Period": rev_tbl["Forecast Period"].to_numpy(dtype=int),
            "Sales to Capital": stc,
            "Revenue growth": delta_revenue,
            "Reinvestment": delta_revenue / stc,
        }
    )


def build_fcff_from_core(
    core: dict, rev_tbl: pd.DataFrame, opex_tbl: pd.DataFrame, reinvestment_tbl: pd.DataFrame
) -> pd.DataFrame:
    tax = float(core["valuation"]["tax_rate"])
    revenue = rev_tbl["Total Revenue"].to_numpy(dtype=float)
    op_margin = opex_tbl["Operating margin"].to_numpy(dtype=float)
    ebit = revenue * op_margin
    ebit_after_tax = ebit * (1 - tax)
    reinvestment = reinvestment_tbl["Reinvestment"].to_numpy(dtype=float)
    fcff = ebit_after_tax - reinvestment
    return pd.DataFrame(
        {
            "Forecast Period": rev_tbl["Forecast Period"].to_numpy(dtype=int),
            "Total Revenues": revenue,
            "Operating Margin": op_margin,
            "EBIT": ebit,
            "EBIT(1-t)": ebit_after_tax,
            "Reinvestment": reinvestment,
            "FCFF": fcff,
        }
    )


def build_valuation_from_core(
    core: dict,
    fcff_tbl: pd.DataFrame,
    dcf_pv_func: Callable[[np.ndarray, float], float],
    terminal_value_func: Callable[[float, float, float], float],
) -> dict:
    v = core["valuation"]
    rf = float(v["risk_free_rate"])
    erp = float(v["equity_risk_premium"])
    beta = float(v["beta"])
    kd = float(v["cost_of_debt"])
    equity_value = float(v["equity_value_input_billion"])
    debt_value = float(v["debt_value_input_billion"])
    cost_of_equity = rf + beta * erp

    # Replicate the workbook formula exactly (including the workbook's structure).
    workbook_wacc_formula = (cost_of_equity * cost_of_equity) / (equity_value + debt_value) + kd * debt_value / (
        equity_value + debt_value
    )

    w_forecast = float(v["forecast_discount_rate_workbook"])
    w_terminal = float(v["terminal_discount_rate"])
    g = float(v["terminal_growth_rate"])
    fcff = fcff_tbl["FCFF"].to_numpy(dtype=float)

    pv_forecast = dcf_pv_func(fcff, w_forecast)
    terminal_value = terminal_value_func(fcff[-1], w_terminal, g)
    pv_terminal = terminal_value / (1 + w_terminal) ** len(fcff)

    return {
        "assumptions": pd.Series(
            {
                "Tax rate": float(v["tax_rate"]),
                "Risk-free rate": rf,
                "Equity risk premium": erp,
                "Beta": beta,
                "Cost of equity (CAPM)": cost_of_equity,
                "Cost of debt": kd,
                "Equity value input (bn)": equity_value,
                "Debt value input (bn)": debt_value,
                "Forecast discount rate (workbook)": w_forecast,
                "Terminal discount rate": w_terminal,
                "Terminal growth rate": g,
                "Workbook WACC formula result": workbook_wacc_formula,
            }
        ),
        "intrinsic_value_components": pd.Series(
            {
                "Terminal Value": terminal_value,
                "PV(Terminal Value)": pv_terminal,
                "PV(CF over Forecast Period)": pv_forecast,
                "Value of Operating Assets": pv_forecast + pv_terminal,
            }
        ),
    }


def build_workbook_style_model(
    core: dict,
    dcf_pv_func: Callable[[np.ndarray, float], float],
    terminal_value_func: Callable[[float, float, float], float],
) -> dict:
    rev_tbl, base_row = build_revenue_forecast_from_core(core)
    gross_margin_tbl = build_gross_margin_from_core(core, rev_tbl, software_revenue_mode="workbook_link")
    opex_tbl = build_operating_expense_from_core(core, rev_tbl, gross_margin_tbl)
    reinvestment_tbl = build_reinvestment_from_core(core, rev_tbl)
    fcff_tbl = build_fcff_from_core(core, rev_tbl, opex_tbl, reinvestment_tbl)
    valuation = build_valuation_from_core(core, fcff_tbl, dcf_pv_func, terminal_value_func)
    return {
        "base_row": base_row,
        "revenue": rev_tbl,
        "gross_margin": gross_margin_tbl,
        "operating_expense": opex_tbl,
        "reinvestment": reinvestment_tbl,
        "fcff": fcff_tbl,
        "valuation": valuation,
    }


def compare_model_to_workbook_exports(model: dict, l7_read_csv: Callable[..., pd.DataFrame]) -> pd.Series:
    out: dict[str, float] = {}

    rev_ref = l7_read_csv("revenue_forecast_workbook.csv")
    rev_ref = rev_ref[pd.to_numeric(rev_ref["Forecast Period"], errors="coerce").notna()].copy()
    rev_ref["Forecast Period"] = pd.to_numeric(rev_ref["Forecast Period"]).astype(int)
    for col in ["Total Revenue", "Automotive", "growth rate", "Energy", "Software", "Robotaxi"]:
        rev_ref[col] = pd.to_numeric(rev_ref[col], errors="raise")
    rev_cmp = model["revenue"].merge(rev_ref, on="Forecast Period", suffixes=("", " ref"))
    out["Revenue forecast (max abs gap)"] = float(
        max(
            (rev_cmp["Total Revenue"] - rev_cmp["Total Revenue ref"]).abs().max(),
            (rev_cmp["Automotive"] - rev_cmp["Automotive ref"]).abs().max(),
            (rev_cmp["growth rate"] - rev_cmp["growth rate ref"]).abs().max(),
            (rev_cmp["Energy"] - rev_cmp["Energy ref"]).abs().max(),
            (rev_cmp["Software"] - rev_cmp["Software ref"]).abs().max(),
            (rev_cmp["Robotaxi"] - rev_cmp["Robotaxi ref"]).abs().max(),
        )
    )

    gm_ref = l7_read_csv("gross_margin_workbook_targets.csv")
    for c in gm_ref.columns:
        if c != "Forecast Period":
            gm_ref[c] = pd.to_numeric(gm_ref[c], errors="raise")
    gm_ref["Forecast Period"] = pd.to_numeric(gm_ref["Forecast Period"]).astype(int)
    gm_cmp = model["gross_margin"].merge(
        gm_ref[["Forecast Period", "Total gross profit", "Total gross margin"]],
        on="Forecast Period",
        suffixes=("", " ref"),
    )
    out["Gross margin sheet (max abs gap)"] = float(
        max(
            (gm_cmp["Total gross profit"] - gm_cmp["Total gross profit ref"]).abs().max(),
            (gm_cmp["Total gross margin"] - gm_cmp["Total gross margin ref"]).abs().max(),
        )
    )

    op_ref = l7_read_csv("operating_expense_workbook_targets.csv")
    for c in op_ref.columns:
        op_ref[c] = pd.to_numeric(op_ref[c], errors="raise")
    op_ref["Forecast Period"] = op_ref["Forecast Period"].astype(int)
    op_cmp = model["operating_expense"].merge(
        op_ref[
            [
                "Forecast Period",
                "R&D",
                "Total operating expense",
                "Operating margin",
                "Operating profit (EBIT)",
            ]
        ],
        on="Forecast Period",
        suffixes=("", " ref"),
    )
    out["Operating expense sheet (max abs gap)"] = float(
        max(
            (op_cmp["R&D"] - op_cmp["R&D ref"]).abs().max(),
            (op_cmp["Total operating expense"] - op_cmp["Total operating expense ref"]).abs().max(),
            (op_cmp["Operating margin"] - op_cmp["Operating margin ref"]).abs().max(),
            (op_cmp["Operating profit (EBIT)"] - op_cmp["Operating profit (EBIT) ref"]).abs().max(),
        )
    )

    fcff_ref = l7_read_csv("fcff_final_results_table.csv")
    for c in fcff_ref.columns:
        fcff_ref[c] = pd.to_numeric(fcff_ref[c], errors="raise")
    fcff_ref["Forecast Period"] = fcff_ref["Forecast Period"].astype(int)
    fcff_cmp = model["fcff"].merge(fcff_ref, on="Forecast Period", suffixes=("", " ref"))
    out["FCFF table (max abs gap)"] = float(
        max(
            (fcff_cmp["EBIT"] - fcff_cmp["EBIT ref"]).abs().max(),
            (fcff_cmp["EBIT(1-t)"] - fcff_cmp["EBIT(1-t) ref"]).abs().max(),
            (fcff_cmp["Reinvestment"] - fcff_cmp["Reinvestment ref"]).abs().max(),
            (fcff_cmp["FCFF"] - fcff_cmp["FCFF ref"]).abs().max(),
        )
    )

    iv_ref = l7_read_csv("final_results_intrinsic_value_components.csv").set_index("Unnamed: 0")["Value"]
    iv_calc = model["valuation"]["intrinsic_value_components"]
    out["Intrinsic value summary (max abs gap)"] = float((iv_calc - iv_ref.reindex(iv_calc.index)).abs().max())
    return pd.Series(out)
