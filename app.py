import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import io
import math
import datetime
import json
from fpdf import FPDF
import tempfile

# ==========================================
# 0. STREAMLIT KONFIGURATION & LOGIN
# ==========================================
st.set_page_config(page_title="Master Integrated ROA & Finance Factory", layout="wide")

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]:
        return True
    st.markdown("## 🔒 Integrated Valuation Model - Login")
    col1, col2 = st.columns([1, 2])
    with col1:
        user = st.text_input("Benutzername")
        pwd = st.text_input("Passwort", type="password")
        if st.button("Anmelden", type="primary"):
            if user == "admin" and pwd == "123":
                st.session_state["password_correct"] = True
                st.rerun()
            else: st.error("Zugangsdaten falsch.")
    return False

if not check_password(): st.stop()

# ==========================================
# 1. INITIALISIERUNG SESSION STATE (VOLLER UMFANG)
# ==========================================
# Wir initialisieren alle Keys, um KeyErrors zu vermeiden
DEFAULTS = {
    "sam": 50000.0, "cap_pct": 5.0, "equity": 50000.0, "loan_initial": 0.0, 
    "min_cash": 10000.0, "loan_rate": 5.0, "wage_inc": 2.0, "inflation": 2.0, 
    "lnk_pct": 25.0, "target_rev_per_fte": 120000.0, "tax_rate": 25.0, 
    "dso": 30, "dpo": 30, "cac": 250.0, "capex_annual": 2000, "depreciation_misc": 5,
    "price_laptop": 1500, "ul_laptop": 3, "price_phone": 800, "ul_phone": 2,
    "price_car": 35000, "ul_car": 6, "price_truck": 50000, "ul_truck": 8,
    "price_desk": 1000, "ul_desk": 10,
    # ROA Defaults
    "p_min_a": 0.005, "p_max_a": 0.010, "q_min_a": 0.150, "q_max_a": 0.250,
    "C_min_a": 0.030, "C_max_a": 0.050, "p_min_b": 0.030, "p_max_b": 0.050,
    "q_min_b": 0.200, "q_max_b": 0.300, "C_min_b": 0.080, "C_max_b": 0.120,
    "kappa": 0.10, "dcm": 100.0, "T_val": 10, "check_year": 3, "trig_val": 0.03,
    "th_low": 0.1, "th_high": 0.2, "gf_active": False
}

for k, v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

if "current_jobs_df" not in st.session_state:
    roles = [
        {"Job Titel": "CEO", "Jahresgehalt (€)": 100000, "FTE Jahr 1": 1.0, "Laptop": True, "Smartphone": True, "Auto": True, "LKW": False, "Büro": True, "Sonstiges (€)": 0},
        {"Job Titel": "Sales", "Jahresgehalt (€)": 60000, "FTE Jahr 1": 1.0, "Laptop": True, "Smartphone": True, "Auto": True, "LKW": False, "Büro": True, "Sonstiges (€)": 500},
        {"Job Titel": "Tech", "Jahresgehalt (€)": 55000, "FTE Jahr 1": 2.0, "Laptop": True, "Smartphone": False, "Auto": False, "LKW": False, "Büro": True, "Sonstiges (€)": 200},
    ]
    st.session_state["current_jobs_df"] = pd.DataFrame(roles)

if "products_df" not in st.session_state:
    st.session_state["products_df"] = pd.DataFrame([
        {"Produkt": "Basis Abo", "Preis (€)": 50.0, "Avg. Rabatt (%)": 0.0, "Herstellungskosten (COGS €)": 5.0, "Take Rate (%)": 70.0, "Wiederkauf Rate (%)": 95.0, "Wiederkauf alle (Monate)": 1},
        {"Produkt": "Pro Abo", "Preis (€)": 150.0, "Avg. Rabatt (%)": 5.0, "Herstellungskosten (COGS €)": 20.0, "Take Rate (%)": 30.0, "Wiederkauf Rate (%)": 90.0, "Wiederkauf alle (Monate)": 1},
    ])

if "cost_centers_df" not in st.session_state:
    st.session_state["cost_centers_df"] = pd.DataFrame([
        {"Kostenstelle": "Server & IT", "Grundwert Jahr 1 (€)": 1200, "Umsatz-Kopplung (%)": 10},
        {"Kostenstelle": "Marketing (Fix)", "Grundwert Jahr 1 (€)": 5000, "Umsatz-Kopplung (%)": 0},
    ])

# ==========================================
# 2. HILFSFUNKTIONEN (ARPU & KOSTEN)
# ==========================================
def calculate_arpu_from_products(df):
    arpu, cogs = 0.0, 0.0
    for _, p in df.iterrows():
        pr = safe_float(p["Preis (€)"])
        tr = safe_float(p["Take Rate (%)"]) / 100
        re = safe_float(p["Wiederkauf Rate (%)"]) / 100
        mo = safe_float(p["Wiederkauf alle (Monate)"])
        freq = (12 / mo * re) if mo > 0 else 1.0
        arpu += pr * tr * freq
        cogs += safe_float(p["Herstellungskosten (COGS €)"]) * tr * freq
    return arpu, (cogs/arpu if arpu > 0 else 0.15)

def get_base_fixed_costs():
    # Personal + Kostenstellen
    jobs = st.session_state["current_jobs_df"]
    cc = st.session_state["cost_centers_df"]
    staff_costs = (jobs["Jahresgehalt (€)"] * jobs["FTE Jahr 1"]).sum() * (1 + st.session_state["lnk_pct"]/100)
    cc_costs = cc["Grundwert Jahr 1 (€)"].sum()
    return staff_costs + cc_costs

# ==========================================
# 3. KERN-LOGIK: SIMULATION & ACCOUNTING
# ==========================================
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   mode='static', trigger_val=0.05, fallback_params=None,
                   check_mode='continuous', check_year=3, growth_metric='share_of_m',
                   switch_config=None):
    N, W, ARPU_trace = [0.0]*T, [0.0]*T, [ARPU]*T
    N[0] = start
    W[0] = N[0] * ARPU - Fixed_Cost
    option_exercised, project_is_dead = False, False
    curr_p, curr_q, curr_C, curr_ARPU, curr_FC = p, q, C, ARPU, Fixed_Cost
    growth_history = []

    for t in range(1, T):
        if project_is_dead:
            N[t], W[t], ARPU_trace[t] = 0.0, 0.0, 0.0
            continue 
        
        N_prev = N[t-1]
        pot_acq = (curr_p + curr_q * (N_prev / M)) * (M - N_prev)
        if pot_acq < 0: pot_acq = 0
        curr_rate = pot_acq / M if growth_metric == 'share_of_m' else (pot_acq / N_prev if N_prev > 0 else 0.0)

        if mode != 'static' and not option_exercised:
            if (check_mode == 'specific' and t == check_year) or (check_mode == 'continuous' and t >= check_year):
                avg_g = sum(growth_history + [curr_rate]) / (len(growth_history) + 1)
                if avg_g < trigger_val:
                    option_exercised = True
                    if mode == 'switch' and fallback_params and switch_config:
                        delta_p = (fallback_params['ARPU'] - curr_ARPU) / curr_ARPU if curr_ARPU > 0 else 0.0
                        zone = 'zone1' if delta_p <= switch_config['thresh_low'] else ('zone2' if delta_p <= switch_config['thresh_high'] else 'zone3')
                        sfx = "_gf" if switch_config['grandfathering'] else "_nogf"
                        shock, q_mult = switch_config[f'shock_{zone}{sfx}'], switch_config[f'q_mult_{zone}{sfx}']
                        curr_p, curr_q, curr_C, curr_ARPU, curr_FC = fallback_params['p'], fallback_params['q']*q_mult, fallback_params['C'], fallback_params['ARPU'], fallback_params['Fixed_Cost']
                        N_prev *= (1.0 - shock)
                        pot_acq = max(0, (curr_p + curr_q * (N_prev / M)) * (M - N_prev))
                    elif mode == 'abandon': project_is_dead = True; continue

        growth_history.append(curr_rate)
        N[t] = min(M, N_prev * (1 - curr_C) + pot_acq)
        ARPU_trace[t] = curr_ARPU
        W[t] = (N[t] * curr_ARPU) - (pot_acq * kappa * Delta_CM) - curr_FC
    return N, ARPU_trace, sum(W), option_exercised

def calculate_financials(N_curve, ARPU_curve, T_years, cogs_ratio):
    res = []
    cash, debt, loss_carry, retained, fixed_assets = st.session_state["equity"], st.session_state["loan_initial"], 0.0, 0.0, 0.0
    asset_reg = {"Laptop":[], "Smartphone":[], "Auto":[], "LKW":[], "Büro":[], "Misc":[]}
    prev_cc = {}
    
    for t in range(T_years):
        n_t, arpu_t = N_curve[t], ARPU_curve[t]
        row = {"Jahr": t+1, "Kunden": n_t, "Umsatz": n_t * arpu_t}
        rev = row["Umsatz"]
        row["Wareneinsatz (COGS)"] = rev * cogs_ratio
        
        # Personal
        wage_idx = (1 + st.session_state["wage_inc"]/100)**t
        pers_cost, total_fte, hw_needs = 0.0, 0.0, {k:0 for k in ["Laptop", "Smartphone", "Auto", "LKW", "Büro"]}
        jobs = st.session_state["current_jobs_df"].to_dict('records')
        target_fte = rev / st.session_state["target_rev_per_fte"] if st.session_state["target_rev_per_fte"] > 0 else 0
        base_sum = sum(safe_float(j.get("FTE Jahr 1")) for j in jobs)
        for j in jobs:
            fte = max(safe_float(j.get("FTE Jahr 1")), target_fte * (safe_float(j.get("FTE Jahr 1"))/base_sum)) if base_sum > 0 else 0
            total_fte += fte
            pers_cost += safe_float(j.get("Jahresgehalt (€)")) * fte * wage_idx * (1 + st.session_state["lnk_pct"]/100)
            for hw in hw_needs.keys(): 
                if j.get(hw): hw_needs[hw] += fte
        row["Personalkosten"], row["FTE Total"] = pers_cost, total_fte
        
        # Kostenstellen
        cc_sum = 0.0
        growth = (rev - res[-1]["Umsatz"])/res[-1]["Umsatz"] if t > 0 and res[-1]["Umsatz"] > 0 else 0.0
        for c in st.session_state["cost_centers_df"].to_dict('records'):
            nm, base, coup = c.get("Kostenstelle"), safe_float(c.get("Grundwert Jahr 1 (€)")), safe_float(c.get("Umsatz-Kopplung (%)"))/100
            val = base if t==0 else prev_cc.get(nm, base) * (1 + growth*coup)
            prev_cc[nm] = val; cc_sum += val
        row["Gesamtkosten (OPEX)"] = pers_cost + cc_sum + (n_t * st.session_state["cac"])
        row["EBITDA"] = rev - row["Wareneinsatz (COGS)"] - row["Gesamtkosten (OPEX)"]
        
        # Assets
        capex, afa = st.session_state["capex_annual"], 0.0
        asset_reg["Misc"].append({"y":t, "v":capex, "ul":st.session_state["depreciation_misc"]})
        as_conf = {"Laptop": ("price_laptop", "ul_laptop"), "Smartphone": ("price_phone", "ul_phone"), "Auto": ("price_car", "ul_car"), "LKW": ("price_truck", "ul_truck"), "Büro": ("price_desk", "ul_desk")}
        for k, (pk, uk) in as_conf.items():
            n_hw, pr, ul = hw_needs[k], st.session_state[pk], st.session_state[uk]
            h_hw = sum(x["amt"] for x in asset_reg[k] if (t - x["y"]) < x["ul"])
            buy = max(0, n_hw - h_hw)
            if buy > 0:
                cost = buy * pr; capex += cost
                asset_reg[k].append({"y":t, "amt":buy, "v":cost, "ul":ul})
        for k in asset_reg:
            for x in asset_reg[k]:
                if 0 <= (t - x["y"]) < x["ul"]: afa += x["v"] / x["ul"]
        
        row["Abschreibungen"], row["Investitionen (Assets)"] = afa, capex
        row["EBIT"] = row["EBITDA"] - afa
        ebt = row["EBIT"] - (debt * st.session_state["loan_rate"]/100)
        tax = max(0, (ebt - min(ebt, loss_carry)) * st.session_state["tax_rate"]/100) if ebt > 0 else 0
        if ebt < 0: loss_carry += abs(ebt)
        else: loss_carry = max(0, loss_carry - ebt)
        row["Steuern"], row["Jahresüberschuss"] = tax, ebt - tax
        
        # Bilanz
        cf_op = row["Jahresüberschuss"] + afa
        pre_f = cash + cf_op - capex
        bor = max(0, st.session_state["min_cash"] - pre_f)
        rep = min(debt, max(0, pre_f - st.session_state["min_cash"])) if pre_f > st.session_state["min_cash"] else 0
        cash, debt = pre_f + bor - rep, debt + bor - rep
        fixed_assets += capex - afa
        retained += row["Jahresüberschuss"]
        row.update({"Kasse": cash, "Bankdarlehen": debt, "Eigenkapital": st.session_state["equity"] + retained, "Anlagevermögen": fixed_assets, "Summe Aktiva": fixed_assets + cash, "Summe Passiva": st.session_state["equity"] + retained + debt, "Net Cash Change": cf_op - capex + bor - rep})
        res.append(row)
    return pd.DataFrame(res)

# ==========================================
# 4. NAVIGATION & TABS
# ==========================================
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Modus", ["Zentrale Eingabe", "Simulation & Analyse", "Finanzberichte"])

# --- TAB: ZENTRALE EINGABE ---
if menu == "Zentrale Eingabe":
    st.header("1. Konfiguration Markt & Strategie")
    t1, t2, t3 = st.tabs(["Strategie A & B", "Personal & Assets", "Produkte & Kostenstellen"])
    with t1:
        st.subheader("Bass Diffusion Parameter")
        c1, c2 = st.columns(2)
        def r_in(lbl, min_v, max_v, sfx):
            cc1, cc2 = st.columns(2)
            return (cc1.number_input(f"{lbl} Min", value=min_v, format="%.3f", key=f"{lbl}_min_{sfx}"), 
                    cc2.number_input(f"{lbl} Max", value=max_v, format="%.3f", key=f"{lbl}_max_{sfx}"))
        with c1:
            st.markdown("**Option A: Standard (Fallback)**")
            p_a = r_in("p", 0.005, 0.010, "a"); q_a = r_in("q", 0.150, 0.250, "a"); c_a = r_in("C", 0.030, 0.050, "a")
        with c2:
            st.markdown("**Option B: Fighter (Start)**")
            p_b = r_in("p", 0.030, 0.050, "b"); q_b = r_in("q", 0.200, 0.300, "b"); c_b = r_in("C", 0.080, 0.120, "b")
        st.divider()
        st.subheader("Option Trigger & Switch Matrix")
        st.session_state["gf_active"] = st.checkbox("Grandfathering", value=st.session_state["gf_active"])
        cl1, cl2, cl3 = st.columns(3)
        st.session_state["trig_val"] = cl1.number_input("Trigger Grenzwert", value=st.session_state["trig_val"])
        st.session_state["check_year"] = cl2.number_input("Prüf-Jahr", value=st.session_state["check_year"])
        st.session_state["sam"] = cl3.number_input("Marktpotenzial (M)", value=st.session_state["sam"])
    with t2:
        st.subheader("Personal & Recruiting")
        st.session_state["current_jobs_df"] = st.data_editor(st.session_state["current_jobs_df"], num_rows="dynamic", use_container_width=True)
        st.session_state["target_rev_per_fte"] = st.number_input("Ziel-Umsatz pro FTE", value=st.session_state["target_rev_per_fte"])
        st.subheader("Anlagen & Assets")
        cl1, cl2, cl3 = st.columns(3)
        st.session_state["price_laptop"] = cl1.number_input("Laptop Preis", value=st.session_state["price_laptop"])
        st.session_state["price_car"] = cl2.number_input("PKW Preis", value=st.session_state["price_car"])
        st.session_state["capex_annual"] = cl3.number_input("Misc Capex p.a.", value=st.session_state["capex_annual"])
    with t3:
        st.subheader("Produkte & Deckungsbeiträge")
        st.session_state["products_df"] = st.data_editor(st.session_state["products_df"], num_rows="dynamic", use_container_width=True)
        st.subheader("Kostenstellen & Gemeinkosten")
        st.session_state["cost_centers_df"] = st.data_editor(st.session_state["cost_centers_df"], num_rows="dynamic", use_container_width=True)
        st.subheader("Finanzierung")
        st.session_state["equity"] = st.number_input("Eigenkapital", value=st.session_state["equity"])

# --- TAB: SIMULATION ---
if menu == "Simulation & Analyse":
    st.header("2. Monte Carlo & Real Option Analyse")
    if st.button("Simulation starten", type="primary", use_container_width=True):
        arpu_calc, cogs_calc = calculate_arpu_from_products(st.session_state["products_df"])
        fc_base = get_base_fixed_costs()
        st.info(f"Basis-Fixkosten: {fc_base:,.0f} € | Berechneter ARPU: {arpu_calc:,.0f} €")
        
        sw_conf = {
            'grandfathering': st.session_state.gf_active, 'thresh_low': st.session_state.th_low, 'thresh_high': st.session_state.th_high,
            'shock_zone1_nogf': 0.02, 'q_mult_zone1_nogf': 1.0, 'shock_zone1_gf': 0.0, 'q_mult_zone1_gf': 1.0,
            'shock_zone2_nogf': 0.10, 'q_mult_zone2_nogf': 0.8, 'shock_zone2_gf': 0.0, 'q_mult_zone2_gf': 0.8,
            'shock_zone3_nogf': 0.30, 'q_mult_zone3_nogf': 0.5, 'shock_zone3_gf': 0.0, 'q_mult_zone3_gf': 0.5,
        }
        
        # Pack Params
        pA = {'M': st.session_state.sam, 'p': (st.session_state.p_min_a, st.session_state.p_max_a), 'q': (st.session_state.q_min_a, st.session_state.q_max_a), 'C': (st.session_state.C_min_a, st.session_state.C_max_a), 'ARPU': arpu_calc, 'kappa': st.session_state.kappa, 'Delta_CM': st.session_state.dcm, 'Fixed_Cost': fc_base}
        pB = {'M': st.session_state.sam, 'p': (st.session_state.p_min_b, st.session_state.p_max_b), 'q': (st.session_state.q_min_b, st.session_state.q_max_b), 'C': (st.session_state.C_min_b, st.session_state.C_max_b), 'ARPU': arpu_calc*0.8, 'kappa': st.session_state.kappa, 'Delta_CM': st.session_state.dcm, 'Fixed_Cost': fc_base*1.2}
        
        res_store = {}; bar = st.progress(0)
        scenarios = [("Standard (A)", pA, 'static', None, 'blue'), ("Fighter (B)", pB, 'static', None, 'red'), ("Switch Option", pB, 'switch', pA, 'green'), ("Abandon Option", pB, 'abandon', None, 'black')]
        
        for idx, (name, pr, mode, fb, col) in enumerate(scenarios):
            n_iter = 500
            sim_sums, all_N, all_A = [], [], []
            for _ in range(n_iter):
                curr = {k: (np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v) for k, v in pr.items()}
                fbp = {k: (np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v) for k, v in fb.items()} if fb else None
                Nt, At, tot, _ = run_simulation(**curr, start=1, T=st.session_state.T_val, mode=mode, trigger_val=st.session_state.trig_val, fallback_params=fbp, check_mode='continuous', check_year=st.session_state.check_year, growth_metric='share_of_m', switch_config=sw_conf)
                sim_sums.append(tot); all_N.append(Nt); all_A.append(At)
            
            arrN = np.array(all_N); idx_b, idx_w, idx_bt = np.argsort(sim_sums)[n_iter//2], np.argsort(sim_sums)[int(n_iter*0.05)], np.argsort(sim_sums)[int(n_iter*0.95)]
            res_store[name] = {
                "sums": sim_sums, "color": col, "mean": np.mean(sim_sums),
                "avg_N": np.mean(arrN, axis=0), "p5_N": np.percentile(arrN, 5, axis=0), "p95_N": np.percentile(arrN, 95, axis=0),
                "fin_base": calculate_financials(arrN[idx_b], all_A[idx_b], st.session_state.T_val, cogs_calc),
                "fin_worst": calculate_financials(arrN[idx_w], all_A[idx_w], st.session_state.T_val, cogs_calc),
                "fin_best": calculate_financials(arrN[idx_bt], all_A[idx_bt], st.session_state.T_val, cogs_calc)
            }
            bar.progress((idx+1)/4)
        st.session_state.simulation_results = res_store

# --- TAB: FINANZBERICHTE ---
if menu == "Finanzberichte":
    if st.session_state.simulation_results:
        scen = st.selectbox("Strategie", list(st.session_state.simulation_results.keys()))
        case = st.radio("Szenario", ["Base", "Worst", "Best"], horizontal=True)
        df = st.session_state.simulation_results[scen][f"fin_{case.lower()}"]
        
        st.subheader(f"Finanzplan: {scen} ({case})")
        t_guv, t_cf, t_bil = st.tabs(["📑 GuV", "💰 Cashflow", "⚖️ Bilanz"])
        with t_guv: st.dataframe(df.set_index("Jahr")[["Umsatz", "Wareneinsatz (COGS)", "Gesamtkosten (OPEX)", "EBITDA", "Abschreibungen", "EBIT", "Steuern", "Jahresüberschuss"]].T.style.format("{:,.0f}"))
        with t_cf: st.dataframe(df.set_index("Jahr")[["Jahresüberschuss", "Abschreibungen", "Investitionen (Assets)", "Net Cash Change", "Kasse"]].T.style.format("{:,.0f}"))
        with t_bil: st.dataframe(df.set_index("Jahr")[["Anlagevermögen", "Kasse", "Summe Aktiva", "Eigenkapital", "Bankdarlehen", "Summe Passiva"]].T.style.format("{:,.0f}"))
