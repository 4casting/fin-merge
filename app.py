import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
import io
import math
import datetime
import tempfile
import json

# ==========================================
# 0. KONFIGURATION & SESSION STATE
# ==========================================
st.set_page_config(page_title="Master Integrated ROA & Finance", layout="wide")

# Hilfsfunktion für Berechnungen
def safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value): return default
        return float(value)
    except: return default

# Initialisierung aller States aus Modell 1 & 2
if "history" not in st.session_state: st.session_state.history = []
if "simulation_results" not in st.session_state: st.session_state.simulation_results = None

# Modell 2 Tabellen Init
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
# 1. DER FINANZ-KERN (MODELL 2 LOGIK)
# ==========================================
def run_detailed_accounting(N_series, ARPU_series, T, params):
    """
    Berechnet GuV, Cashflow und Bilanz basierend auf einer Kundenkurve N.
    """
    results = []
    # State-Variablen für Bilanz/Finanzen
    debt = params['loan_initial']
    cash = params['equity']
    loss_carry = 0.0
    retained = 0.0
    fixed_assets = 0.0
    asset_reg = {"Laptop":[], "Smartphone":[], "Auto":[], "LKW":[], "Büro":[], "Misc":[]}
    prev_cc = {}
    
    jobs = params['jobs_df'].to_dict('records')
    ccs = params['cc_df'].to_dict('records')
    
    # ARPU/COGS Ratio aus Produkttabelle (vereinfacht für Simulation)
    cogs_ratio = params['cogs_ratio']
    
    for t in range(T):
        row = {"Jahr": t+1}
        n_t = N_series[t]
        arpu_t = ARPU_series[t]
        
        # 1. GuV - Umsatz & COGS
        rev = n_t * arpu_t
        row["Umsatz"] = rev
        row["Kunden"] = n_t
        cogs = rev * cogs_ratio
        row["Wareneinsatz (COGS)"] = cogs
        
        # 2. GuV - Personal & FTE Skalierung
        wage_idx = (1 + params['wage_inc']/100)**t
        pers_cost = 0.0
        hw_needs = {"Laptop": 0, "Smartphone": 0, "Auto": 0, "LKW": 0, "Büro": 0}
        
        # FTE Wachstum basierend auf Zielumsatz pro FTE
        base_ftes = sum(safe_float(j.get("FTE Jahr 1")) for j in jobs)
        target_fte = rev / params['target_rev_per_fte'] if params['target_rev_per_fte'] > 0 else 0
        
        curr_total_fte = 0
        for j in jobs:
            base = safe_float(j.get("FTE Jahr 1"))
            fte = max(base, target_fte * (base/base_ftes)) if base_ftes > 0 else 0
            curr_total_fte += fte
            pers_cost += safe_float(j.get("Jahresgehalt (€)")) * fte * wage_idx * (1 + params['lnk_pct']/100)
            for hw in hw_needs.keys():
                if j.get(hw): hw_needs[hw] += fte
        
        row["Personalkosten"] = pers_cost
        row["FTE Total"] = curr_total_fte
        
        # 3. GuV - Kostenstellen (OPEX)
        cc_sum = 0.0
        growth = (rev - results[-1]["Umsatz"])/results[-1]["Umsatz"] if t > 0 and results[-1]["Umsatz"] > 0 else 0.0
        for c in ccs:
            nm = c.get("Kostenstelle")
            base = safe_float(c.get("Grundwert Jahr 1 (€)"))
            coup = safe_float(c.get("Umsatz-Kopplung (%)"))/100
            last = prev_cc.get(nm, base)
            curr = base if t==0 else last * (1 + growth*coup)
            prev_cc[nm] = curr
            cc_sum += curr
        
        # CAC & Sonstiges OPEX
        total_opex = pers_cost + cc_sum + (n_t * params['cac'])
        row["Gesamtkosten (OPEX)"] = total_opex
        row["EBITDA"] = rev - cogs - total_opex
        
        # 4. Abschreibungen & CAPEX
        capex = 0.0; afa = 0.0
        # Misc Capex
        misc_p = params['capex_annual']; misc_ul = params['depreciation_misc']
        asset_reg["Misc"].append({"y":t, "v":misc_p, "ul":misc_ul})
        capex += misc_p
        
        as_conf = {"Laptop": ("price_laptop", "ul_laptop"), "Smartphone": ("price_phone", "ul_phone"), 
                   "Auto": ("price_car", "ul_car"), "LKW": ("price_truck", "ul_truck"), "Büro": ("price_desk", "ul_desk")}
        
        for k, (pk, uk) in as_conf.items():
            needed = hw_needs[k]; price = params[pk]; ul = params[uk]
            have = sum(x["amt"] for x in asset_reg[k] if (t - x["y"]) < x["ul"])
            buy = max(0, needed - have)
            if buy > 0:
                cost = buy * price; capex += cost
                asset_reg[k].append({"y":t, "amt":buy, "v":cost, "ul":ul})
        
        for k in asset_reg:
            for x in asset_reg[k]:
                if 0 <= (t - x["y"]) < x["ul"]: afa += x["v"] / x["ul"]
        
        row["Abschreibungen"] = afa
        row["Investitionen (Assets)"] = capex
        row["EBIT"] = row["EBITDA"] - afa
        
        # 5. Zinsen & Steuern
        intr = debt * (params['loan_rate']/100)
        ebt = row["EBIT"] - intr
        tax = 0.0
        if ebt < 0: loss_carry += abs(ebt)
        else:
            use = min(ebt, loss_carry); loss_carry -= use
            tax = (ebt - use) * (params['tax_rate']/100)
        row["Steuern"] = tax
        net = ebt - tax
        row["Jahresüberschuss"] = net
        
        # 6. Cashflow & Bilanz
        cf_op = net + afa
        cf_inv = -capex
        cash_start = cash
        pre_fin = cash_start + cf_op + cf_inv
        
        borrow = 0.0; repay = 0.0
        if pre_fin < params['min_cash']: borrow = params['min_cash'] - pre_fin
        elif pre_fin > params['min_cash'] and debt > 0: repay = min(debt, pre_fin - params['min_cash'])
        
        cash = pre_cash_end = pre_fin + borrow - repay
        debt = debt + borrow - repay
        fixed_assets = max(0, fixed_assets + capex - afa)
        retained += net
        
        row["Kasse"] = cash
        row["Bankdarlehen"] = debt
        row["Net Cash Change"] = cf_op + cf_inv + borrow - repay
        row["Eigenkapital"] = params['equity'] + retained
        row["Anlagevermögen"] = fixed_assets
        row["Summe Aktiva"] = fixed_assets + cash
        row["Summe Passiva"] = row["Eigenkapital"] + debt
        row["Kreditaufnahme"] = borrow
        row["Tilgung"] = repay
        row["Verb. LL"] = 0 
        
        results.append(row)
    
    return pd.DataFrame(results)

# ==========================================
# 2. SIMULATIONS-KERN (MODELL 1 LOGIK)
# ==========================================
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   mode='static', trigger_val=0.05, fallback_params=None,
                   check_mode='continuous', check_year=3, growth_metric='share_of_m',
                   switch_config=None):
    N = [0.0] * T
    W = [0.0] * T
    A = [ARPU] * T # Trace ARPU changes
    N[0] = start
    W[0] = N[0] * ARPU - Fixed_Cost

    option_exercised = False
    project_is_dead = False
    curr_p, curr_q, curr_C = p, q, C
    curr_ARPU, curr_kappa, curr_Delta_CM = ARPU, kappa, Delta_CM
    curr_FC, curr_M = Fixed_Cost, M
    growth_history = []

    for t in range(1, T):
        if project_is_dead:
            N[t] = 0.0; W[t] = 0.0; A[t] = 0.0; continue 

        N_prev = N[t-1]
        potential_acquisition = (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev)
        if potential_acquisition < 0: potential_acquisition = 0
        
        current_rate = potential_acquisition / curr_M if growth_metric == 'share_of_m' else (potential_acquisition / N_prev if N_prev > 0 else 0.0)

        # Trigger Prüfung
        if mode != 'static' and not option_exercised:
            is_check = (check_mode == 'specific' and t == check_year) or (check_mode == 'continuous' and t >= check_year)
            if is_check:
                all_rates = growth_history + [current_rate]
                avg_growth = sum(all_rates) / len(all_rates) if all_rates else 0
                if avg_growth < trigger_val:
                    option_exercised = True
                    if mode == 'switch' and fallback_params and switch_config:
                        # Switch Logic
                        delta_p = (fallback_params['ARPU'] - curr_ARPU) / curr_ARPU if curr_ARPU > 0 else 0.0
                        zone = 'zone1' if delta_p <= switch_config['thresh_low'] else ('zone2' if delta_p <= switch_config['thresh_high'] else 'zone3')
                        suffix = "_gf" if switch_config['grandfathering'] else "_nogf"
                        shock = switch_config[f'shock_{zone}{suffix}']
                        curr_q = fallback_params['q'] * switch_config[f'q_mult_{zone}{suffix}']
                        curr_p, curr_C, curr_ARPU = fallback_params['p'], fallback_params['C'], fallback_params['ARPU']
                        curr_FC = fallback_params['Fixed_Cost']
                        N_prev = N_prev * (1.0 - shock)
                        potential_acquisition = (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev)
                    elif mode == 'abandon':
                        project_is_dead = True; N[t] = 0.0; W[t] = 0.0; A[t] = 0.0; continue

        growth_history.append(current_rate)
        N[t] = min(curr_M, N_prev * (1 - curr_C) + potential_acquisition)
        A[t] = curr_ARPU
        W[t] = (N[t] * curr_ARPU) - (potential_acquisition * curr_kappa * curr_Delta_CM) - curr_FC
        
    return N, A, sum(W), option_exercised

# ==========================================
# 3. PDF REPORT GENERATOR (MODELL 2 STYLE)
# ==========================================
class PDFReport(FPDF):
    def fix_text(self, text):
        return str(text).replace("€", "EUR").encode('latin-1', 'replace').decode('latin-1')
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, self.fix_text('Strategisches Finanzmodell & ROA Report'), 0, 1, 'C')
        self.ln(10)
    def section_title(self, title):
        self.set_font('Arial', 'B', 12); self.set_fill_color(230, 240, 255)
        self.cell(0, 10, self.fix_text(title), 0, 1, 'L', 1); self.ln(4)
    def add_df_table(self, df, col_widths=None):
        self.set_font('Arial', 'B', 8); self.set_fill_color(240, 240, 240)
        if not col_widths: w = 277 / len(df.columns); widths = [w] * len(df.columns)
        else: widths = col_widths
        for i, col in enumerate(df.columns): self.cell(widths[i], 7, self.fix_text(str(col)), 1, 0, 'C', 1)
        self.ln(); self.set_font('Arial', '', 8)
        for _, row in df.iterrows():
            for i, col in enumerate(df.columns):
                val = row[col]; txt = f"{val:,.0f}" if isinstance(val, (int, float)) else str(val)
                self.cell(widths[i], 6, self.fix_text(txt), 1, 0, 'R')
            self.ln()
        self.ln(5)

# ==========================================
# 4. UI LAYOUT & TABS
# ==========================================
st.title("💠 Integrated Real Option Analysis & Financial Master Model")

tabs = st.tabs(["📊 Simulation & ROA", "👥 Personal & Assets", "📦 Produkte & Kostenstellen", "⚙️ ROA Konfiguration", "📑 Finanz-Reports"])

# --- TAB: ROA CONFIG ---
with tabs[3]:
    st.header("Real Options Switch Matrix & Trigger")
    col1, col2 = st.columns(2)
    with col1:
        check_mode_in = st.selectbox("Trigger-Zeitpunkt", ["specific", "continuous"])
        metric_in = st.selectbox("Wachstumsmetrik", ["share_of_m", "relative"])
        trig_val_in = st.slider("Trigger Grenzwert", 0.0, 0.5, 0.05)
    with col2:
        gf_active = st.checkbox("Grandfathering (Kein Churn bei Switch)")
        th_low = st.number_input("Grenze Sicherheitszone", 0.0, 1.0, 0.1)
        th_high = st.number_input("Grenze Gefahrenzone", 0.0, 1.0, 0.2)
    
    st.write("Schock-Parameter:")
    c_z1, c_z2, c_z3 = st.columns(3)
    s1 = c_z1.number_input("Shock Zone 1", 0.0, 1.0, 0.02)
    q1 = c_z1.number_input("q-Mult Zone 1", 0.0, 2.0, 1.0)
    s2 = c_z2.number_input("Shock Zone 2", 0.0, 1.0, 0.10)
    q2 = c_z2.number_input("q-Mult Zone 2", 0.0, 2.0, 0.8)
    s3 = c_z3.number_input("Shock Zone 3", 0.0, 1.0, 0.30)
    q3 = c_z3.number_input("q-Mult Zone 3", 0.0, 2.0, 0.5)

    switch_config = {
        'grandfathering': gf_active, 'thresh_low': th_low, 'thresh_high': th_high,
        'shock_zone1_nogf': s1, 'q_mult_zone1_nogf': q1, 'shock_zone1_gf': 0.0, 'q_mult_zone1_gf': q1,
        'shock_zone2_nogf': s2, 'q_mult_zone2_nogf': q2, 'shock_zone2_gf': 0.0, 'q_mult_zone2_gf': q2,
        'shock_zone3_nogf': s3, 'q_mult_zone3_nogf': q3, 'shock_zone3_gf': 0.0, 'q_mult_zone3_gf': q3,
    }

# --- TAB: SIMULATION & ROA ---
with tabs[0]:
    st.header("Eingaben für Standard (A) & Fighter (B)")
    col_a, col_b = st.columns(2)
    
    def range_input(lbl, min_v, max_v, key):
        c1, c2 = st.columns(2)
        return (c1.number_input(f"{lbl} Min", min_v, key=f"{key}_min"), c2.number_input(f"{lbl} Max", max_v, key=f"{key}_max"))

    with col_a:
        st.subheader("Option A: Standard (Fallback)")
        p_a = range_input("p", 0.005, 0.010, "pa")
        q_a = range_input("q", 0.15, 0.25, "qa")
        c_a = range_input("Churn", 0.03, 0.05, "ca")
        arpu_a = range_input("ARPU", 3800.0, 4200.0, "ar_a")
        fc_a = range_input("Fixkosten p.a.", 140000.0, 160000.0, "fc_a")
    with col_b:
        st.subheader("Option B: Fighter (Start)")
        p_b = range_input("p", 0.030, 0.050, "pb")
        q_b = range_input("q", 0.20, 0.30, "qb")
        c_b = range_input("Churn", 0.08, 0.12, "cb")
        arpu_b = range_input("ARPU", 3000.0, 3500.0, "ar_b")
        fc_b = range_input("Fixkosten p.a.", 180000.0, 200000.0, "fc_b")
    
    st.divider()
    M_in = st.number_input("Marktpotenzial (M)", 100, 10000, 500)
    T_in = st.slider("Zeitraum (Jahre)", 5, 30, 10)
    sim_btn = st.button("Simulation & ROA Analyse starten", type="primary")

# --- TAB: PERSONAL & ASSETS ---
with tabs[1]:
    st.subheader("Personal & Recruiting Strategie")
    st.session_state["current_jobs_df"] = st.data_editor(st.session_state["current_jobs_df"], num_rows="dynamic", use_container_width=True)
    target_rev_fte = st.number_input("Ziel-Umsatz pro FTE (€)", value=120000)
    
    st.subheader("Asset-Konfiguration (Hardware/Invest)")
    c1, c2, c3 = st.columns(3)
    p_lp = c1.number_input("Laptop Preis", value=1500); ul_lp = c1.number_input("Laptop Jahre", value=3)
    p_ph = c2.number_input("Handy Preis", value=800); ul_ph = c2.number_input("Handy Jahre", value=2)
    p_car = c3.number_input("PKW Preis", value=35000); ul_car = c3.number_input("PKW Jahre", value=6)

# --- TAB: PRODUKTE & KOSTENSTELLEN ---
with tabs[2]:
    st.session_state["products_df"] = st.data_editor(st.session_state["products_df"], num_rows="dynamic", use_container_width=True)
    st.session_state["cost_centers_df"] = st.data_editor(st.session_state["cost_centers_df"], num_rows="dynamic", use_container_width=True)
    
    st.subheader("Finanzielle Rahmenbedingungen")
    cf1, cf2, cf3 = st.columns(3)
    equity = cf1.number_input("Eigenkapital", value=50000)
    loan = cf1.number_input("Start-Kredit", value=0)
    tax_rate = cf2.number_input("Steuersatz %", value=25.0)
    min_cash = cf2.number_input("Min. Kasse", value=10000)
    wage_inc = cf3.number_input("Lohnsteigerung %", value=2.0)
    lnk_pct = cf3.number_input("Lohnnebenkosten %", value=25.0)

# ==========================================
# 5. SIMULATIONS-AUSFÜHRUNG & INTEGRATION
# ==========================================
if sim_btn:
    def rnd(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1])
    
    params_A = {'M': M_in, 'p': p_a, 'q': q_a, 'C': c_a, 'ARPU': arpu_a, 'kappa': 0.1, 'Delta_CM': 100, 'Fixed_Cost': fc_a}
    params_B = {'M': M_in, 'p': p_b, 'q': q_b, 'C': c_b, 'ARPU': arpu_b, 'kappa': 0.1, 'Delta_CM': 100, 'Fixed_Cost': fc_b}
    
    scenarios = [("Standard (A)", params_A, 'static', None, 'blue'), 
                 ("Fighter (B)", params_B, 'static', None, 'red'),
                 ("Switch Option", params_B, 'switch', params_A, 'green'),
                 ("Abandon Option", params_B, 'abandon', None, 'black')]
    
    all_res = {}
    bar = st.progress(0)
    
    for idx, (name, p_rng, mode, fb_rng, color) in enumerate(scenarios):
        sim_sums, sim_N, sim_A = [], [], []
        for _ in range(500): # Monte Carlo Runs
            curr = {k: rnd(v) if isinstance(v, tuple) else v for k, v in p_rng.items()}
            fb = {k: rnd(v) if isinstance(v, tuple) else v for k, v in fb_rng.items()} if fb_rng else None
            N_t, A_t, tot, exc = run_simulation(**curr, start=1, T=T_in, mode=mode, trigger_val=trig_val_in, 
                                               fallback_params=fb, switch_config=switch_config)
            sim_sums.append(tot); sim_N.append(N_t); sim_A.append(A_t)
        
        # Bestimme Base/Worst/Best für detailliertes Accounting
        idx_base = np.argsort(sim_sums)[len(sim_sums)//2]
        idx_worst = np.argsort(sim_sums)[int(len(sim_sums)*0.05)]
        idx_best = np.argsort(sim_sums)[int(len(sim_sums)*0.95)]
        
        # Produktdaten für Accounting aggregieren
        df_p = st.session_state["products_df"]
        total_take = df_p["Take Rate (%)"].sum() / 100
        avg_cogs_ratio = (df_p["Herstellungskosten (COGS €)"] * df_p["Take Rate (%)"]).sum() / (df_p["Preis (€)"] * df_p["Take Rate (%)"]).sum()

        acc_params = {
            'equity': equity, 'loan_initial': loan, 'tax_rate': tax_rate, 'min_cash': min_cash,
            'wage_inc': wage_inc, 'lnk_pct': lnk_pct, 'target_rev_per_fte': target_rev_fte,
            'cac': 250, 'capex_annual': 2000, 'depreciation_misc': 5,
            'price_laptop': p_lp, 'ul_laptop': ul_lp, 'price_phone': p_ph, 'ul_phone': ul_ph,
            'price_car': p_car, 'ul_car': ul_car, 'price_truck': 50000, 'ul_truck': 8, 'price_desk': 1000, 'ul_desk': 10,
            'jobs_df': st.session_state["current_jobs_df"], 'cc_df': st.session_state["cost_centers_df"],
            'cogs_ratio': avg_cogs_ratio
        }

        all_res[name] = {
            "color": color, "sums": sim_sums,
            "Base": run_detailed_accounting(sim_N[idx_base], sim_A[idx_base], T_in, acc_params),
            "Worst": run_detailed_accounting(sim_N[idx_worst], sim_A[idx_worst], T_in, acc_params),
            "Best": run_detailed_accounting(sim_N[idx_best], sim_A[idx_best], T_in, acc_params)
        }
        bar.progress((idx+1)/4)
    
    st.session_state.simulation_results = all_res

# --- TAB: FINANZ-REPORTS (AUSGABE) ---
with tabs[4]:
    if st.session_state.simulation_results:
        res = st.session_state.simulation_results
        sel_scen = st.selectbox("Szenario für Detail-Report wählen", list(res.keys()))
        sel_case = st.radio("Fall", ["Base", "Worst", "Best"], horizontal=True)
        
        df_final = res[sel_scen][sel_case]
        
        st.subheader(f"Detaillierter Finanzplan: {sel_scen} ({sel_case} Case)")
        
        k1, k2, k3, k4 = st.columns(4)
        last = df_final.iloc[-1]
        k1.metric("Umsatz J-End", f"{last['Umsatz']:,.0f} €")
        k2.metric("EBITDA J-End", f"{last['EBITDA']:,.0f} €")
        k3.metric("Kasse J-End", f"{last['Kasse']:,.0f} €")
        k4.metric("Eigenkapital J-End", f"{last['Eigenkapital']:,.0f} €")
        
        st.markdown("### 📑 Gewinn- und Verlustrechnung (GuV)")
        st.dataframe(df_final.set_index("Jahr")[["Umsatz", "Wareneinsatz (COGS)", "Gesamtkosten (OPEX)", "EBITDA", "Abschreibungen", "EBIT", "Steuern", "Jahresüberschuss"]].T.style.format("{:,.0f}"))
        
        st.markdown("### 💰 Cashflow-Rechnung")
        st.dataframe(df_final.set_index("Jahr")[["Jahresüberschuss", "Abschreibungen", "Investitionen (Assets)", "Net Cash Change", "Kasse"]].T.style.format("{:,.0f}"))
        
        st.markdown("### ⚖️ Bilanz")
        st.dataframe(df_final.set_index("Jahr")[["Anlagevermögen", "Kasse", "Summe Aktiva", "Eigenkapital", "Bankdarlehen", "Summe Passiva"]].T.style.format("{:,.0f}"))

        # PDF GENERATOR
        if st.button("📄 Detaillierten PDF-Report generieren"):
            pdf = PDFReport(orientation='L', unit='mm', format='A4')
            pdf.add_page()
            pdf.section_title(f"Management Summary: {sel_scen} ({sel_case})")
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, f"Umsatz Jahr {T_in}: {last['Umsatz']:,.0f} EUR | EBITDA: {last['EBITDA']:,.0f} EUR", 0, 1)
            
            pdf.add_page()
            pdf.section_title("Gewinn- und Verlustrechnung")
            pdf.add_df_table(df_final[["Jahr", "Umsatz", "Wareneinsatz (COGS)", "EBITDA", "EBIT", "Jahresüberschuss"]])
            
            pdf.add_page()
            pdf.section_title("Cashflow & Bilanz")
            pdf.add_df_table(df_final[["Jahr", "Jahresüberschuss", "Investitionen (Assets)", "Kasse", "Eigenkapital", "Bankdarlehen"]])
            
            buf = io.BytesIO()
            pdf_out = pdf.output(dest='S').encode('latin-1')
            st.download_button("📄 PDF Report Herunterladen", pdf_out, f"Report_{sel_scen}.pdf", "application/pdf")

    else:
        st.info("Bitte starten Sie zuerst die Simulation im ersten Tab.")
