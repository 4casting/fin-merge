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
st.set_page_config(page_title="Master Integrated ROA & Finance", layout="wide")

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]:
        return True

    st.markdown("## 🔒 Finanz- & Bewertungsmodell - Login")
    col1, col2 = st.columns([1, 2])
    with col1:
        user = st.text_input("Benutzername")
        pwd = st.text_input("Passwort", type="password")
        if st.button("Anmelden", type="primary"):
            if user == "admin" and pwd == "123":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Zugangsdaten falsch.")
    return False

if not check_password():
    st.stop()

# ==========================================
# 1. INITIALISIERUNG SESSION STATE (FULL)
# ==========================================
if 'history' not in st.session_state: st.session_state.history = []
if 'simulation_results' not in st.session_state: st.session_state.simulation_results = None
if 'pdf_buffer' not in st.session_state: st.session_state.pdf_buffer = None

# Modell 2 Defaults
DEFAULTS_FIN = {
    "sam": 50000.0, "cap_pct": 5.0, "p_pct": 0.03, "q_pct": 0.38, "churn": 5.0, "manual_arpu": 1500.0,
    "equity": 50000.0, "loan_initial": 0.0, "min_cash": 10000.0, "loan_rate": 5.0,
    "wage_inc": 2.0, "inflation": 2.0, "lnk_pct": 25.0, "target_rev_per_fte": 120000.0,
    "tax_rate": 25.0, "dso": 30, "dpo": 30, "cac": 250.0,
    "capex_annual": 2000, "depreciation_misc": 5,
    "price_laptop": 1500, "ul_laptop": 3, "price_phone": 800, "ul_phone": 2,
    "price_car": 35000, "ul_car": 6, "price_truck": 50000, "ul_truck": 8,
    "price_desk": 1000, "ul_desk": 10,
}
for k, v in DEFAULTS_FIN.items():
    if k not in st.session_state: st.session_state[k] = v

if "current_jobs_df" not in st.session_state:
    roles = [
        {"Job Titel": "CEO", "Jahresgehalt (€)": 100000, "FTE Jahr 1": 1.0, "Laptop": True, "Smartphone": True, "Auto": True, "LKW": False, "Büro": True, "Sonstiges (€)": 0},
        {"Job Titel": "Sales", "Jahresgehalt (€)": 60000, "FTE Jahr 1": 1.0, "Laptop": True, "Smartphone": True, "Auto": True, "LKW": False, "Büro": True, "Sonstiges (€)": 500},
        {"Job Titel": "Tech", "Jahresgehalt (€)": 55000, "FTE Jahr 1": 2.0, "Laptop": True, "Smartphone": False, "Auto": False, "LKW": False, "Büro": True, "Sonstiges (€)": 200},
    ]
    for i in range(1, 10): roles.append({"Job Titel": f"Rolle {i}", "Jahresgehalt (€)": 0, "FTE Jahr 1": 0.0, "Laptop": False, "Smartphone": False, "Auto": False, "LKW": False, "Büro": False, "Sonstiges (€)": 0})
    st.session_state["current_jobs_df"] = pd.DataFrame(roles)

if "products_df" not in st.session_state:
    st.session_state["products_df"] = pd.DataFrame([
        {"Produkt": "Basis Abo", "Preis (€)": 50.0, "Avg. Rabatt (%)": 0.0, "Herstellungskosten (COGS €)": 5.0, "Take Rate (%)": 70.0, "Wiederkauf Rate (%)": 95.0, "Wiederkauf alle (Monate)": 1},
        {"Produkt": "Pro Abo", "Preis (€)": 150.0, "Avg. Rabatt (%)": 5.0, "Herstellungskosten (COGS €)": 20.0, "Take Rate (%)": 30.0, "Wiederkauf Rate (%)": 90.0, "Wiederkauf alle (Monate)": 1},
        {"Produkt": "Onboarding", "Preis (€)": 500.0, "Avg. Rabatt (%)": 0.0, "Herstellungskosten (COGS €)": 100.0, "Take Rate (%)": 50.0, "Wiederkauf Rate (%)": 0.0, "Wiederkauf alle (Monate)": 0},
    ])

if "cost_centers_df" not in st.session_state:
    st.session_state["cost_centers_df"] = pd.DataFrame([
        {"Kostenstelle": "Server & IT", "Grundwert Jahr 1 (€)": 1200, "Umsatz-Kopplung (%)": 10},
        {"Kostenstelle": "Marketing (Fix)", "Grundwert Jahr 1 (€)": 5000, "Umsatz-Kopplung (%)": 0},
        {"Kostenstelle": "Logistik", "Grundwert Jahr 1 (€)": 0, "Umsatz-Kopplung (%)": 5},
    ])

# ==========================================
# 2. KERN-LOGIK: SIMULATION & STATISTIK (MODELL 1)
# ==========================================
def safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value): return default
        return float(value)
    except: return default

def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   mode='static', trigger_val=0.05, fallback_params=None,
                   check_mode='continuous', check_year=3, growth_metric='share_of_m',
                   switch_config=None):
    N, W, ARPU_trace = [0.0]*T, [0.0]*T, [ARPU]*T
    N[0] = start
    W[0] = N[0] * ARPU - Fixed_Cost
    option_exercised, project_is_dead = False, False
    curr_p, curr_q, curr_C, curr_ARPU, curr_kappa, curr_Delta_CM, curr_FC, curr_M = p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, M
    growth_history = []

    for t in range(1, T):
        if project_is_dead:
            N[t], W[t], ARPU_trace[t] = 0.0, 0.0, 0.0
            continue 

        N_prev = N[t-1]
        potential_acquisition = (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev)
        if potential_acquisition < 0: potential_acquisition = 0
        current_rate = potential_acquisition / curr_M if growth_metric == 'share_of_m' else (potential_acquisition / N_prev if N_prev > 0 else 0.0)

        if mode != 'static' and not option_exercised:
            if (check_mode == 'specific' and t == check_year) or (check_mode == 'continuous' and t >= check_year):
                avg_growth = sum(growth_history + [current_rate]) / (len(growth_history) + 1)
                if avg_growth < trigger_val:
                    option_exercised = True
                    if mode == 'switch' and fallback_params and switch_config:
                        delta_p = (fallback_params['ARPU'] - curr_ARPU) / curr_ARPU if curr_ARPU > 0 else 0.0
                        zone = 'zone1' if delta_p <= switch_config['thresh_low'] else ('zone2' if delta_p <= switch_config['thresh_high'] else 'zone3')
                        sfx = "_gf" if switch_config['grandfathering'] else "_nogf"
                        shock, q_mult = switch_config[f'shock_{zone}{sfx}'], switch_config[f'q_mult_{zone}{sfx}']
                        curr_p, curr_q, curr_C, curr_ARPU, curr_kappa, curr_Delta_CM, curr_FC = fallback_params['p'], fallback_params['q']*q_mult, fallback_params['C'], fallback_params['ARPU'], fallback_params['kappa'], fallback_params['Delta_CM'], fallback_params['Fixed_Cost']
                        N_prev *= (1.0 - shock)
                        potential_acquisition = max(0, (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev))
                    elif mode == 'abandon':
                        project_is_dead = True
                        continue

        growth_history.append(current_rate)
        N[t] = min(curr_M, N_prev * (1 - curr_C) + potential_acquisition)
        ARPU_trace[t] = curr_ARPU
        W[t] = (N[t] * curr_ARPU) - (potential_acquisition * curr_kappa * curr_Delta_CM) - curr_FC
        
    return N, ARPU_trace, sum(W), option_exercised

def calculate_cochran_n(params_dict, T, mode='static', fallback=None, trigger=0.05, c_mode='continuous', c_year=3, g_metric='share_of_m', sw_conf=None):
    pilot_n = 200; results = []
    def get_val(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v
    for _ in range(pilot_n):
        curr = {k: get_val(v) for k, v in params_dict.items()}
        curr_fb = {k: get_val(v) for k, v in fallback.items()} if fallback else None
        _, _, val, _ = run_simulation(**curr, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=curr_fb, check_mode=c_mode, check_year=c_year, growth_metric=g_metric, switch_config=sw_conf)
        results.append(val)
    std, mean = np.std(results), np.mean(results)
    if mean == 0 or abs(mean*0.01) == 0: return 1000
    n = (1.96 * std / abs(mean * 0.01)) ** 2
    return max(int(math.ceil(n)), 1000)

def get_tornado_data(base_params, ranges, T, mode, trigger, fallback_ranges, c_mode, c_year, g_metric, sw_conf):
    def mid(v): return (v[0]+v[1])/2 if isinstance(v, tuple) else v
    base_inputs = {k: mid(v) for k, v in ranges.items()}
    fb_inputs = {k: mid(v) for k, v in fallback_ranges.items()} if fallback_ranges else None
    _, _, base_val, _ = run_simulation(**base_inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs, check_mode=c_mode, check_year=c_year, growth_metric=g_metric, switch_config=sw_conf)
    data = []
    for param, val_range in ranges.items():
        if not isinstance(val_range, tuple): continue
        for val, lbl in [(val_range[0], "Low"), (val_range[1], "High")]:
            inputs = base_inputs.copy(); inputs[param] = val
            _, _, v_res, _ = run_simulation(**inputs, start=1, T=T, mode=mode, trigger_val=trigger, fallback_params=fb_inputs, check_mode=c_mode, check_year=c_year, growth_metric=g_metric, switch_config=sw_conf)
            if lbl == "Low": v_low = v_res
            else: v_high = v_res
        data.append({"Parameter": param, "Low": v_low - base_val, "High": v_high - base_val, "Range": abs(v_high - v_low)})
    return pd.DataFrame(data).sort_values(by="Range", ascending=True), base_val

def get_regression_sensitivity(df_inputs, y_values):
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(df_inputs)
    model = LinearRegression(); model.fit(X_scaled, y_values)
    return pd.DataFrame({"Parameter": df_inputs.columns, "Beta": model.coef_}).sort_values(by="Beta", key=abs, ascending=True), model.score(X_scaled, y_values)

# ==========================================
# 3. BUCHHALTUNG & FINANZPLAN (MODELL 2)
# ==========================================
def calculate_detailed_financials(N_curve, ARPU_curve, T_years):
    res = []
    cash, debt, loss_carry, retained, fixed_assets = st.session_state["equity"], st.session_state["loan_initial"], 0.0, 0.0, 0.0
    asset_reg = {"Laptop":[], "Smartphone":[], "Auto":[], "LKW":[], "Büro":[], "Misc":[]}
    prev_cc = {}
    
    prod_df = st.session_state["products_df"]
    # Gewichteter COGS Ratio
    weighted_rev = (prod_df["Preis (€)"] * prod_df["Take Rate (%)"]).sum()
    weighted_cogs = (prod_df["Herstellungskosten (COGS €)"] * prod_df["Take Rate (%)"]).sum()
    cogs_ratio = weighted_cogs / weighted_rev if weighted_rev > 0 else 0.15

    for t in range(T_years):
        n_t, arpu_t = N_curve[t], ARPU_curve[t]
        row = {"Jahr": t+1, "Kunden": n_t, "Umsatz": n_t * arpu_t}
        rev = row["Umsatz"]
        row["Wareneinsatz (COGS)"] = rev * cogs_ratio
        
        # Personal & Assets
        wage_idx = (1 + st.session_state["wage_inc"]/100)**t
        pers_cost, total_fte, hw_needs = 0.0, 0.0, {k:0 for k in ["Laptop", "Smartphone", "Auto", "LKW", "Büro"]}
        jobs = st.session_state["current_jobs_df"].to_dict('records')
        target_fte = rev / st.session_state["target_rev_per_fte"] if st.session_state["target_rev_per_fte"] > 0 else 0
        base_ftes_sum = sum(safe_float(j.get("FTE Jahr 1")) for j in jobs)
        
        for j in jobs:
            base_fte = safe_float(j.get("FTE Jahr 1"))
            fte = max(base_fte, target_fte * (base_fte/base_ftes_sum)) if base_ftes_sum > 0 else 0
            total_fte += fte
            pers_cost += safe_float(j.get("Jahresgehalt (€)")) * fte * wage_idx * (1 + st.session_state["lnk_pct"]/100)
            for hw in hw_needs.keys(): 
                if j.get(hw): hw_needs[hw] += fte
        
        row["Personalkosten"], row["FTE Total"] = pers_cost, total_fte
        
        cc_sum = 0.0
        growth = (rev - res[-1]["Umsatz"])/res[-1]["Umsatz"] if t > 0 and res[-1]["Umsatz"] > 0 else 0.0
        for c in st.session_state["cost_centers_df"].to_dict('records'):
            nm, base, coup = c.get("Kostenstelle"), safe_float(c.get("Grundwert Jahr 1 (€)")), safe_float(c.get("Umsatz-Kopplung (%)"))/100
            val = base if t==0 else prev_cc.get(nm, base) * (1 + growth*coup)
            prev_cc[nm] = val; cc_sum += val
        
        row["Gesamtkosten (OPEX)"] = pers_cost + cc_sum + (n_t * st.session_state["cac"])
        row["EBITDA"] = rev - row["Wareneinsatz (COGS)"] - row["Gesamtkosten (OPEX)"]
        
        # Abschreibungen
        capex, afa = st.session_state["capex_annual"], 0.0
        asset_reg["Misc"].append({"y":t, "v":capex, "ul":st.session_state["depreciation_misc"]})
        as_conf = {"Laptop": ("price_laptop", "ul_laptop"), "Smartphone": ("price_phone", "ul_phone"), "Auto": ("price_car", "ul_car"), "LKW": ("price_truck", "ul_truck"), "Büro": ("price_desk", "ul_desk")}
        for k, (pk, uk) in as_conf.items():
            needed, price, ul = hw_needs[k], st.session_state[pk], st.session_state[uk]
            have = sum(x["amt"] for x in asset_reg[k] if (t - x["y"]) < x["ul"])
            buy = max(0, needed - have)
            if buy > 0:
                cost = buy * price; capex += cost
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
        
        # Cashflow & Bilanz
        cf_op = row["Jahresüberschuss"] + afa
        pre_fin = cash + cf_op - capex
        borrow = max(0, st.session_state["min_cash"] - pre_fin)
        repay = min(debt, max(0, pre_fin - st.session_state["min_cash"])) if pre_fin > st.session_state["min_cash"] else 0
        cash = pre_fin + borrow - repay
        debt += borrow - repay
        fixed_assets += capex - afa
        retained += row["Jahresüberschuss"]
        row.update({"Kasse": cash, "Bankdarlehen": debt, "Net Cash Change": cf_op - capex + borrow - repay, "Eigenkapital": st.session_state["equity"] + retained, "Anlagevermögen": fixed_assets, "Summe Aktiva": fixed_assets + cash, "Summe Passiva": st.session_state["equity"] + retained + debt})
        res.append(row)
    return pd.DataFrame(res)

# ==========================================
# 4. PDF REPORT KLASSE (MODELL 2 DESIGN)
# ==========================================
class PDFReport(FPDF):
    def fix_text(self, text): return str(text).replace("€", "EUR").encode('latin-1', 'replace').decode('latin-1')
    def header(self):
        self.set_font('Arial', 'B', 16); self.cell(0, 10, self.fix_text('Integrated Valuation & Financial Report'), 0, 1, 'C'); self.ln(10)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, self.fix_text(f'Seite {self.page_no()}'), 0, 0, 'C')
    def section_title(self, title):
        self.set_font('Arial', 'B', 12); self.set_fill_color(230, 240, 255); self.cell(0, 10, self.fix_text(title), 0, 1, 'L', 1); self.ln(4)
    def add_df_table(self, df, col_widths=None):
        self.set_font('Arial', 'B', 8); self.set_fill_color(240, 240, 240)
        widths = col_widths if col_widths else [277/len(df.columns)]*len(df.columns)
        for i, col in enumerate(df.columns): self.cell(widths[i], 7, self.fix_text(str(col)), 1, 0, 'C', 1)
        self.ln(); self.set_font('Arial', '', 8)
        for _, row in df.iterrows():
            for i, col in enumerate(df.columns):
                val = row[col]; txt = f"{val:,.0f}" if isinstance(val, (int, float)) else str(val)
                self.cell(widths[i], 6, self.fix_text(txt), 1, 0, 'R')
            self.ln()
        self.ln(5)

# ==========================================
# 5. UI LAYOUT & TABS
# ==========================================
st.title("💠 Integrated Strategic Valuation & Finance Model (V.Full)")

# SIDEBAR HISTORY
with st.sidebar:
    st.header("Verlauf & Daten")
    if st.button("🔄 Alles zurücksetzen"): st.session_state.clear(); st.rerun()
    up = st.file_uploader("JSON Konfiguration laden", type=["json"])
    if up: 
        d = json.load(up)
        for k,v in d.items(): 
            if k in DEFAULTS_FIN: st.session_state[k] = v
        st.success("Konfiguration geladen.")

# TABS
tab_roa, tab_jobs, tab_prod, tab_roa_conf, tab_res, tab_fin = st.tabs(["📈 ROA Inputs", "👥 Personal & Assets", "📦 Produkte & Kosten", "⚙️ Switch Matrix", "📊 Sim Ergebnisse", "📑 Finanzberichte"])

# --- TAB: ROA INPUTS ---
with tab_roa:
    st.header("ROA Strategie Parameter (Standard vs. Fighter)")
    c_a, c_b = st.columns(2)
    def r_in(lbl, min_v, max_v, sfx):
        cc1, cc2 = st.columns(2)
        return (cc1.number_input(f"{lbl} Min", value=min_v, format="%.3f", key=f"{lbl}_min_{sfx}"),
                cc2.number_input(f"{lbl} Max", value=max_v, format="%.3f", key=f"{lbl}_max_{sfx}"))

    with c_a:
        st.subheader("Option A: Standard")
        p_a = r_in("p", 0.005, 0.010, "a"); q_a = r_in("q", 0.15, 0.25, "a")
        c_a = r_in("C", 0.03, 0.05, "a"); arpu_a = r_in("ARPU", 3800.0, 4200.0, "a")
        fc_a = r_in("Fixed_Cost", 140000.0, 160000.0, "a")
    with c_b:
        st.subheader("Option B: Fighter")
        p_b = r_in("p", 0.030, 0.050, "b"); q_b = r_in("q", 0.20, 0.30, "b")
        c_b = r_in("C", 0.08, 0.12, "b"); arpu_b = r_in("ARPU", 3000.0, 3500.0, "b")
        fc_b = r_in("Fixed_Cost", 180000.0, 200000.0, "b")
    
    st.divider()
    col_glb1, col_glb2 = st.columns(2)
    with col_glb1:
        M_in = st.number_input("Marktpotenzial (M)", 100, 100000, 500)
        T_in = st.slider("Laufzeit (T)", 5, 30, 10)
    with col_glb2:
        kappa_in = st.number_input("Kannibalisierungsrate (kappa)", 0.0, 1.0, 0.1)
        dcm_in = st.number_input("Delta Margin (€)", 0.0, 500.0, 100.0)

# --- TAB: PERSONAL & ASSETS ---
with tab_jobs:
    st.session_state["current_jobs_df"] = st.data_editor(st.session_state["current_jobs_df"], num_rows="dynamic", use_container_width=True)
    st.number_input("Ziel-Umsatz pro FTE", key="target_rev_per_fte")
    st.divider()
    st.subheader("Hardware & Anlagen Preise")
    cc1, cc2, cc3 = st.columns(3)
    cc1.number_input("Laptop Preis", key="price_laptop"); cc1.number_input("Handy Preis", key="price_phone")
    cc2.number_input("Auto Preis", key="price_car"); cc2.number_input("LKW Preis", key="price_truck")
    cc3.number_input("Büro Preis", key="price_desk"); cc3.number_input("AfA Misc p.a.", key="capex_annual")

# --- TAB: PRODUKTE & KOSTEN ---
with tab_prod:
    st.session_state["products_df"] = st.data_editor(st.session_state["products_df"], num_rows="dynamic", use_container_width=True)
    st.session_state["cost_centers_df"] = st.data_editor(st.session_state["cost_centers_df"], num_rows="dynamic", use_container_width=True)
    st.divider()
    st.subheader("Finanzierung & Makro")
    ccc1, ccc2, ccc3 = st.columns(3)
    ccc1.number_input("Eigenkapital", key="equity"); ccc1.number_input("Kredit Start", key="loan_initial")
    ccc2.number_input("Steuersatz %", key="tax_rate"); ccc2.number_input("Kredit-Zins %", key="loan_rate")
    ccc3.number_input("Inflation %", key="inflation"); ccc3.number_input("Lohnsteigerung %", key="wage_inc")

# --- TAB: SWITCH MATRIX ---
with tab_roa_conf:
    st.header("Option Trigger & Switch Matrix")
    c_trig1, c_trig2 = st.columns(2)
    with c_trig1:
        check_mode_in = st.selectbox("Wann prüfen?", ["specific", "continuous"], key="check_mode_in")
        metric_in = st.selectbox("Metrik", ["share_of_m", "relative"], key="metric_in")
    with c_trig2:
        check_year_in = st.number_input("Start-Periode", 1, 30, 3, key="check_year_in")
        trig_val_in = st.number_input("Grenzwert (<)", 0.0, 1.0, 0.03, key="trig_val_in")
    
    st.markdown("### Schock-Zonen (Switch Matrix)")
    gf_active = st.checkbox("Grandfathering anwenden?", key="gf_active")
    th_low = st.number_input("Sicherheitszone bis %", 0.0, 1.0, 0.1, key="th_low")
    th_high = st.number_input("Warnzone bis %", 0.0, 1.0, 0.2, key="th_high")
    
    col_z1, col_z2, col_z3 = st.columns(3)
    s1_no = col_z1.number_input("Churn Schock Z1", 0.0, 1.0, 0.02, key="s1_no")
    q1_no = col_z1.number_input("q-Faktor Z1", 0.0, 1.5, 1.0, key="q1_no")
    s2_no = col_z2.number_input("Churn Schock Z2", 0.0, 1.0, 0.10, key="s2_no")
    q2_no = col_z2.number_input("q-Faktor Z2", 0.0, 1.5, 0.8, key="q2_no")
    s3_no = col_z3.number_input("Churn Schock Z3", 0.0, 1.0, 0.30, key="s3_no")
    q3_no = col_z3.number_input("q-Faktor Z3", 0.0, 1.5, 0.5, key="q3_no")

# --- SIMULATION STARTEN ---
st.divider()
if st.button("🚀 INTEGRATIONS-SIMULATION STARTEN", type="primary", use_container_width=True):
    sw_conf = {
        'grandfathering': st.session_state.gf_active, 'thresh_low': st.session_state.th_low, 'thresh_high': st.session_state.th_high,
        'shock_zone1_nogf': st.session_state.s1_no, 'q_mult_zone1_nogf': st.session_state.q1_no, 'shock_zone1_gf': 0.0, 'q_mult_zone1_gf': st.session_state.q1_no,
        'shock_zone2_nogf': st.session_state.s2_no, 'q_mult_zone2_nogf': st.session_state.q2_no, 'shock_zone2_gf': 0.0, 'q_mult_zone2_gf': st.session_state.q2_no,
        'shock_zone3_nogf': st.session_state.s3_no, 'q_mult_zone3_nogf': st.session_state.q3_no, 'shock_zone3_gf': 0.0, 'q_mult_zone3_gf': st.session_state.q3_no,
    }
    
    p_A = {'M': (M_in*0.9, M_in*1.1), 'p': (st.session_state.p_min_a, st.session_state.p_max_a), 'q': (st.session_state.q_min_a, st.session_state.q_max_a), 'C': (st.session_state.C_min_a, st.session_state.C_max_a), 'ARPU': (st.session_state.ARPU_min_a, st.session_state.ARPU_max_a), 'kappa': kappa_in, 'Delta_CM': dcm_in, 'Fixed_Cost': (st.session_state.Fixed_Cost_min_a, st.session_state.Fixed_Cost_max_a)}
    p_B = {'M': (M_in*0.9, M_in*1.1), 'p': (st.session_state.p_min_b, st.session_state.p_max_b), 'q': (st.session_state.q_min_b, st.session_state.q_max_b), 'C': (st.session_state.C_min_b, st.session_state.C_max_b), 'ARPU': (st.session_state.ARPU_min_b, st.session_state.ARPU_max_b), 'kappa': kappa_in, 'Delta_CM': dcm_in, 'Fixed_Cost': (st.session_state.Fixed_Cost_min_b, st.session_state.Fixed_Cost_max_b)}
    
    with st.spinner("Berechne statistische Stichproben..."):
        n_A = calculate_cochran_n(p_A, T_in); n_B = calculate_cochran_n(p_B, T_in)
        n_Sw = calculate_cochran_n(p_B, T_in, 'switch', p_A, st.session_state.trig_val_in, st.session_state.check_mode_in, st.session_state.check_year_in, st.session_state.metric_in, sw_conf)
        n_Ab = calculate_cochran_n(p_B, T_in, 'abandon', None, st.session_state.trig_val_in, st.session_state.check_mode_in, st.session_state.check_year_in, st.session_state.metric_in, sw_conf)
    
    res_store = {}; bar = st.progress(0)
    scenarios = [("Standard (A)", n_A, p_A, 'static', None, 'tab:blue'), ("Fighter (B)", n_B, p_B, 'static', None, 'tab:red'), ("Switch Option", n_Sw, p_B, 'switch', p_A, 'tab:green'), ("Abandon Option", n_Ab, p_B, 'abandon', None, 'black')]
    
    def rnd(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v
    
    for idx, (name, n, pr, mode, fb, col) in enumerate(scenarios):
        sim_sums, all_N, all_ARPU, sim_inputs = [], [], [], []
        for _ in range(n):
            curr = {k: rnd(v) for k, v in pr.items()}
            fbp = {k: rnd(v) for k, v in fb.items()} if fb else None
            Nt, At, tot, _ = run_simulation(**curr, start=1, T=T_in, mode=mode, trigger_val=st.session_state.trig_val_in, fallback_params=fbp, check_mode=st.session_state.check_mode_in, check_year=st.session_state.check_year_in, growth_metric=st.session_state.metric_in, switch_config=sw_conf)
            sim_sums.append(tot); all_N.append(Nt); all_ARPU.append(At); sim_inputs.append(curr)
        
        arr_N, arr_ARPU = np.array(all_N), np.array(all_ARPU)
        idx_base = np.argsort(sim_sums)[len(sim_sums)//2]
        idx_worst = np.argsort(sim_sums)[int(len(sim_sums)*0.05)]
        idx_best = np.argsort(sim_sums)[int(len(sim_sums)*0.95)]
        
        torn, base_v = get_tornado_data(None, pr, T_in, mode, st.session_state.trig_val_in, fb, st.session_state.check_mode_in, st.session_state.check_year_in, st.session_state.metric_in, sw_conf)
        df_in = pd.DataFrame(sim_inputs); df_reg = df_in.loc[:, df_in.std() > 0]
        reg, r2 = get_regression_sensitivity(df_reg, sim_sums) if not df_reg.empty else (None, 0)

        res_store[name] = {
            "n": n, "sums": sim_sums, "color": col, "mean": np.mean(sim_sums), "std": np.std(sim_sums), "r2": r2,
            "avg_N": np.mean(arr_N, axis=0), "p5_N": np.percentile(arr_N, 5, axis=0), "p95_N": np.percentile(arr_N, 95, axis=0),
            "tornado": (torn, base_v), "regression": reg,
            "fin_base": calculate_detailed_financials(arr_N[idx_base], arr_ARPU[idx_base], T_in),
            "fin_worst": calculate_detailed_financials(arr_N[idx_worst], arr_ARPU[idx_worst], T_in),
            "fin_best": calculate_detailed_financials(arr_N[idx_best], arr_ARPU[idx_best], T_in)
        }
        bar.progress((idx+1)/4)
    st.session_state.simulation_results = res_store
    st.success("Simulation & Accounting abgeschlossen.")

# --- TAB: SIM ERGEBNISSE ---
with tab_res:
    if st.session_state.simulation_results:
        res = st.session_state.simulation_results
        st.subheader("Simulations-Zusammenfassung")
        summ_data = [{"Szenario": k, "Runs": d['n'], "Mean": f"{d['mean']:,.0f} €", "Std": f"{d['std']:,.0f} €", "R2": f"{d['r2']:.2f}"} for k, d in res.items()]
        st.table(pd.DataFrame(summ_data))
        
        c1, c2 = st.columns(2)
        fig_n, ax_n = plt.subplots(); fig_w, ax_w = plt.subplots()
        for n, d in res.items():
            ax_n.plot(d["avg_N"], label=n, color=d["color"]); ax_n.fill_between(range(T_in), d["p5_N"], d["p95_N"], color=d["color"], alpha=0.1)
            ax_w.hist(d["sums"], bins=30, alpha=0.5, label=n, color=d["color"])
        ax_n.legend(); c1.pyplot(fig_n); ax_w.legend(); c2.pyplot(fig_w)

# --- TAB: FINANZBERICHTE ---
with tab_fin:
    if st.session_state.simulation_results:
        scen = st.selectbox("Strategie wählen", list(st.session_state.simulation_results.keys()))
        case = st.radio("Szenario-Fall", ["Base", "Worst", "Best"], horizontal=True)
        df_f = st.session_state.simulation_results[scen][f"fin_{case.lower()}"]
        
        st.subheader(f"Detaillierte Finanzplanung: {scen} ({case})")
        t_guv, t_cf, t_bil = st.tabs(["📑 GuV", "💰 Cashflow", "⚖️ Bilanz"])
        with t_guv: st.dataframe(df_f.set_index("Jahr")[["Umsatz", "Wareneinsatz (COGS)", "Gesamtkosten (OPEX)", "EBITDA", "Abschreibungen", "EBIT", "Steuern", "Jahresüberschuss"]].T.style.format("{:,.0f}"))
        with t_cf: st.dataframe(df_f.set_index("Jahr")[["Jahresüberschuss", "Abschreibungen", "Investitionen (Assets)", "Net Cash Change", "Kasse"]].T.style.format("{:,.0f}"))
        with t_bil: st.dataframe(df_f.set_index("Jahr")[["Anlagevermögen", "Kasse", "Summe Aktiva", "Eigenkapital", "Bankdarlehen", "Summe Passiva"]].T.style.format("{:,.0f}"))

        if st.button("📄 PDF REPORT GENERIEREN (Full Detail)"):
            pdf = PDFReport(orientation='L', unit='mm', format='A4')
            pdf.add_page(); pdf.section_title(f"Simulation & Finanzplan: {scen} ({case})")
            pdf.add_df_table(df_f[["Jahr", "Umsatz", "Wareneinsatz (COGS)", "EBITDA", "EBIT", "Jahresüberschuss"]])
            pdf.add_page(); pdf.section_title("Cashflow & Bilanz")
            pdf.add_df_table(df_f[["Jahr", "Jahresüberschuss", "Investitionen (Assets)", "Kasse", "Eigenkapital", "Bankdarlehen"]])
            
            # Tornado & Regression auf neue Seite
            pdf.add_page(); pdf.section_title("Sensitivitätsanalyse")
            st.info("PDF Download bereit.")
            st.download_button("Download PDF", pdf.output(dest='S').encode('latin-1'), f"Report_{scen}.pdf", "application/pdf")
