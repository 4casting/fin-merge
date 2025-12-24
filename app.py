import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import io
import math
import datetime
import tempfile

# ==========================================
# 0. KONFIGURATION & HELPER
# ==========================================
st.set_page_config(page_title="Integrated Real Option & Financial Model", layout="wide")

if 'history' not in st.session_state: st.session_state.history = []
if 'sim_results' not in st.session_state: st.session_state.sim_results = None

def safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value): 
            return default
        return float(value)
    except: 
        return default

# Initialisierung der detaillierten Kostendaten (aus Modell 2)
if "current_jobs_df" not in st.session_state:
    roles = [
        {"Job Titel": "CEO", "Jahresgehalt (€)": 120000, "FTE Jahr 1": 1.0, "Laptop": True, "Smartphone": True, "Auto": True, "Büro": True},
        {"Job Titel": "Sales", "Jahresgehalt (€)": 60000, "FTE Jahr 1": 2.0, "Laptop": True, "Smartphone": True, "Auto": True, "Büro": True},
        {"Job Titel": "Tech", "Jahresgehalt (€)": 65000, "FTE Jahr 1": 3.0, "Laptop": True, "Smartphone": False, "Auto": False, "Büro": True},
    ]
    st.session_state["current_jobs_df"] = pd.DataFrame(roles)

if "cost_centers_df" not in st.session_state:
    st.session_state["cost_centers_df"] = pd.DataFrame([
        {"Kostenstelle": "Server & IT", "Grundwert Jahr 1 (€)": 12000, "Umsatz-Kopplung (%)": 2.0},
        {"Kostenstelle": "Marketing (Fix)", "Grundwert Jahr 1 (€)": 50000, "Umsatz-Kopplung (%)": 0.0},
        {"Kostenstelle": "Logistik/Ops", "Grundwert Jahr 1 (€)": 5000, "Umsatz-Kopplung (%)": 5.0},
    ])

if "assets_params" not in st.session_state:
    st.session_state["assets_params"] = {
        "price_laptop": 1500, "ul_laptop": 3,
        "price_phone": 800, "ul_phone": 2,
        "price_car": 40000, "ul_car": 6,
        "price_desk": 1000, "ul_desk": 10,
        "capex_annual": 5000, "depreciation_misc": 5
    }

if "fin_params" not in st.session_state:
    st.session_state["fin_params"] = {
        "equity": 250000.0, "loan_initial": 0.0, "loan_rate": 5.0,
        "tax_rate": 25.0, "wage_inc": 2.5, "inflation": 2.0, "lnk_pct": 25.0,
        "cac": 150.0, "target_rev_per_fte": 150000.0, "min_cash": 20000.0
    }

# ==========================================
# 1. KERN-LOGIK: SIMULATION (MODEL 1)
# ==========================================
def run_simulation(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   mode='static', trigger_val=0.05, fallback_params=None,
                   check_mode='continuous', check_year=3, growth_metric='share_of_m',
                   switch_config=None):
    
    # Arrays für Zeitreihen
    N = np.zeros(T); W = np.zeros(T)
    ARPU_series = np.zeros(T) # Speichern des effektiv realisierten ARPU
    
    N[0] = start
    # Initial W (vereinfacht für Simulation, wird später detailliert berechnet)
    W[0] = N[0] * ARPU - Fixed_Cost 
    ARPU_series[0] = ARPU

    option_exercised = False
    project_is_dead = False
    
    curr_p, curr_q, curr_C = p, q, C
    curr_ARPU, curr_kappa, curr_Delta_CM = ARPU, kappa, Delta_CM
    curr_FC, curr_M = Fixed_Cost, M

    growth_history = []

    for t in range(1, T):
        if project_is_dead:
            N[t] = 0.0; W[t] = 0.0; ARPU_series[t] = 0.0
            continue 

        N_prev = N[t-1]
        
        # --- PROGNOSE ---
        potential_acquisition = (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev)
        if potential_acquisition < 0: potential_acquisition = 0
        
        if growth_metric == 'share_of_m':
            current_rate = potential_acquisition / curr_M
        else:
            current_rate = (potential_acquisition / N_prev) if N_prev > 0 else 0.0

        # --- TRIGGER PRÜFUNG ---
        if mode != 'static' and not option_exercised:
            is_check_time = False
            if check_mode == 'specific' and t == check_year: is_check_time = True
            elif check_mode == 'continuous' and t >= check_year: is_check_time = True
            
            if is_check_time:
                all_rates = growth_history + [current_rate]
                avg_growth = sum(all_rates) / len(all_rates) if all_rates else 0
                
                if avg_growth < trigger_val:
                    option_exercised = True
                    
                    if mode == 'switch' and fallback_params and switch_config:
                        # Logik für Switch (Parameter ändern)
                        target_ARPU = fallback_params['ARPU']
                        delta_p = (target_ARPU - curr_ARPU) / curr_ARPU if curr_ARPU > 0 else 0.0
                        
                        if delta_p <= switch_config['thresh_low']: zone_prefix = 'zone1'
                        elif delta_p <= switch_config['thresh_high']: zone_prefix = 'zone2'
                        else: zone_prefix = 'zone3'
                        
                        use_gf = switch_config['grandfathering']
                        suffix = "_gf" if use_gf else "_nogf"
                        
                        shock_factor = switch_config[f'shock_{zone_prefix}{suffix}']
                        q_multiplier = switch_config[f'q_mult_{zone_prefix}{suffix}']
                        
                        # Parameter Update
                        curr_p = fallback_params['p']
                        curr_C = fallback_params['C']
                        curr_ARPU = fallback_params['ARPU'] # Neuer Preis
                        curr_kappa = fallback_params['kappa']
                        curr_Delta_CM = fallback_params['Delta_CM']
                        curr_FC = fallback_params['Fixed_Cost']
                        
                        base_target_q = fallback_params['q']
                        curr_q = base_target_q * q_multiplier 
                        
                        # Churn Schock auf Bestand
                        N_prev = N_prev * (1.0 - shock_factor)
                        if N_prev < 0: N_prev = 0
                        
                        # Neue Akquise Basis berechnen nach Schock
                        potential_acquisition = (curr_p + curr_q * (N_prev / curr_M)) * (curr_M - N_prev)
                        if potential_acquisition < 0: potential_acquisition = 0
                        
                    elif mode == 'abandon':
                        project_is_dead = True
                        N[t] = 0.0; W[t] = 0.0; ARPU_series[t] = 0.0
                        continue

        # Tracking für Trigger Historie
        realized_rate = 0
        if growth_metric == 'share_of_m':
            realized_rate = potential_acquisition / curr_M
        else:
            realized_rate = (potential_acquisition / N_prev) if N_prev > 0 else 0.0
        growth_history.append(realized_rate)

        # Kundenbestand Update
        retention = N_prev * (1 - curr_C)
        N[t] = retention + potential_acquisition
        if N[t] > curr_M: N[t] = curr_M
        
        # Finanzwert (Vereinfacht für MC, dient nur als Proxy für Value)
        revenue = N[t] * curr_ARPU
        cannib = potential_acquisition * curr_kappa * curr_Delta_CM
        W[t] = revenue - cannib - curr_FC
        
        ARPU_series[t] = curr_ARPU
        
    return N, W, sum(W), option_exercised, ARPU_series

# ==========================================
# 2. KERN-LOGIK: FINANZ-DETAILLIERUNG (MODEL 2)
# ==========================================
def calculate_detailed_financials(years_T, N_curve, ARPU_curve, jobs_df, cc_df, asset_conf, fin_conf):
    """
    Nimmt die simulierten Kurven (N, ARPU) und berechnet den vollen Finanzplan.
    """
    results = []
    
    # State Variablen
    debt = fin_conf["loan_initial"]
    cash = fin_conf["equity"]
    retained = 0.0
    fixed_assets = 0.0
    loss_carry = 0.0
    
    # Assets Register
    asset_reg = {"Laptop":[], "Smartphone":[], "Auto":[], "Büro":[], "Misc":[]}
    mapping = {
        "Laptop": ("price_laptop", "ul_laptop"), "Smartphone": ("price_phone", "ul_phone"),
        "Auto": ("price_car", "ul_car"), "Büro": ("price_desk", "ul_desk")
    }
    
    prev_cc = {}
    jobs = jobs_df.to_dict('records')
    ccs = cc_df.to_dict('records')
    
    # Loop über Jahre (Simulation Start ist t=0, Finanzen meist t=1..T)
    # N_curve hat Länge T (z.B. 30). Wir nehmen die Jahre 1 bis T.
    
    # Initial Setup t=0 (nicht im Result, nur Startwerte)
    
    for t_idx in range(1, len(N_curve)):
        t = t_idx # Jahr 1, 2, 3...
        row = {"Jahr": t}
        
        # 1. Umsatz & Kunden (aus Simulation übernommen)
        n_t = N_curve[t_idx]
        arpu_t = ARPU_curve[t_idx]
        
        if n_t <= 0:
            # Abandon Fall oder Simulation Ende
            results.append({k: 0 for k in ["Umsatz", "EBITDA", "Jahresüberschuss", "Kasse", "FTE Total"]})
            results[-1]["Jahr"] = t
            continue

        rev = n_t * arpu_t
        row["Kunden"] = n_t
        row["Umsatz"] = rev
        
        prev_rev = results[-1]["Umsatz"] if len(results) > 0 else rev
        growth = (rev - prev_rev)/prev_rev if len(results) > 0 and prev_rev > 0 else 0.0

        # 2. Kosten (COGS Annahme: Pauschal 20% oder aus Delta_CM ableitbar, hier vereinfacht)
        # In Modell 1 war Delta_CM der Margenverlust. Wir nehmen an COGS ~ 15% vom Umsatz
        cogs = rev * 0.15 
        row["Wareneinsatz (COGS)"] = cogs
        
        # 3. Personal
        wage_idx = (1 + fin_conf["wage_inc"]/100)**(t-1)
        pers_cost = 0.0
        hw_needs = {k: 0.0 for k in mapping}
        
        base_ftes = sum(safe_float(j.get("FTE Jahr 1")) for j in jobs)
        target_fte = rev / fin_conf["target_rev_per_fte"] if fin_conf["target_rev_per_fte"] > 0 else 0
        
        curr_total_fte = 0
        for j in jobs:
            base = safe_float(j.get("FTE Jahr 1"))
            sal = safe_float(j.get("Jahresgehalt (€)"))
            # Skalierung FTE mit Umsatzwachstum (Modell 2 Logik)
            fte = max(base, target_fte * (base/base_ftes)) if base_ftes > 0 else 0
            curr_total_fte += fte
            pers_cost += sal * fte * wage_idx * (1 + fin_conf["lnk_pct"]/100)
            
            if j.get("Laptop"): hw_needs["Laptop"] += fte
            if j.get("Smartphone"): hw_needs["Smartphone"] += fte
            if j.get("Auto"): hw_needs["Auto"] += fte
            if j.get("Büro"): hw_needs["Büro"] += fte
            
        row["Personalkosten"] = pers_cost
        row["FTE Total"] = curr_total_fte

        # 4. Kostenstellen & OpEx
        cc_sum = 0.0
        for c in ccs:
            nm = c.get("Kostenstelle"); base = safe_float(c.get("Grundwert Jahr 1 (€)")); coup = safe_float(c.get("Umsatz-Kopplung (%)"))/100
            last = prev_cc.get(nm, base)
            curr = base if t==1 else last * (1 + growth*coup)
            # Inflation drauf
            curr = curr * (1 + fin_conf["inflation"]/100)
            prev_cc[nm] = curr; cc_sum += curr
        
        # Akquisekosten (simuliert)
        n_prev = N_curve[t_idx-1]
        new_cust = max(0, n_t - (n_prev * 0.95)) # Simple approx for new customers
        marketing_cac = new_cust * fin_conf["cac"]
        
        opex = pers_cost + cc_sum + marketing_cac
        row["Gesamtkosten (OPEX)"] = opex
        
        # 5. Ergebnisse
        ebitda = rev - cogs - opex
        row["EBITDA"] = ebitda
        
        # Assets CAPEX & AFA
        capex = 0.0; afa = 0.0
        # Misc Capex
        misc_p = asset_conf["capex_annual"]; misc_ul = asset_conf["depreciation_misc"]
        asset_reg["Misc"].append({"y":t, "v":misc_p, "ul":misc_ul})
        capex += misc_p
        
        for k, (pk_key, ul_key) in mapping.items():
            needed = hw_needs[k]
            price = asset_conf[pk_key]; ul = asset_conf[ul_key]
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
        row["EBIT"] = ebitda - afa
        
        # Zinsen & Steuern
        intr = debt * (fin_conf["loan_rate"]/100)
        ebt = row["EBIT"] - intr
        
        tax = 0.0
        if ebt < 0: loss_carry += abs(ebt)
        else:
            use = min(ebt, loss_carry); loss_carry -= use
            tax = (ebt - use) * (fin_conf["tax_rate"]/100)
            
        row["Steuern"] = tax
        net = ebt - tax
        row["Jahresüberschuss"] = net
        
        # Cashflow
        cf_op = net + afa
        cf_inv = -capex
        
        cash_start = results[-1]["Kasse"] if len(results) > 0 else fin_conf["equity"]
        pre_fin = cash_start + cf_op + cf_inv
        
        min_c = fin_conf["min_cash"]
        borrow = 0.0; repay = 0.0
        if pre_fin < min_c: borrow = min_c - pre_fin
        elif pre_fin > min_c and debt > 0: repay = min(debt, pre_fin - min_c)
        
        cash_end = pre_fin + borrow - repay
        debt_end = debt + borrow - repay
        
        row["Kasse"] = cash_end
        row["Bankdarlehen"] = debt_end
        row["Eigenkapital"] = fin_conf["equity"] + retained + net
        
        cash = cash_end; debt = debt_end
        fixed_assets = max(0, fixed_assets + capex - afa)
        retained += net
        
        results.append(row)
        
    return pd.DataFrame(results)

# ==========================================
# 3. PDF REPORT GENERATOR KLASSE
# ==========================================
class DetailedReport(FPDF):
    def __init__(self, scenario_name, case_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scenario_name = scenario_name
        self.case_type = case_type
    
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, f'Simulation Report: {self.scenario_name}', 0, 1, 'L')
        self.set_font('Arial', 'I', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f'Case: {self.case_type} | Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'L')
        self.line(10, 28, 200, 28)
        self.ln(10)
        
    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 6, label, 0, 1, 'L', 1)
        self.ln(4)

    def add_plot(self, fig):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, dpi=150, bbox_inches='tight')
            self.image(tmpfile.name, w=180)
        self.ln(5)
        
    def add_table(self, df):
        self.set_font('Arial', 'B', 7)
        cols = list(df.columns)
        w = 190 / len(cols)
        for col in cols:
            self.cell(w, 5, str(col), 1, 0, 'C')
        self.ln()
        self.set_font('Arial', '', 7)
        for _, row in df.iterrows():
            for col in cols:
                val = row[col]
                txt = f"{val:,.0f}" if isinstance(val, (int, float)) else str(val)
                self.cell(w, 5, txt, 1, 0, 'R')
            self.ln()
        self.ln(5)

def generate_pdf_bytes(df_fin, scenario, case, N_series):
    pdf = DetailedReport(scenario, case, orientation='P', unit='mm', format='A4')
    pdf.alias_nb_pages()
    
    # Seite 1: Übersicht & Graphen
    pdf.add_page()
    pdf.chapter_title("Management Summary")
    
    # Key Metrics
    last = df_fin.iloc[-1] if not df_fin.empty else None
    if last is not None:
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 5, f"Jahr 10 Umsatz: {last['Umsatz']:,.0f} EUR", 0, 1)
        pdf.cell(0, 5, f"Jahr 10 EBITDA: {last['EBITDA']:,.0f} EUR", 0, 1)
        pdf.cell(0, 5, f"Kumulierter Cash: {last['Kasse']:,.0f} EUR", 0, 1)
        pdf.ln(5)

    # Graphen erstellen
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    ax1.plot(df_fin["Jahr"], df_fin["Umsatz"], label="Umsatz", color="blue")
    ax1.plot(df_fin["Jahr"], df_fin["Gesamtkosten (OPEX)"], label="Opex", color="red", linestyle="--")
    ax1.set_title("Umsatz & Kosten Entwicklung")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(df_fin["Jahr"], df_fin["EBITDA"], color="green", alpha=0.6)
    ax2.set_title("EBITDA")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.add_plot(fig)
    plt.close(fig)
    
    # Seite 2: P&L Table
    pdf.add_page()
    pdf.chapter_title("Gewinn- und Verlustrechnung (GuV)")
    cols_guv = ["Jahr", "Umsatz", "Wareneinsatz (COGS)", "Gesamtkosten (OPEX)", "EBITDA", "EBIT", "Jahresüberschuss"]
    # Check if cols exist
    use_cols = [c for c in cols_guv if c in df_fin.columns]
    pdf.add_table(df_fin[use_cols].head(15)) # max 15 Jahre auf eine Seite
    
    # Seite 3: Cashflow & Bilanz
    pdf.add_page()
    pdf.chapter_title("Cashflow & Bilanz Indikatoren")
    cols_cf = ["Jahr", "Investitionen (Assets)", "Kreditaufnahme", "Tilgung", "Kasse", "Bankdarlehen", "Eigenkapital"]
    use_cols_cf = [c for c in cols_cf if c in df_fin.columns]
    pdf.add_table(df_fin[use_cols_cf].head(15))
    
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ==========================================
# 4. GUI IMPLEMENTIERUNG
# ==========================================
st.title("Integrated Valuation Model (Bass + Real Options + Financials)")
st.markdown("Dieses Tool kombiniert eine stochastische Marktsimulation mit einem detaillierten Finanzplan.")

tab_sim, tab_cost = st.tabs(["📊 Simulation & Strategie", "🏢 Kostenstruktur & Finanzplan"])

# --- TAB 2: KOSTENSTRUKTUR (MODEL 2 INPUTS) ---
with tab_cost:
    st.markdown("### Detaillierte Kostenannahmen")
    st.info("Diese Annahmen werden genutzt, um aus den Simulationsdaten (Kundenanzahl) einen detaillierten Finanzplan zu erstellen.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Finanzierung & Makro")
        fp = st.session_state["fin_params"]
        fp["equity"] = st.number_input("Eigenkapital (€)", value=fp["equity"], step=5000.0)
        fp["loan_rate"] = st.number_input("Kreditzins %", value=fp["loan_rate"], step=0.1)
        fp["tax_rate"] = st.number_input("Steuersatz %", value=fp["tax_rate"], step=1.0)
        fp["wage_inc"] = st.number_input("Lohnsteigerung p.a. %", value=fp["wage_inc"], step=0.1)
        fp["cac"] = st.number_input("Marketing CAC (€)", value=fp["cac"], step=10.0)
        
    with c2:
        st.subheader("Asset Preise")
        ap = st.session_state["assets_params"]
        c2a, c2b = st.columns(2)
        with c2a:
            ap["price_laptop"] = st.number_input("Preis Laptop", value=ap["price_laptop"])
            ap["price_car"] = st.number_input("Preis PKW", value=ap["price_car"])
        with c2b:
            ap["capex_annual"] = st.number_input("Sonst. Capex p.a.", value=ap["capex_annual"])
            
    st.markdown("---")
    st.subheader("Personalplan (Initial)")
    st.session_state["current_jobs_df"] = st.data_editor(st.session_state["current_jobs_df"], num_rows="dynamic", use_container_width=True)
    
    st.subheader("Kostenstellen (Fix & Variabel)")
    st.session_state["cost_centers_df"] = st.data_editor(st.session_state["cost_centers_df"], num_rows="dynamic", use_container_width=True)

    # Berechne Basis-Fixkosten für die Simulation (Proxy)
    # Summe Gehälter + Fixe Kostenstellen + Capex Abschreibung approx
    jobs_sum = st.session_state["current_jobs_df"]["Jahresgehalt (€)"].sum() * 1.25 # inkl NK
    cc_sum = st.session_state["cost_centers_df"]["Grundwert Jahr 1 (€)"].sum()
    estimated_fixed_costs = jobs_sum + cc_sum + ap["capex_annual"]

# --- TAB 1: SIMULATION ---
with tab_sim:
    # Sidebar History
    with st.sidebar:
        st.header("Simulation Settings")
        T_in = st.slider("Laufzeit (Jahre)", 5, 20, 10)
        M_in = st.number_input("Marktpotenzial (M)", value=50000, step=1000)
        
        st.markdown("---")
        st.markdown("**Trigger Settings**")
        trig_val_in = st.slider("Wachstumsgrenze (<)", 0.0, 0.2, 0.05, 0.01)
        check_year_in = st.number_input("Prüfung ab Jahr", 1, T_in, 3)
        
        st.markdown("---")
        st.caption(f"Geschätzte Fixkosten (aus Tab 2): {estimated_fixed_costs:,.0f} €")
        
    # Input Ranges für Simulation
    col_left, col_right = st.columns(2)
    
    def range_in(label, def_min, def_max, key_suffix, step=0.01):
        c1, c2 = st.columns(2)
        v_min = c1.number_input(f"{label} Min", value=float(def_min), step=step, key=f"{label}_min_{key_suffix}")
        v_max = c2.number_input(f"{label} Max", value=float(def_max), step=step, key=f"{label}_max_{key_suffix}")
        return (v_min, v_max)

    with col_left:
        st.markdown("### Option A: Standard (Safe)")
        p_a = range_in("p", 0.005, 0.010, "a", 0.001)
        q_a = range_in("q", 0.15, 0.25, "a", 0.01)
        c_a = range_in("Churn", 0.03, 0.05, "a", 0.01)
        arpu_a = range_in("ARPU", 1800, 2200, "a", 50.0)
        # Fixkosten nehmen wir global aus Tab 2, aber Kappa/Delta_CM sind strategiesspezifisch
        kap_a = range_in("Kappa", 0.05, 0.10, "a", 0.01)
        dcm_a = range_in("Delta Margin", 50, 100, "a", 10.0)

    with col_right:
        st.markdown("### Option B: Fighter (Aggressiv)")
        p_b = range_in("p", 0.030, 0.050, "b", 0.001)
        q_b = range_in("q", 0.25, 0.40, "b", 0.01)
        c_b = range_in("Churn", 0.08, 0.12, "b", 0.01)
        arpu_b = range_in("ARPU", 1200, 1500, "b", 50.0)
        kap_b = range_in("Kappa", 0.15, 0.25, "b", 0.01)
        dcm_b = range_in("Delta Margin", 50, 100, "b", 10.0)

    with st.expander("⚙️ Switch Konfiguration"):
        # Switch Matrix Inputs (Hardcoded defaults for brevity, can be expanded)
        switch_config_dict = {
            'grandfathering': False, 'thresh_low': 0.10, 'thresh_high': 0.20,
            'shock_zone1_nogf': 0.05, 'q_mult_zone1_nogf': 0.9,
            'shock_zone2_nogf': 0.15, 'q_mult_zone2_nogf': 0.7,
            'shock_zone3_nogf': 0.30, 'q_mult_zone3_nogf': 0.5,
            # Placeholder for GF
            'shock_zone1_gf': 0.0, 'q_mult_zone1_gf': 0.95,
            'shock_zone2_gf': 0.0, 'q_mult_zone2_gf': 0.8,
            'shock_zone3_gf': 0.0, 'q_mult_zone3_gf': 0.6,
        }
        st.write("Verwendet Standard-Matrix Parameter für Preisschocks.")

    if st.button("🚀 Simulation Starten", type="primary", use_container_width=True):
        
        # 1. Parameter Zusammenbau
        params_A = {'M': (M_in, M_in), 'p': p_a, 'q': q_a, 'C': c_a, 'ARPU': arpu_a, 'kappa': kap_a, 'Delta_CM': dcm_a, 'Fixed_Cost': (estimated_fixed_costs, estimated_fixed_costs)}
        params_B = {'M': (M_in, M_in), 'p': p_b, 'q': q_b, 'C': c_b, 'ARPU': arpu_b, 'kappa': kap_b, 'Delta_CM': dcm_b, 'Fixed_Cost': (estimated_fixed_costs, estimated_fixed_costs)}
        
        scenarios = [
            ("Standard (A)", params_A, 'static', None),
            ("Fighter (B)", params_B, 'static', None),
            ("Switch Option", params_B, 'switch', params_A),
            ("Abandon Option", params_B, 'abandon', None)
        ]
        
        sim_store = {}
        iterations = 500 # Für Performance im Demo-Mode
        
        progress_bar = st.progress(0)
        
        def get_rnd(v): return np.random.triangular(v[0], (v[0]+v[1])/2, v[1]) if isinstance(v, tuple) else v

        # 2. Monte Carlo Loop
        for idx, (name, p_rng, mode, fb_rng) in enumerate(scenarios):
            
            runs_N = []
            runs_W = []
            runs_ARPU = []
            final_sums = []
            
            for _ in range(iterations):
                curr = {k: get_rnd(v) for k, v in p_rng.items()}
                fb = {k: get_rnd(v) for k, v in fb_rng.items()} if fb_rng else None
                
                N_t, W_t, sum_w, _, arpu_t = run_simulation(
                    **curr, start=10, T=T_in, mode=mode, trigger_val=trig_val_in,
                    fallback_params=fb, check_year=check_year_in, switch_config=switch_config_dict
                )
                
                runs_N.append(N_t)
                runs_W.append(W_t)
                runs_ARPU.append(arpu_t)
                final_sums.append(sum_w)
            
            # 3. Statistiken extrahieren
            arr_N = np.array(runs_N)
            arr_ARPU = np.array(runs_ARPU)
            
            # Wir identifizieren Worst (P5), Base (Mean-nähest), Best (P95) basierend auf dem Total Value
            sums = np.array(final_sums)
            idx_worst = np.argsort(sums)[int(iterations * 0.05)]
            idx_best = np.argsort(sums)[int(iterations * 0.95)]
            # Für Base nehmen wir den Median Index
            idx_base = np.argsort(sums)[int(iterations * 0.50)]
            
            sim_store[name] = {
                "mean_val": np.mean(sums),
                "std_val": np.std(sums),
                "N_worst": arr_N[idx_worst], "ARPU_worst": arr_ARPU[idx_worst],
                "N_base": arr_N[idx_base], "ARPU_base": arr_ARPU[idx_base],
                "N_best": arr_N[idx_best], "ARPU_best": arr_ARPU[idx_best],
                "all_N": arr_N
            }
            progress_bar.progress((idx + 1) / 4)
            
        st.session_state.sim_results = sim_store
        st.success("Simulation abgeschlossen!")

    # --- ERGEBNIS ANZEIGE ---
    if st.session_state.sim_results:
        res = st.session_state.sim_results
        
        st.markdown("### Simulations-Ergebnisse (High Level)")
        
        # Übersichtstabelle
        summ = []
        for k, v in res.items():
            summ.append({"Szenario": k, "Ø Net Value (Proxy)": f"{v['mean_val']:,.0f}", "Risk (Std)": f"{v['std_val']:,.0f}"})
        st.dataframe(pd.DataFrame(summ))
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        for k, v in res.items():
            # Zeige Base Case Kurve
            ax.plot(v["N_base"], label=k)
            # Unsicherheitsbereich (statistisch berechnet aus allen Runs)
            p5 = np.percentile(v["all_N"], 5, axis=0)
            p95 = np.percentile(v["all_N"], 95, axis=0)
            ax.fill_between(range(len(p5)), p5, p95, alpha=0.1)
        ax.set_title("Kundenentwicklung (Base Case + Uncertainty)")
        ax.legend()
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### 📥 Detaillierte Finanzberichte generieren")
        st.info("Wähle ein Szenario und lade den PDF Report herunter. Der Report nutzt die Kostendaten aus dem 'Kostenstruktur'-Tab.")
        
        cols = st.columns(4)
        for i, (name, data) in enumerate(res.items()):
            with cols[i]:
                st.markdown(f"**{name}**")
                
                # WORST CASE BUTTON
                if st.button(f"Worst Case PDF", key=f"btn_w_{i}"):
                    df_fin = calculate_detailed_financials(T_in, data["N_worst"], data["ARPU_worst"], 
                                                         st.session_state["current_jobs_df"], 
                                                         st.session_state["cost_centers_df"], 
                                                         st.session_state["assets_params"], 
                                                         st.session_state["fin_params"])
                    pdf_data = generate_pdf_bytes(df_fin, name, "Worst Case (P5)", data["N_worst"])
                    st.download_button("⬇️ Download", pdf_data, f"{name}_Worst.pdf", "application/pdf", key=f"dl_w_{i}")

                # BASE CASE BUTTON
                if st.button(f"Base Case PDF", key=f"btn_b_{i}"):
                    df_fin = calculate_detailed_financials(T_in, data["N_base"], data["ARPU_base"], 
                                                         st.session_state["current_jobs_df"], 
                                                         st.session_state["cost_centers_df"], 
                                                         st.session_state["assets_params"], 
                                                         st.session_state["fin_params"])
                    pdf_data = generate_pdf_bytes(df_fin, name, "Base Case (Median)", data["N_base"])
                    st.download_button("⬇️ Download", pdf_data, f"{name}_Base.pdf", "application/pdf", key=f"dl_b_{i}")

                # BEST CASE BUTTON
                if st.button(f"Best Case PDF", key=f"btn_best_{i}"):
                    df_fin = calculate_detailed_financials(T_in, data["N_best"], data["ARPU_best"], 
                                                         st.session_state["current_jobs_df"], 
                                                         st.session_state["cost_centers_df"], 
                                                         st.session_state["assets_params"], 
                                                         st.session_state["fin_params"])
                    pdf_data = generate_pdf_bytes(df_fin, name, "Best Case (P95)", data["N_best"])
                    st.download_button("⬇️ Download", pdf_data, f"{name}_Best.pdf", "application/pdf", key=f"dl_best_{i}")