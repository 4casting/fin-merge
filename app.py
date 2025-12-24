import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import datetime

# ==========================================
# 0. KONFIGURATION & INIT
# ==========================================
st.set_page_config(page_title="Strategic Valuation Master", layout="wide")

# Session State Initialisierung für Persistenz
if 'sim_results' not in st.session_state: st.session_state.sim_results = None

# Standardwerte für Tabellen
if "staff_df" not in st.session_state:
    st.session_state["staff_df"] = pd.DataFrame([
        {"Rolle": "Management", "Gehalt (€)": 120000, "FTE Start": 1.0, "FTE Zuwachs pro 1M Umsatz": 0.0, "Laptop": True, "Auto": True},
        {"Rolle": "Sales", "Gehalt (€)": 60000, "FTE Start": 2.0, "FTE Zuwachs pro 1M Umsatz": 1.0, "Laptop": True, "Auto": True},
        {"Rolle": "Tech & Dev", "Gehalt (€)": 70000, "FTE Start": 3.0, "FTE Zuwachs pro 1M Umsatz": 0.5, "Laptop": True, "Auto": False},
        {"Rolle": "Support", "Gehalt (€)": 45000, "FTE Start": 1.0, "FTE Zuwachs pro 1M Umsatz": 2.0, "Laptop": False, "Auto": False},
    ])

if "cost_df" not in st.session_state:
    st.session_state["cost_df"] = pd.DataFrame([
        {"Kostenstelle": "Miete & Office", "Fixkosten (€ p.a.)": 30000, "Variable Kosten (% v. Umsatz)": 0.0},
        {"Kostenstelle": "Server & Cloud", "Fixkosten (€ p.a.)": 5000, "Variable Kosten (% v. Umsatz)": 3.0},
        {"Kostenstelle": "Marketing (Media)", "Fixkosten (€ p.a.)": 0, "Variable Kosten (% v. Umsatz)": 10.0},
        {"Kostenstelle": "Recht & Beratung", "Fixkosten (€ p.a.)": 15000, "Variable Kosten (% v. Umsatz)": 0.0},
    ])

# ==========================================
# 1. CORE LOGIC: SIMULATION (BASS + OPTIONS)
# ==========================================
def run_simulation_logic(T, start_n, M_range, p_range, q_range, arpu_range, 
                         churn_range, base_fix_costs,
                         mode='static', trigger_conf=None, switch_conf=None):
    
    # 1.1 Parameter für diesen Run ziehen (Monte Carlo)
    def rnd(r): return np.random.uniform(r[0], r[1])
    
    M = rnd(M_range)
    p = rnd(p_range)
    q = rnd(q_range)
    arpu = rnd(arpu_range)
    churn = rnd(churn_range)
    
    # Arrays initialisieren
    N = np.zeros(T)
    ARPU_trace = np.zeros(T)
    Marketing_Factor = np.zeros(T) # Tracking für Var Kosten
    
    N[0] = start_n
    ARPU_trace[0] = arpu
    Marketing_Factor[0] = 1.0
    
    # State
    curr_p, curr_q, curr_arpu, curr_churn = p, q, arpu, churn
    curr_fix = base_fix_costs
    is_switched = False
    is_dead = False
    
    # Historie für Trigger
    growth_log = []
    
    for t in range(1, T):
        if is_dead:
            N[t] = 0; ARPU_trace[t] = 0
            continue
            
        N_prev = N[t-1]
        
        # --- BASS DIFFUSION ---
        # Innovation + Imitation
        potential = (curr_p + curr_q * (N_prev / M)) * (M - N_prev)
        if potential < 0: potential = 0
        
        # Growth Rate berechnen (für Trigger)
        growth_rate = potential / N_prev if N_prev > 10 else 0
        growth_log.append(growth_rate)
        
        # --- OPTION TRIGGER CHECK (Start ab Jahr 3) ---
        if mode != 'static' and not is_switched and t >= 3:
            # Gleitender Durchschnitt der letzten 2 Jahre
            recent_growth = np.mean(growth_log[-2:])
            
            if recent_growth < trigger_conf['threshold']:
                
                if mode == 'abandon':
                    is_dead = True
                    N[t]=0; ARPU_trace[t]=0
                    continue
                    
                elif mode == 'switch':
                    is_switched = True
                    # Logik: Wir wechseln auf "Standard" Strategie Werte
                    # 1. Preis hoch (Fallback)
                    target_arpu = switch_conf['fallback_arpu']
                    price_delta = (target_arpu - curr_arpu) / curr_arpu
                    
                    # 2. Churn Schock (Kunden hauen ab wegen Preis)
                    shock = min(0.4, price_delta * switch_conf['elasticity'])
                    if not switch_conf['grandfathering']:
                        N_prev = N_prev * (1.0 - shock)
                    
                    # 3. Parameter Reset
                    curr_arpu = target_arpu
                    curr_p = switch_conf['fallback_p']
                    # Penalty auf q wegen Reputationsschaden
                    curr_q = switch_conf['fallback_q'] * 0.8 
                    
                    # Neuberechnung Potential mit neuen Werten
                    potential = (curr_p + curr_q * (N_prev / M)) * (M - N_prev)
                    if potential < 0: potential = 0

        # --- UPDATE ---
        churned_users = N_prev * curr_churn
        N[t] = N_prev + potential - churned_users
        if N[t] > M: N[t] = M
        
        ARPU_trace[t] = curr_arpu
        
    # Proxy Valuation für Sortierung (Simpler NPV)
    # Wir nutzen hier vereinfachte Margen, detailliert kommt später
    rev_stream = N * ARPU_trace
    proxy_margin = rev_stream * 0.3 - curr_fix
    npv_proxy = np.sum(proxy_margin) # Keine Diskontierung für Sortierung nötig
    
    return N, ARPU_trace, npv_proxy

# ==========================================
# 2. CORE LOGIC: FINANCIAL DETAILED (Modell 2)
# ==========================================
def calculate_financials(N_curve, ARPU_curve, staff_df, cost_df, asset_conf, fin_conf):
    """
    Berechnet GuV/Bilanz basierend auf einer Simualtionskurve (N, Price)
    """
    T = len(N_curve)
    rows = []
    
    # Init Bilanz
    cash = fin_conf['equity']
    loan = fin_conf['loan']
    retained = 0.0
    loss_carry = 0.0
    
    # Assets Register
    assets = [] # [{'val': 1000, 'ul': 3, 'age': 0}]
    
    # Helper Data
    staff = staff_df.to_dict('records')
    ccs = cost_df.to_dict('records')
    
    for t in range(1, T): # Start bei 1, da 0 Startwert ist
        r = {"Jahr": t}
        
        # 1. Umsatz (aus Simulation)
        n = N_curve[t]
        arpu = ARPU_curve[t]
        
        if n <= 1: # Projekt tot
            rows.append({k:0 for k in ["Umsatz", "EBITDA", "Jahresüberschuss", "Kasse", "FTE"]})
            rows[-1]["Jahr"] = t
            continue
            
        umsatz = n * arpu
        r["Kunden"] = n
        r["ARPU"] = arpu
        r["Umsatz"] = umsatz
        
        # 2. OpEx (Detailliert)
        # 2a. Personal (Variable FTEs)
        total_fte = 0
        total_personnel = 0
        needed_laptops = 0
        needed_cars = 0
        
        wage_idx = (1 + fin_conf['wage_inc']/100)**(t-1)
        lnk = 1 + fin_conf['lnk']/100
        
        for job in staff:
            # Logik: Start FTE + (Umsatz in Mio * Faktor)
            variable_part = (umsatz / 1_000_000) * job['FTE Zuwachs pro 1M Umsatz']
            fte = job['FTE Start'] + variable_part
            cost = fte * job['Gehalt (€)'] * wage_idx * lnk
            
            total_fte += fte
            total_personnel += cost
            
            if job['Laptop']: needed_laptops += fte
            if job['Auto']: needed_cars += fte
            
        r["Personal"] = total_personnel
        r["FTE"] = total_fte
        
        # 2b. Kostenstellen
        fix_opex = 0
        var_opex = 0
        inflation = (1 + fin_conf['inflation']/100)**(t-1)
        
        for cc in ccs:
            fix_opex += cc['Fixkosten (€ p.a.)'] * inflation
            var_opex += (cc['Variable Kosten (% v. Umsatz)'] / 100) * umsatz
            
        r["OpEx_Sach"] = fix_opex + var_opex
        
        # EBITDA
        ebitda = umsatz - total_personnel - r["OpEx_Sach"]
        r["EBITDA"] = ebitda
        
        # 3. Capex & AfA
        # Invest: Wir kaufen Laptops/Autos für neue FTEs + Ersatz
        # Vereinfachung: Bestand in Registry prüfen
        current_laptops = sum(1 for a in assets if a['type'] == 'laptop' and a['age'] < a['ul'])
        current_cars = sum(1 for a in assets if a['type'] == 'car' and a['age'] < a['ul'])
        
        capex = asset_conf['misc_capex']
        
        # Kauf Laptops
        if needed_laptops > current_laptops:
            buy_n = needed_laptops - current_laptops
            cost = buy_n * asset_conf['p_laptop']
            capex += cost
            for _ in range(int(buy_n)): assets.append({'type': 'laptop', 'val': asset_conf['p_laptop'], 'ul': asset_conf['ul_laptop'], 'age': 0})
            
        # Kauf Autos
        if needed_cars > current_cars:
            buy_n = needed_cars - current_cars
            cost = buy_n * asset_conf['p_car']
            capex += cost
            for _ in range(int(buy_n)): assets.append({'type': 'car', 'val': asset_conf['p_car'], 'ul': asset_conf['ul_car'], 'age': 0})
            
        r["Capex"] = capex
        
        # Abschreibung
        afa = 0
        for a in assets:
            if a['age'] < a['ul']:
                afa += a['val'] / a['ul']
                a['age'] += 1
        r["AfA"] = afa
        
        # 4. Net Income
        ebit = ebitda - afa
        zinsen = loan * (fin_conf['interest']/100)
        ebt = ebit - zinsen
        
        tax = 0
        if ebt < 0:
            loss_carry += abs(ebt)
        else:
            use = min(ebt, loss_carry)
            loss_carry -= use
            tax = (ebt - use) * (fin_conf['tax']/100)
            
        net = ebt - tax
        r["Jahresüberschuss"] = net
        
        # 5. Cashflow & Bilanz
        cf = net + afa - capex
        
        # Einfache Cash Logik
        pre_cash = cash + cf
        borrow = 0; repay = 0
        
        if pre_cash < fin_conf['min_cash']:
            borrow = fin_conf['min_cash'] - pre_cash
        elif pre_cash > fin_conf['min_cash'] and loan > 0:
            repay = min(loan, pre_cash - fin_conf['min_cash'])
            
        cash = pre_cash + borrow - repay
        loan = loan + borrow - repay
        retained += net
        
        r["Kasse"] = cash
        r["Kredit"] = loan
        r["Eigenkapital"] = fin_conf['equity'] + retained
        
        rows.append(r)
        
    return pd.DataFrame(rows)

# ==========================================
# 3. PDF ENGINE
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(50, 50, 50)
        self.cell(0, 10, 'Financial & Strategy Report', 0, 1, 'L')
        self.line(10, 20, 200, 20)
        self.ln(10)

def create_pdf(df, scenario_name, case_type, fig):
    pdf = PDFReport()
    pdf.add_page()
    
    # Header Info
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Strategie: {scenario_name} | Szenario: {case_type}", 0, 1)
    
    last = df.iloc[-1]
    
    # KPI Box
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Umsatz (Endjahr): {last['Umsatz']:,.0f} EUR", 0, 1, 'L', 1)
    pdf.cell(0, 8, f"EBITDA (Endjahr): {last['EBITDA']:,.0f} EUR", 0, 1, 'L', 1)
    pdf.cell(0, 8, f"Cash Bestand: {last['Kasse']:,.0f} EUR", 0, 1, 'L', 1)
    pdf.ln(5)
    
    # Plot
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig.savefig(tmpfile.name, dpi=100, bbox_inches='tight')
        pdf.image(tmpfile.name, x=10, w=180)
    pdf.ln(10)
    
    # Table
    pdf.set_font("Arial", "B", 8)
    cols = ["Jahr", "Umsatz", "Personal", "EBITDA", "Jahresüberschuss", "Kasse"]
    for c in cols: pdf.cell(30, 6, c, 1)
    pdf.ln()
    pdf.set_font("Arial", "", 8)
    for _, row in df.iterrows():
        for c in cols:
            pdf.cell(30, 6, f"{row[c]:,.0f}", 1)
        pdf.ln()
        
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 4. UI LAYOUT (UNIFIED)
# ==========================================
st.title("💠 Integrated Business Planner")
st.markdown("Ein Interface für alle Eingaben. Simulation und Finanzplan sind voll integriert.")

# TABS FÜR EINGABEN
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Markt & Strategie", 
    "🏢 Organisation & Kosten", 
    "💰 Finanzierung & Assets",
    "🎲 Simulation (Engine)",
    "📑 Reporting (Output)"
])

# --- TAB 1: MARKT & STRATEGIE (Modell 1 Inputs) ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Marktparameter (Bass)")
        M_in = st.number_input("Marktpotenzial (Total)", 10000, 5000000, 100000, step=10000)
        
        st.markdown("**Innovationskoeffizient (p)**")
        p_min = st.number_input("p Min", 0.001, 0.1, 0.01, format="%.4f")
        p_max = st.number_input("p Max", 0.001, 0.1, 0.02, format="%.4f")
        
        st.markdown("**Imitationskoeffizient (q)**")
        q_min = st.number_input("q Min", 0.1, 1.0, 0.3, format="%.2f")
        q_max = st.number_input("q Max", 0.1, 1.0, 0.5, format="%.2f")
        
    with col2:
        st.subheader("Pricing Strategien")
        st.info("Definiere die Bandbreiten für die Simulation")
        
        st.markdown("**(A) Standard Strategie**")
        arpu_std_min = st.number_input("ARPU Min (Std)", 0, 5000, 1800)
        arpu_std_max = st.number_input("ARPU Max (Std)", 0, 5000, 2200)
        
        st.divider()
        st.markdown("**(B) Fighter Strategie**")
        arpu_fight_min = st.number_input("ARPU Min (Fight)", 0, 5000, 1200)
        arpu_fight_max = st.number_input("ARPU Max (Fight)", 0, 5000, 1400)
        
        st.divider()
        st.subheader("Real Option Trigger")
        trigger_threshold = st.slider("Trigger wenn Wachstum unter:", 0.01, 0.20, 0.05)
        gf_active = st.checkbox("Grandfathering? (Bestand behält alten Preis bei Switch)")

# --- TAB 2: ORGA & KOSTEN (Modell 2 Inputs) ---
with tab2:
    st.subheader("Personalplanung (Dynamisch)")
    st.caption("FTEs wachsen automatisch mit dem Umsatz, wenn konfiguriert.")
    st.session_state["staff_df"] = st.data_editor(st.session_state["staff_df"], num_rows="dynamic", use_container_width=True)
    
    st.subheader("Kostenstellen (OpEx)")
    st.session_state["cost_df"] = st.data_editor(st.session_state["cost_df"], num_rows="dynamic", use_container_width=True)

    # Berechne Basis-Fixkosten für die Simulation
    # Annahme: Start-Personal + Fixe Kostenstellen
    staff_base = (st.session_state["staff_df"]["Gehalt (€)"] * st.session_state["staff_df"]["FTE Start"]).sum()
    cc_base = st.session_state["cost_df"]["Fixkosten (€ p.a.)"].sum()
    base_fix_costs = staff_base * 1.25 + cc_base # +25% Lohnnebenkosten Pauschale für Sim

# --- TAB 3: FINANZIERUNG & ASSETS (Modell 2 Inputs) ---
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Finanzierung")
        equity_in = st.number_input("Eigenkapital (€)", value=250000)
        loan_in = st.number_input("Initialer Kredit (€)", value=0)
        interest_in = st.number_input("Zinssatz %", value=5.0)
        tax_in = st.number_input("Steuersatz %", value=25.0)
        min_cash_in = st.number_input("Mindest-Liquidität (€)", value=20000)
        
        st.subheader("Makro")
        wage_inc = st.number_input("Lohnsteigerung % p.a.", value=2.5)
        inflation_in = st.number_input("Inflation % p.a.", value=2.0)
        
    with c2:
        st.subheader("Asset Konfiguration")
        p_laptop = st.number_input("Preis Laptop (€)", value=1500)
        ul_laptop = st.number_input("AfA Dauer Laptop (J)", value=3)
        p_car = st.number_input("Preis PKW (€)", value=40000)
        ul_car = st.number_input("AfA Dauer PKW (J)", value=6)
        misc_capex = st.number_input("Sonstiges Jährl. Capex (€)", value=5000)
        
        # Pack Asset Config
        asset_conf = {
            'p_laptop': p_laptop, 'ul_laptop': ul_laptop,
            'p_car': p_car, 'ul_car': ul_car,
            'misc_capex': misc_capex
        }
        
        # Pack Fin Config
        fin_conf = {
            'equity': equity_in, 'loan': loan_in, 'interest': interest_in,
            'tax': tax_in, 'min_cash': min_cash_in,
            'wage_inc': wage_inc, 'inflation': inflation_in, 'lnk': 25.0
        }

# --- TAB 4: SIMULATION ENGINE ---
with tab4:
    st.header("Monte Carlo Simulation")
    st.markdown(f"Basis-Fixkosten (berechnet aus Tab 2): **{base_fix_costs:,.0f} €**")
    
    T_sim = st.number_input("Laufzeit (Jahre)", 5, 30, 10)
    iterations = st.slider("Simulationen pro Strategie", 100, 2000, 500)
    
    if st.button("▶️ Simulation Starten", type="primary"):
        with st.spinner("Berechne Tausende Szenarien..."):
            
            # 1. Setup der Strategien
            # Fallback (Standard) Werte für Switch Option
            fb_p = (p_min + p_max) / 2
            fb_q = (q_min + q_max) / 2
            fb_arpu = (arpu_std_min + arpu_std_max) / 2
            
            switch_conf_data = {
                'threshold': trigger_threshold,
                'fallback_p': fb_p,
                'fallback_q': fb_q,
                'fallback_arpu': fb_arpu,
                'grandfathering': gf_active,
                'elasticity': 1.5 # Wie stark reagiert Kunde auf Preis
            }
            
            strategies = [
                # Name, Mode, ARPU Range, TriggerConf, SwitchConf
                ("1. Standard", "static", (arpu_std_min, arpu_std_max), None, None),
                ("2. Fighter", "static", (arpu_fight_min, arpu_fight_max), None, None),
                ("3. Switch Option", "switch", (arpu_fight_min, arpu_fight_max), {'threshold': trigger_threshold}, switch_conf_data),
                ("4. Abandon Option", "abandon", (arpu_fight_min, arpu_fight_max), {'threshold': trigger_threshold}, None),
            ]
            
            results = {}
            
            for name, mode, arpu_rng, trig_c, sw_c in strategies:
                sim_data = [] # Stores tuple (N_curve, ARPU_curve, NPV)
                
                for i in range(iterations):
                    N, ARPU_trace, NPV = run_simulation_logic(
                        T=T_sim + 1, # +1 für Index 0
                        start_n=50,
                        M_range=(M_in*0.9, M_in*1.1),
                        p_range=(p_min, p_max),
                        q_range=(q_min, q_max),
                        arpu_range=arpu_rng,
                        churn_range=(0.04, 0.08),
                        base_fix_costs=base_fix_costs,
                        mode=mode,
                        trigger_conf=trig_c,
                        switch_conf=sw_c
                    )
                    sim_data.append((N, ARPU_trace, NPV))
                
                # Sortieren nach NPV um Percentiles zu finden
                sim_data.sort(key=lambda x: x[2]) # Sort by NPV
                
                idx_worst = int(iterations * 0.05)
                idx_base = int(iterations * 0.50)
                idx_best = int(iterations * 0.95)
                
                results[name] = {
                    "Worst": sim_data[idx_worst],
                    "Base": sim_data[idx_base],
                    "Best": sim_data[idx_best],
                    "All_NPV": [x[2] for x in sim_data]
                }
            
            st.session_state.sim_results = results
            st.success("Berechnung abgeschlossen! Ergebnisse in Tab 5.")

# --- TAB 5: REPORTING (Output) ---
with tab5:
    if st.session_state.sim_results is None:
        st.info("Bitte starte zuerst die Simulation in Tab 4.")
    else:
        res = st.session_state.sim_results
        
        st.subheader("Ergebnis-Übersicht")
        cols = st.columns(4)
        for i, (name, data) in enumerate(res.items()):
            npvs = data["All_NPV"]
            cols[i].metric(name, f"Ø {np.mean(npvs)/1e6:.1f}M €", f"Risk: {np.std(npvs)/1e6:.1f}M")
            
        st.divider()
        st.header("Detail-Report Generator")
        
        sel_strat = st.selectbox("Strategie auswählen", list(res.keys()))
        data = res[sel_strat]
        
        col_w, col_b, col_h = st.columns(3)
        
        def render_case(title, sim_tuple, col):
            with col:
                st.subheader(title)
                N_c, ARPU_c, _ = sim_tuple
                
                # DER SCHLÜSSEL: Hier werden Daten aus Tab 1, 2, 3 und 4 gemerged
                df_fin = calculate_financials(
                    N_c, ARPU_c, 
                    st.session_state["staff_df"], 
                    st.session_state["cost_df"], 
                    asset_conf, 
                    fin_conf
                )
                
                # Plot
                fig, ax = plt.subplots(figsize=(4,3))
                ax.plot(df_fin['Jahr'], df_fin['Umsatz'], label="Umsatz", color="#1f77b4")
                ax.plot(df_fin['Jahr'], df_fin['EBITDA'], label="EBITDA", color="#2ca02c")
                ax.set_title("Finanzvorschau")
                ax.legend(fontsize=8)
                st.pyplot(fig)
                
                # KPIs
                sum_profit = df_fin["Jahresüberschuss"].sum()
                st.metric("Kumulierter Gewinn", f"{sum_profit/1e6:.2f}M €")
                
                # PDF Export
                pdf_data = create_pdf(df_fin, sel_strat, title, fig)
                st.download_button(f"📄 PDF {title}", pdf_data, f"Report_{sel_strat}_{title}.pdf", "application/pdf")
                plt.close(fig)

        render_case("Worst Case (P5)", data["Worst"], col_w)
        render_case("Base Case (P50)", data["Base"], col_b)
        render_case("Best Case (P95)", data["Best"], col_h)
        
        with st.expander("🔍 Tabellarische Daten (Base Case) anzeigen"):
             N_base, ARPU_base, _ = data["Base"]
             df_debug = calculate_financials(N_base, ARPU_base, st.session_state["staff_df"], st.session_state["cost_df"], asset_conf, fin_conf)
             st.dataframe(df_debug.style.format("{:,.0f}"))
