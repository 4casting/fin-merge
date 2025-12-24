import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
import io
import datetime
import tempfile

# ==========================================
# 0. SETUP & HELPER
# ==========================================
st.set_page_config(page_title="Master Integrated Valuation", layout="wide", initial_sidebar_state="expanded")

if 'sim_results' not in st.session_state: st.session_state.sim_results = None

def safe_float(val):
    try: return float(val)
    except: return 0.0

# Initialisierung der Finanz-Daten (aus Modell 2)
if "staff_df" not in st.session_state:
    st.session_state["staff_df"] = pd.DataFrame([
        {"Rolle": "Management", "Gehalt (€)": 120000, "FTE Jahr 1": 1.0, "Steigerung/Umsatz": 0.0, "Laptop": True, "Auto": True},
        {"Rolle": "Sales", "Gehalt (€)": 60000, "FTE Jahr 1": 2.0, "Steigerung/Umsatz": 0.00005, "Laptop": True, "Auto": True},
        {"Rolle": "Tech/Support", "Gehalt (€)": 55000, "FTE Jahr 1": 3.0, "Steigerung/Umsatz": 0.00002, "Laptop": True, "Auto": False},
        {"Rolle": "Admin", "Gehalt (€)": 45000, "FTE Jahr 1": 1.0, "Steigerung/Umsatz": 0.00001, "Laptop": False, "Auto": False},
    ])

if "cost_df" not in st.session_state:
    st.session_state["cost_df"] = pd.DataFrame([
        {"Kostenstelle": "Büromiete", "Fix (€/Jahr)": 24000, "Variabel (% v. Umsatz)": 0.0},
        {"Kostenstelle": "Server/IT", "Fix (€/Jahr)": 5000, "Variabel (% v. Umsatz)": 2.0},
        {"Kostenstelle": "Marketing", "Fix (€/Jahr)": 50000, "Variabel (% v. Umsatz)": 5.0},
        {"Kostenstelle": "Beratung/Legal", "Fix (€/Jahr)": 10000, "Variabel (% v. Umsatz)": 0.0},
    ])

if "params_fin" not in st.session_state:
    st.session_state["params_fin"] = {
        "equity": 200000.0, "loan": 0.0, "loan_rate": 5.0, "tax": 25.0,
        "wage_inc": 2.5, "inflation": 2.0, "lnk": 25.0, # Lohnnebenkosten
        "dso": 30, "dpo": 30, "min_cash": 10000.0
    }

if "params_asset" not in st.session_state:
    st.session_state["params_asset"] = {
        "p_laptop": 1500, "ul_laptop": 3,
        "p_auto": 40000, "ul_auto": 6,
        "capex_misc": 5000, "ul_misc": 5
    }

# ==========================================
# 1. LOGIK: FINANZ-ENGINE (Detailliert)
# ==========================================
def calculate_financial_plan(T_years, N_series, ARPU_series, staff_df, cost_df, fin_par, asset_par):
    """
    Erstellt den detaillierten Finanzplan basierend auf einer exakten Kunden- und Preiskurve.
    """
    # Vorbereitung
    rows = []
    
    # State Vars
    cash = fin_par["equity"]
    debt = fin_par["loan"]
    retained_earnings = 0.0
    fixed_assets_net = 0.0
    loss_carryforward = 0.0
    
    # Asset Register (Liste von Dicts: {'year_bought': t, 'value': v, 'ul': 3})
    asset_registry = [] 
    
    # Datenaufbereitung
    staff = staff_df.to_dict('records')
    ccs = cost_df.to_dict('records')
    
    # Simulation Loop (Jahre 1 bis T)
    # Achtung: N_series startet oft bei t=0. Wir mappen Simulation t auf Geschäftsjahr t+1
    
    limit = min(len(N_series), T_years + 1)
    
    for t in range(1, limit):
        row = {"Jahr": t}
        
        # --- A. UMSATZ (Aus Simulation) ---
        n_curr = N_series[t]
        arpu_curr = ARPU_series[t]
        
        if n_curr <= 0: # Abandoned
            rows.append({k:0 for k in ["Umsatz", "EBITDA", "Jahresüberschuss", "Kasse"]})
            rows[-1]["Jahr"] = t
            continue

        revenue = n_curr * arpu_curr
        row["Kunden"] = n_curr
        row["ARPU"] = arpu_curr
        row["Umsatz"] = revenue
        
        # --- B. KOSTEN ---
        # 1. COGS (Vereinfacht: Wir nehmen an, Delta Margin aus Sim ist Deckungsbeitrag. 
        # Hier nehmen wir pauschal an, dass COGS ca 20% sind, falls nicht anders definiert)
        cogs = revenue * 0.15 
        row["COGS"] = cogs
        
        # 2. Personal (Skalierung mit Umsatz)
        wage_factor = (1 + fin_par["wage_inc"]/100)**(t-1)
        lnk_factor = (1 + fin_par["lnk"]/100)
        
        personnel_cost = 0.0
        fte_count = 0.0
        new_assets_needed = {"Laptop": 0, "Auto": 0}
        
        for s in staff:
            # Dynamische FTE Berechnung: Basis + (Umsatz * Faktor)
            fte_needed = s["FTE Jahr 1"] + (revenue * s["Steigerung/Umsatz"])
            cost = fte_needed * s["Gehalt (€)"] * wage_factor * lnk_factor
            personnel_cost += cost
            fte_count += fte_needed
            
            if s["Laptop"]: new_assets_needed["Laptop"] += fte_needed
            if s["Auto"]: new_assets_needed["Auto"] += fte_needed
            
        row["Personal"] = personnel_cost
        row["FTE"] = fte_count
        
        # 3. Sonstige Opex (Kostenstellen)
        opex_fix = 0.0
        opex_var = 0.0
        infl_factor = (1 + fin_par["inflation"]/100)**(t-1)
        
        for c in ccs:
            opex_fix += c["Fix (€/Jahr)"] * infl_factor
            opex_var += (c["Variabel (% v. Umsatz)"]/100) * revenue
            
        row["OpEx_Sonst"] = opex_fix + opex_var
        
        total_opex = personnel_cost + row["OpEx_Sonst"]
        ebitda = revenue - cogs - total_opex
        row["EBITDA"] = ebitda
        
        # --- C. ASSETS & AFA ---
        # Asset Bedarf prüfen vs. Bestand
        # Vereinfachung: Wir kaufen jedes Jahr neu für den Bestand (delta) und ersetzen alte
        # Besser: Capex pauschal + Delta FTE
        
        capex = asset_par["capex_misc"] # Laufendes Invest
        
        # Einfache Asset Logik: Pro FTE ein Laptop/Auto wenn nötig
        # Hier vereinfacht: Wir nehmen an, wir investieren ca. 2% vom Umsatz in Hardware + Fixum
        capex_it = (fte_count * 200) + asset_par["capex_misc"]
        
        # Zum Register hinzufügen
        asset_registry.append({'val': capex_it, 'ul': asset_par["ul_misc"], 'age': 0})
        
        # Abschreibung berechnen
        afa = 0.0
        new_registry = []
        for asset in asset_registry:
            if asset['age'] < asset['ul']:
                rate = asset['val'] / asset['ul']
                afa += rate
                asset['age'] += 1
                new_registry.append(asset)
        asset_registry = new_registry
        
        row["AfA"] = afa
        row["Capex"] = capex_it
        
        # --- D. ERGEBNIS ---
        ebit = ebitda - afa
        interest = debt * (fin_par["loan_rate"]/100)
        ebt = ebit - interest
        
        tax = 0.0
        if ebt < 0:
            loss_carryforward += abs(ebt)
        else:
            use_loss = min(ebt, loss_carryforward)
            loss_carryforward -= use_loss
            tax_base = ebt - use_loss
            tax = tax_base * (fin_par["tax"]/100)
            
        net_income = ebt - tax
        row["Jahresüberschuss"] = net_income
        
        # --- E. CASHFLOW & BILANZ ---
        # Working Capital Change (vereinfacht)
        # Forderungen steigen wenn Umsatz steigt
        receivables = revenue * (fin_par["dso"]/360)
        payables = (cogs + row["OpEx_Sonst"]) * (fin_par["dpo"]/360)
        
        prev_nwc = rows[-1]["NWC"] if len(rows) > 0 else 0
        curr_nwc = receivables - payables
        delta_nwc = curr_nwc - prev_nwc
        row["NWC"] = curr_nwc
        
        cf_op = net_income + afa - delta_nwc
        cf_inv = -capex_it
        
        cash_pre = cash + cf_op + cf_inv
        
        borrow = 0.0
        repay = 0.0
        
        if cash_pre < fin_par["min_cash"]:
            borrow = fin_par["min_cash"] - cash_pre
        elif cash_pre > fin_par["min_cash"] and debt > 0:
            repay = min(debt, cash_pre - fin_par["min_cash"])
            
        cash_final = cash_pre + borrow - repay
        debt_final = debt + borrow - repay
        
        row["Kasse"] = cash_final
        row["Schulden"] = debt_final
        row["Cashflow"] = cf_op + cf_inv + borrow - repay
        
        # Bilanz Übergabe
        cash = cash_final
        debt = debt_final
        fixed_assets_net = max(0, fixed_assets_net + capex_it - afa)
        retained_earnings += net_income
        
        row["Eigenkapital"] = fin_par["equity"] + retained_earnings
        row["Bilanzsumme"] = fixed_assets_net + cash_final + receivables
        
        rows.append(row)
        
    return pd.DataFrame(rows)

# ==========================================
# 2. LOGIK: SIMULATION ENGINE (Bass + Options)
# ==========================================
def run_simulation_logic(M, p, q, C, ARPU, kappa, Delta_CM, Fixed_Cost, start, T, 
                   mode='static', trigger_val=0.05, fallback_params=None,
                   switch_config=None):
    
    # Initialisierung Arrays
    N = np.zeros(T)
    W = np.zeros(T)
    ARPU_trace = np.zeros(T) # WICHTIG: Wir speichern den Preisverlauf für Modell 2
    
    N[0] = start
    W[0] = -Fixed_Cost # T=0 Invest
    ARPU_trace[0] = ARPU
    
    # Runtime Variablen
    curr_p, curr_q, curr_C = p, q, C
    curr_ARPU, curr_FC = ARPU, Fixed_Cost
    curr_M = M
    
    option_exercised = False
    project_dead = False
    
    growth_rates = []
    
    for t in range(1, T):
        if project_dead:
            N[t] = 0; W[t] = 0; ARPU_trace[t] = 0
            continue
            
        N_prev = N[t-1]
        
        # 1. Bass Diffusion
        # Potential = (p + q * (N/M)) * (M - N)
        pot_new = (curr_p + curr_q * (N_prev/curr_M)) * (curr_M - N_prev)
        if pot_new < 0: pot_new = 0
        
        rate = pot_new / curr_M # Share of Market Growth
        
        # 2. Option Trigger Check (ab Jahr 3)
        if mode != 'static' and not option_exercised and t >= 3:
            avg_growth = np.mean(growth_rates) if growth_rates else 0
            
            if avg_growth < trigger_val:
                option_exercised = True
                
                if mode == 'abandon':
                    project_dead = True
                    N[t] = 0; W[t] = 0; ARPU_trace[t] = 0
                    continue
                    
                elif mode == 'switch' and fallback_params:
                    # Switch Logik: Wir fallen auf die "sichere" Strategie zurück
                    # Aber: Das verursacht einen Schock (Preis hoch -> Kunden weg)
                    
                    # Parameter Swap
                    curr_p = fallback_params['p']
                    # Reputationsschaden auf q
                    curr_q = fallback_params['q'] * switch_config.get('q_penalty', 0.8) 
                    curr_C = fallback_params['C']
                    curr_FC = fallback_params['Fixed_Cost']
                    
                    target_arpu = fallback_params['ARPU']
                    
                    # Preisschock berechnen
                    price_increase = (target_arpu - curr_ARPU) / curr_ARPU if curr_ARPU > 0 else 0
                    churn_shock = min(0.5, price_increase * switch_config.get('elasticity', 1.0))
                    
                    # Bestand reduzieren
                    N_prev = N_prev * (1 - churn_shock)
                    curr_ARPU = target_arpu
                    
                    # Neuberechnung Bass mit neuen Werten
                    pot_new = (curr_p + curr_q * (N_prev/curr_M)) * (curr_M - N_prev)
                    if pot_new < 0: pot_new = 0

        growth_rates.append(rate)
        
        # 3. Bestandsberechnung
        churned = N_prev * curr_C
        N[t] = N_prev - churned + pot_new
        if N[t] > curr_M: N[t] = curr_M
        
        # 4. Value Valuation (High Level für Simulation Ranking)
        rev = N[t] * curr_ARPU
        # Cannibalization Proxy
        cannib = pot_new * kappa * Delta_CM
        W[t] = rev - cannib - curr_FC
        
        ARPU_trace[t] = curr_ARPU
        
    return N, W, np.sum(W), ARPU_trace

# ==========================================
# 3. PDF GENERATOR
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Integriertes Strategie- & Finanz-Gutachten', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Seite {self.page_no()}', 0, 0, 'C')

def generate_detailed_pdf(scenario_name, case_name, df_fin, sim_plot_fig):
    pdf = PDFReport()
    pdf.add_page()
    
    # Title Info
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Strategie: {scenario_name}", 0, 1)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Szenario: {case_name} (Automatisch selektiert)", 0, 1)
    pdf.cell(0, 8, f"Datum: {datetime.datetime.now().strftime('%d.%m.%Y')}", 0, 1)
    pdf.ln(5)
    
    # KPIs
    last_row = df_fin.iloc[-1]
    cum_cash = last_row['Kasse']
    cum_profit = df_fin['Jahresüberschuss'].sum()
    
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(60, 10, f"Cash Jahr 10: {cum_cash:,.0f} EUR", 1, 0, 'C', 1)
    pdf.cell(60, 10, f"Kum. Gewinn: {cum_profit:,.0f} EUR", 1, 1, 'C', 1)
    pdf.ln(10)
    
    # Plot einfügen
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        sim_plot_fig.savefig(tmpfile.name, dpi=100, bbox_inches="tight")
        pdf.image(tmpfile.name, x=10, w=190)
    pdf.ln(10)
    
    # Tabelle GuV (Auszug)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 10, "Detaillierte Finanzkennzahlen (Auszug)", 0, 1)
    
    cols = ["Jahr", "Umsatz", "EBITDA", "Jahresüberschuss", "Kasse"]
    col_width = 35
    
    pdf.set_font("Arial", "B", 8)
    for c in cols: pdf.cell(col_width, 6, c, 1, 0, 'C', 1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 8)
    for _, row in df_fin.iterrows():
        for c in cols:
            val = row[c]
            txt = f"{val:,.0f}" if isinstance(val, (int, float)) else str(val)
            pdf.cell(col_width, 6, txt, 1, 0, 'R')
        pdf.ln()
        
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 4. UI: SEITENSTRUKTUR
# ==========================================

st.title("💠 Integrated Real Options & Financial Factory")
st.markdown("""
Dieses Modell verbindet **Markt-Unsicherheit (Bass Diffusion)** mit **betriebswirtschaftlicher Realität (Finanzplan)**.
1. Definiere die Kostenstruktur der Firma.
2. Simuliere Marktszenarien (Standard vs. Fighter vs. Switch).
3. Erzeuge detaillierte PDF-Reports für Worst/Base/Best Cases.
""")

tab_config, tab_sim, tab_results = st.tabs(["1. Firmen-Setup (Kosten)", "2. Markt-Simulation", "3. Reports & Export"])

# --- TAB 1: KOSTEN SETUP ---
with tab_config:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Personal & Struktur")
        st.session_state["staff_df"] = st.data_editor(st.session_state["staff_df"], num_rows="dynamic")
        
        # Fixkosten Berechnung für Simulation
        staff_fix = st.session_state["staff_df"]["Gehalt (€)"].sum() * (1 + st.session_state["params_fin"]["lnk"]/100)
        
    with c2:
        st.subheader("Kostenstellen (OpEx)")
        st.session_state["cost_df"] = st.data_editor(st.session_state["cost_df"], num_rows="dynamic")
        
        cc_fix = st.session_state["cost_df"]["Fix (€/Jahr)"].sum()
    
    st.divider()
    
    fc_1, fc_2, fc_3 = st.columns(3)
    with fc_1:
        st.subheader("Finanzierung")
        p = st.session_state["params_fin"]
        p["equity"] = st.number_input("Eigenkapital", value=p["equity"])
        p["tax"] = st.number_input("Steuersatz %", value=p["tax"])
        
    with fc_2:
        st.subheader("Assets")
        a = st.session_state["params_asset"]
        a["capex_misc"] = st.number_input("Jährliches Capex Budget", value=a["capex_misc"])
        
    # Total Fixkosten Proxy berechnen
    total_fix_proxy = staff_fix + cc_fix + a["capex_misc"]
    with fc_3:
        st.info(f"ℹ️ Berechnete Basis-Fixkosten für Simulation:\n\n**{total_fix_proxy:,.0f} € / Jahr**\n\n(Wird automatisch in Tab 2 übernommen)")

# --- TAB 2: SIMULATION ---
with tab_sim:
    s1, s2 = st.columns([1, 3])
    with s1:
        st.header("Parameter")
        T_sim = st.slider("Laufzeit (Jahre)", 5, 20, 10)
        M_sim = st.number_input("Marktpotenzial", 10000, 1000000, 50000)
        
        st.markdown("### Real Option Trigger")
        trig_growth = st.slider("Wachstumsgrenze (<%)", 0.0, 0.2, 0.05)
        st.caption("Wenn durchschnittliches Wachstum unter X% fällt, wird gewechselt/abgebrochen.")
        
    with s2:
        st.markdown("#### Szenarien Konfiguration")
        
        def range_input(label, default_min, default_max, key):
            c_a, c_b = st.columns(2)
            rmin = c_a.number_input(f"{label} Min", value=default_min, key=f"{key}_min")
            rmax = c_b.number_input(f"{label} Max", value=default_max, key=f"{key}_max")
            return (rmin, rmax)
        
        st.subheader("Option A: Standard (Hoher Preis, langsames Wachstum)")
        p_a = range_input("Innovation (p)", 0.005, 0.010, "pa")
        q_a = range_input("Imitation (q)", 0.15, 0.25, "qa")
        arpu_a = range_input("ARPU (€)", 2000.0, 2200.0, "arpua")
        
        st.divider()
        
        st.subheader("Option B: Fighter (Niedriger Preis, aggressiv)")
        p_b = range_input("Innovation (p)", 0.020, 0.040, "pb")
        q_b = range_input("Imitation (q)", 0.30, 0.45, "qb")
        arpu_b = range_input("ARPU (€)", 1200.0, 1400.0, "arpub")
        
        # Fixkosten nehmen wir global
        fc_sim = (total_fix_proxy, total_fix_proxy)
        
    if st.button("🚀 Monte Carlo Simulation Starten", type="primary", use_container_width=True):
        with st.spinner("Simuliere 1.000 Pfade pro Strategie..."):
            
            # Switch Config
            sw_conf = {'q_penalty': 0.8, 'elasticity': 1.5}
            
            # Parameter Packs
            # Fallback Params (Standard) für Switch Option
            fb_params = {
                'p': (p_a[0]+p_a[1])/2, 
                'q': (q_a[0]+q_a[1])/2, 
                'C': 0.05, 
                'ARPU': (arpu_a[0]+arpu_a[1])/2, 
                'Fixed_Cost': total_fix_proxy
            }
            
            strategies = [
                ("1. Standard", {'p':p_a, 'q':q_a, 'ARPU':arpu_a}, 'static', None),
                ("2. Fighter", {'p':p_b, 'q':q_b, 'ARPU':arpu_b}, 'static', None),
                ("3. Switch Option", {'p':p_b, 'q':q_b, 'ARPU':arpu_b}, 'switch', fb_params),
                ("4. Abandon Option", {'p':p_b, 'q':q_b, 'ARPU':arpu_b}, 'abandon', None),
            ]
            
            results = {}
            ITERATIONS = 500
            
            for name, rng, mode, fb in strategies:
                sim_n = []; sim_arpu = []; sim_w = []
                
                for i in range(ITERATIONS):
                    # Randomize Inputs
                    curr_p = np.random.uniform(rng['p'][0], rng['p'][1])
                    curr_q = np.random.uniform(rng['q'][0], rng['q'][1])
                    curr_arpu = np.random.uniform(rng['ARPU'][0], rng['ARPU'][1])
                    curr_c = np.random.uniform(0.05, 0.08) # Churn Range
                    
                    # Run Logic
                    N_trace, W_trace, W_sum, ARPU_trace = run_simulation_logic(
                        M=M_sim, p=curr_p, q=curr_q, C=curr_c, ARPU=curr_arpu, 
                        kappa=0.1, Delta_CM=100, Fixed_Cost=total_fix_proxy, 
                        start=50, T=T_sim, mode=mode, trigger_val=trig_growth, 
                        fallback_params=fb, switch_config=sw_conf
                    )
                    
                    sim_n.append(N_trace)
                    sim_arpu.append(ARPU_trace)
                    sim_w.append(W_sum)
                
                # Identify Cases (Best/Base/Worst by NPV)
                arr_w = np.array(sim_w)
                idx_sort = np.argsort(arr_w)
                
                idx_worst = idx_sort[int(ITERATIONS * 0.05)]
                idx_base = idx_sort[int(ITERATIONS * 0.50)]
                idx_best = idx_sort[int(ITERATIONS * 0.95)]
                
                results[name] = {
                    "mean_npv": np.mean(arr_w),
                    "std_npv": np.std(arr_w),
                    "all_n": np.array(sim_n),
                    "cases": {
                        "Worst Case": {"N": sim_n[idx_worst], "ARPU": sim_arpu[idx_worst]},
                        "Base Case": {"N": sim_n[idx_base], "ARPU": sim_arpu[idx_base]},
                        "Best Case": {"N": sim_n[idx_best], "ARPU": sim_arpu[idx_best]},
                    }
                }
            
            st.session_state.sim_results = results
            st.success("Simulation berechnet. Gehe zu Tab 3 für Ergebnisse.")

# --- TAB 3: ERGEBNISSE & REPORTING ---
with tab_results:
    if st.session_state.sim_results is None:
        st.warning("Bitte zuerst Simulation in Tab 2 starten.")
    else:
        res = st.session_state.sim_results
        
        # 1. Übersicht
        st.markdown("### Strategie-Vergleich")
        cols = st.columns(len(res))
        for i, (name, data) in enumerate(res.items()):
            cols[i].metric(name, f"{data['mean_npv']/1e6:.2f} Mio €", f"Risk: {data['std_npv']/1e6:.2f} Mio")
            
        # 2. Detail Analyse mit PDF Engine
        st.divider()
        st.subheader("Detaillierter Finanzplan Generator")
        
        sel_strat = st.selectbox("Wähle Strategie:", list(res.keys()))
        
        # Hole Daten für gewählte Strategie
        strat_data = res[sel_strat]
        cases = strat_data["cases"]
        
        c1, c2, c3 = st.columns(3)
        
        # Helper zum Plotten und Rechnen
        def process_case(case_label, col_container):
            with col_container:
                st.markdown(f"**{case_label}**")
                N_curve = cases[case_label]["N"]
                ARPU_curve = cases[case_label]["ARPU"]
                
                # FINANZPLAN BERECHNEN
                df_fin = calculate_financial_plan(
                    len(N_curve), N_curve, ARPU_curve,
                    st.session_state["staff_df"],
                    st.session_state["cost_df"],
                    st.session_state["params_fin"],
                    st.session_state["params_asset"]
                )
                
                # Mini Stats
                rev_sum = df_fin["Umsatz"].sum()
                ebitda_sum = df_fin["EBITDA"].sum()
                st.caption(f"Ø Umsatz: {rev_sum/len(df_fin):,.0f}")
                st.caption(f"Ø EBITDA: {ebitda_sum/len(df_fin):,.0f}")
                
                # Plot erstellen (für Streamlit UND PDF)
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot(df_fin["Jahr"], df_fin["Umsatz"], label="Umsatz", color="blue")
                ax.plot(df_fin["Jahr"], df_fin["OpEx_Sonst"]+df_fin["Personal"], label="Kosten", color="red", ls="--")
                ax.fill_between(df_fin["Jahr"], 0, df_fin["EBITDA"], color="green", alpha=0.3, label="EBITDA")
                ax.set_title(f"{case_label}")
                ax.legend(fontsize='x-small')
                st.pyplot(fig)
                
                # PDF Button
                pdf_bytes = generate_detailed_pdf(sel_strat, case_label, df_fin, fig)
                st.download_button(
                    f"📄 Report {case_label}", 
                    pdf_bytes, 
                    f"{sel_strat}_{case_label}.pdf", 
                    "application/pdf",
                    key=f"btn_{sel_strat}_{case_label}"
                )
                plt.close(fig)

        process_case("Worst Case", c1)
        process_case("Base Case", c2)
        process_case("Best Case", c3)
        
        st.divider()
        with st.expander("🔍 Blick in die Rohdaten (Base Case)"):
             N_base = cases["Base Case"]["N"]
             ARPU_base = cases["Base Case"]["ARPU"]
             df_debug = calculate_financial_plan(
                    len(N_base), N_base, ARPU_base,
                    st.session_state["staff_df"],
                    st.session_state["cost_df"],
                    st.session_state["params_fin"],
                    st.session_state["params_asset"]
             )
             st.dataframe(df_debug.style.format("{:,.0f}"))
