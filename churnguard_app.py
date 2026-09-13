"""
ChurnGuard Dashboard — Visualisation du risque de churn clients
Lancer : streamlit run churnguard_app.py
"""

import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import shap
import plotly.express as px
import plotly.graph_objects as go

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.risk-critique { background:#fee2e2; border-left:5px solid #dc2626; padding:10px; border-radius:4px; }
.risk-eleve    { background:#ffedd5; border-left:5px solid #f97316; padding:10px; border-radius:4px; }
.risk-modere   { background:#fef9c3; border-left:5px solid #eab308; padding:10px; border-radius:4px; }
.risk-faible   { background:#dcfce7; border-left:5px solid #22c55e; padding:10px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── Chargement du modèle ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load("churnguard_model.pkl")
    feature_names = joblib.load("churnguard_features.pkl")
    with open("churnguard_config.json") as f:
        config = json.load(f)
    return model, feature_names, config

try:
    model, feature_names, config = load_model()
    THRESHOLD = config["threshold"]
except Exception as e:
    st.error(f"Impossible de charger le modele : {e}")
    st.info("Lance d'abord le notebook ChurnGuard pour generer les fichiers .pkl")
    st.stop()


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(data_dict):
    data = pd.DataFrame([data_dict])
    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        if data[col].nunique() <= 2:
            data[col] = le.fit_transform(data[col].astype(str))
        else:
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data.drop(columns=[col]), dummies], axis=1)

    tenure = float(data.get('tenure', [1])[0])
    monthly = float(data.get('MonthlyCharges', [0])[0])
    total   = float(data.get('TotalCharges', [0])[0])

    data['charges_per_tenure'] = total / tenure if tenure > 0 else monthly
    service_cols = [c for c in data.columns if any(s in c for s in
        ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
         'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])]
    data['nb_services']      = data[service_cols].sum(axis=1) if service_cols else 0
    data['risque_financier'] = int(monthly > 65 and tenure < 12)
    data['tenure_ratio']     = tenure / 72

    for col in feature_names:
        if col not in data.columns:
            data[col] = 0
    return data[feature_names].astype(float)


def get_risk(proba):
    if proba >= 0.70: return "CRITIQUE", "#dc2626", "critique"
    if proba >= 0.50: return "ELEVE",    "#f97316", "eleve"
    if proba >= 0.30: return "MODERE",   "#eab308", "modere"
    return              "FAIBLE",    "#22c55e", "faible"


def get_reco(risk):
    return {
        "CRITIQUE": "Contact immediat — offre de retention prioritaire (remise, upgrade)",
        "ELEVE":    "Contact sous 48h — proposer un contrat annuel",
        "MODERE":   "Email de fidelisation — offre personnalisee",
        "FAIBLE":   "Client fidele — aucune action requise",
    }.get(risk, "Suivi standard")


# ── HEADER ─────────────────────────────────────────────────────────────────────
col_icon, col_title = st.columns([1, 8])
with col_icon: st.markdown("## 📈")
with col_title:
    st.title("ChurnGuard — Prédiction du Churn Clients")
    st.caption(f"LightGBM · AUC-ROC : {config['auc_roc']} · Seuil : {THRESHOLD:.2f} · v{config['model_version']}")

st.divider()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Analyse d'un client", "Analyse en lot", "Infos modèle"])


# ══════════════════════════════════════════════════════════
# TAB 1 — ANALYSE INDIVIDUELLE
# ══════════════════════════════════════════════════════════
with tab1:
    with st.sidebar:
        st.header("Profil du client")

        with st.expander("Informations personnelles", expanded=True):
            gender         = st.selectbox("Genre", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior (65+)", [0, 1])
            partner        = st.selectbox("Partenaire", ["Yes", "No"])
            dependents     = st.selectbox("Dependants", ["Yes", "No"])

        with st.expander("Contrat & Facturation", expanded=True):
            tenure          = st.slider("Anciennete (mois)", 0, 72, 12)
            contract        = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
            paperless       = st.selectbox("Facturation electronique", ["Yes", "No"])
            payment_method  = st.selectbox("Mode de paiement",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.slider("Charges mensuelles (euros)", 18.0, 119.0, 65.0, 0.5)
            total_charges   = st.number_input("Charges totales (euros)", 0.0, 8700.0,
                                               float(monthly_charges * max(tenure, 1)), 10.0)

        with st.expander("Services", expanded=True):
            phone_service  = st.selectbox("Service telephonique", ["Yes", "No"])
            multiple_lines = st.selectbox("Lignes multiples", ["Yes", "No", "No phone service"])
            internet       = st.selectbox("Service internet", ["Fiber optic", "DSL", "No"])
            online_security= st.selectbox("Securite en ligne", ["Yes", "No", "No internet service"])
            online_backup  = st.selectbox("Sauvegarde en ligne", ["Yes", "No", "No internet service"])
            device_prot    = st.selectbox("Protection appareil", ["Yes", "No", "No internet service"])
            tech_support   = st.selectbox("Support technique", ["Yes", "No", "No internet service"])
            streaming_tv   = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies=st.selectbox("Streaming films", ["Yes", "No", "No internet service"])

        predict_btn = st.button("Analyser ce client", use_container_width=True, type="primary")

    if predict_btn:
        client = {
            "gender": gender, "SeniorCitizen": senior_citizen,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_prot, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
        }

        with st.spinner("Analyse en cours..."):
            X = preprocess(client)
            proba = float(model.predict_proba(X)[0, 1])
            churn = proba >= THRESHOLD
            risk_label, risk_color, risk_class = get_risk(proba)

            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
            if isinstance(sv, list): sv_arr = sv[1]
            elif hasattr(sv, 'ndim') and sv.ndim == 3: sv_arr = sv[:, :, 1]
            else: sv_arr = sv

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Impact":  sv_arr[0],
                "Valeur":  X.values[0]
            }).sort_values("Impact", key=abs, ascending=False)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Probabilite de churn", f"{proba:.1%}")
        k2.metric("Niveau de risque", risk_label)
        k3.metric("Decision", "CHURN PROBABLE" if churn else "CLIENT FIDELE")
        k4.metric("Anciennete", f"{tenure} mois")

        st.divider()
        col_gauge, col_shap = st.columns(2)

        with col_gauge:
            st.subheader("Score de risque")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Probabilite de churn (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color, "thickness": 0.3},
                    "steps": [
                        {"range": [0,  30], "color": "#dcfce7"},
                        {"range": [30, 50], "color": "#fef9c3"},
                        {"range": [50, 70], "color": "#ffedd5"},
                        {"range": [70,100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#1e293b", "width": 3},
                        "thickness": 0.8,
                        "value": THRESHOLD * 100
                    }
                },
                number={"suffix": "%", "font": {"size": 36}}
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(
                f'<div class="risk-{risk_class}"><b>Recommandation</b><br/>{get_reco(risk_label)}</div>',
                unsafe_allow_html=True)

        with col_shap:
            st.subheader("Facteurs déterminants")
            top8 = shap_df.head(8)
            fig_shap = go.Figure(go.Bar(
                x=top8["Impact"], y=top8["Feature"], orientation="h",
                marker_color=["#dc2626" if v > 0 else "#22c55e" for v in top8["Impact"]],
                text=[f"{v:+.3f}" for v in top8["Impact"]], textposition="outside"
            ))
            fig_shap.update_layout(
                height=280, xaxis_title="Impact SHAP",
                margin=dict(t=10, b=10, l=10, r=60),
                xaxis=dict(zeroline=True, zerolinecolor="#94a3b8", zerolinewidth=2)
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("Rouge = augmente le risque · Vert = diminue le risque")

        st.subheader("Détail des 5 facteurs les plus importants")
        top5 = shap_df.head(5).copy()
        top5["Direction"] = top5["Impact"].apply(lambda x: "Augmente le risque" if x > 0 else "Diminue le risque")
        top5["Impact absolu"] = top5["Impact"].abs().round(4)
        st.dataframe(top5[["Feature","Valeur","Impact","Impact absolu","Direction"]].reset_index(drop=True),
                     use_container_width=True)

    else:
        st.info("Renseignez le profil du client dans le panneau gauche et cliquez sur 'Analyser ce client'")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Comment utiliser")
            st.markdown("""
1. Remplissez le profil client à gauche
2. Cliquez sur **Analyser ce client**
3. Consultez le score de risque et les facteurs SHAP
4. Suivez la recommandation commerciale
            """)
        with c2:
            st.markdown("#### Niveaux de risque")
            st.markdown("""
| Niveau | Probabilité | Action |
|--------|-------------|--------|
| FAIBLE | < 30% | Aucune action |
| MODÉRÉ | 30-50% | Email fidelisation |
| ÉLEVÉ | 50-70% | Appel sous 48h |
| CRITIQUE | > 70% | Action immediate |
            """)


# ══════════════════════════════════════════════════════════
# TAB 2 — ANALYSE EN LOT
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Analyse en lot — Importer un fichier CSV")
    st.caption("Format attendu : memes colonnes que telco_churn.csv (sans customerID ni Churn)")

    uploaded = st.file_uploader("Importer un CSV clients", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"Fichier charge : {len(df_up)} clients")
        st.dataframe(df_up.head(5), use_container_width=True)

        if st.button("Analyser tous les clients", type="primary"):
            with st.spinner(f"Analyse de {len(df_up)} clients..."):
                probas = []
                for _, row in df_up.iterrows():
                    try:
                        X_r = preprocess(row.to_dict())
                        probas.append(float(model.predict_proba(X_r)[0, 1]))
                    except:
                        probas.append(0.0)

            df_up['probabilite_churn'] = probas
            df_up['niveau_risque'] = [get_risk(p)[0] for p in probas]
            df_up['churn_predit'] = [int(p >= THRESHOLD) for p in probas]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total clients", len(df_up))
            c2.metric("Churns predits", df_up['churn_predit'].sum())
            c3.metric("Taux de churn predit", f"{df_up['churn_predit'].mean():.1%}")
            c4.metric("Proba moyenne", f"{np.mean(probas):.1%}")

            fig_risk = px.pie(df_up, names='niveau_risque',
                              title="Repartition par niveau de risque",
                              color='niveau_risque',
                              color_discrete_map={"FAIBLE":"#22c55e","MODERE":"#eab308",
                                                  "ELEVE":"#f97316","CRITIQUE":"#dc2626"})
            st.plotly_chart(fig_risk, use_container_width=True)
            st.dataframe(df_up.sort_values('probabilite_churn', ascending=False),
                         use_container_width=True)
            csv_exp = df_up.to_csv(index=False).encode('utf-8')
            st.download_button("Telecharger les resultats", csv_exp,
                               "churnguard_resultats.csv", "text/csv")

    else:
        st.info("Importez un CSV pour analyser plusieurs clients en une seule fois")

        st.subheader("Demo — Clients aleatoires")
        n_demo = st.slider("Nombre de clients", 10, 200, 50)
        if st.button("Generer et analyser"):
            np.random.seed(42)
            demo = pd.DataFrame({
                'tenure':         np.random.randint(0, 72, n_demo),
                'MonthlyCharges': np.random.uniform(18, 119, n_demo).round(2),
                'TotalCharges':   np.random.uniform(0, 8000, n_demo).round(2),
                'Contract':       np.random.choice(['Month-to-month','One year','Two year'],
                                                    n_demo, p=[0.55, 0.25, 0.20]),
                'InternetService':np.random.choice(['Fiber optic','DSL','No'], n_demo, p=[0.44,0.34,0.22]),
                'gender':         np.random.choice(['Male','Female'], n_demo),
                'SeniorCitizen':  np.random.choice([0,1], n_demo, p=[0.84,0.16]),
                'Partner':        np.random.choice(['Yes','No'], n_demo),
                'Dependents':     np.random.choice(['Yes','No'], n_demo),
                'PhoneService':   np.random.choice(['Yes','No'], n_demo, p=[0.90,0.10]),
                'MultipleLines':  np.random.choice(['Yes','No','No phone service'], n_demo),
                'OnlineSecurity': np.random.choice(['Yes','No','No internet service'], n_demo),
                'OnlineBackup':   np.random.choice(['Yes','No','No internet service'], n_demo),
                'DeviceProtection':np.random.choice(['Yes','No','No internet service'], n_demo),
                'TechSupport':    np.random.choice(['Yes','No','No internet service'], n_demo),
                'StreamingTV':    np.random.choice(['Yes','No','No internet service'], n_demo),
                'StreamingMovies':np.random.choice(['Yes','No','No internet service'], n_demo),
                'PaperlessBilling':np.random.choice(['Yes','No'], n_demo),
                'PaymentMethod':  np.random.choice(['Electronic check','Mailed check',
                                  'Bank transfer (automatic)','Credit card (automatic)'], n_demo),
            })

            probas_demo = []
            for _, row in demo.iterrows():
                try:
                    X_d = preprocess(row.to_dict())
                    probas_demo.append(float(model.predict_proba(X_d)[0, 1]))
                except:
                    probas_demo.append(0.0)

            demo['probabilite_churn'] = probas_demo
            demo['niveau_risque']     = [get_risk(p)[0] for p in probas_demo]
            demo['churn_predit']      = [int(p >= THRESHOLD) for p in probas_demo]

            c1, c2, c3 = st.columns(3)
            c1.metric("Clients analyses", n_demo)
            c2.metric("Churns predits", demo['churn_predit'].sum())
            c3.metric("Taux predit", f"{demo['churn_predit'].mean():.1%}")

            fig_hist = px.histogram(demo, x='probabilite_churn', color='niveau_risque',
                                    nbins=20, title="Distribution des probabilites de churn",
                                    color_discrete_map={"FAIBLE":"#22c55e","MODERE":"#eab308",
                                                        "ELEVE":"#f97316","CRITIQUE":"#dc2626"})
            st.plotly_chart(fig_hist, use_container_width=True)
            st.dataframe(demo.sort_values('probabilite_churn', ascending=False).head(20),
                         use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 3 — INFOS MODELE
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("Informations sur le modele ChurnGuard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithme", "LightGBM")
    c2.metric("AUC-ROC", config['auc_roc'])
    c3.metric("Seuil de decision", f"{THRESHOLD:.2f}")
    c4.metric("Version", config['model_version'])

    st.divider()
    col_info, col_feat = st.columns(2)

    with col_info:
        st.markdown("#### Description")
        st.markdown(f"""
- **Type :** Classification binaire (churn = 1, retained = 0)
- **Algorithme :** LightGBM avec gestion du desequilibre
- **Seuil :** {THRESHOLD:.2f} (optimise pour recall >= 70%)
- **AUC-ROC :** {config['auc_roc']}
- **Dataset :** IBM Telco Customer Churn (7043 clients)

#### Niveaux de risque
| Niveau | Probabilite | Action |
|--------|-------------|--------|
| FAIBLE | < 30% | Aucune action |
| MODERE | 30-50% | Email fidelisation |
| ELEVE | 50-70% | Appel sous 48h |
| CRITIQUE | > 70% | Action immediate |
        """)

    with col_feat:
        st.markdown("#### Features du modele")
        feat_df = pd.DataFrame({"Feature": feature_names, "Index": range(len(feature_names))})
        st.dataframe(feat_df, use_container_width=True, height=350)

        st.markdown("#### Features creees")
        st.markdown("""
| Feature | Calcul | Interpretation |
|---------|--------|----------------|
| charges_per_tenure | TotalCharges / tenure | Charge mensuelle moyenne |
| nb_services | Somme des services | Engagement du client |
| risque_financier | Charges > Q3 ET tenure < 12 | Profil fragile |
| tenure_ratio | tenure / 72 | Anciennete normalisee |
        """)
