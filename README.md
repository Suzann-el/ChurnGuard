    # 📈 ChurnGuard — Prédiction du Churn Clients

## Description
Modèle de machine learning pour prédire quels clients vont résilier leur abonnement
télécom, avec explicabilité SHAP et recommandations de rétention.

## Dataset
**Telco Customer Churn** — IBM (7043 clients, 21 colonnes)
- Source : IBM/telco-customer-churn-on-icp4d (GitHub)
- Fichier : `telco_churn.csv`
- Taux de churn : ~26%

## Stack technique
```
Python · Pandas · NumPy · Scikit-learn
LightGBM · SMOTE (imbalanced-learn)
SHAP · Matplotlib · Seaborn · Plotly
```

## Installation
```bash
pip install pandas numpy scikit-learn lightgbm imbalanced-learn shap matplotlib seaborn plotly
```

## Structure du notebook
| Section | Contenu |
|---------|---------|
| 0. Imports | Configuration de l'environnement |
| 1. Chargement | Lecture du CSV, première exploration |
| 2. EDA | Distribution du churn, analyses par variable |
| 3. Nettoyage | TotalCharges, encodage, suppression colonnes |
| 4. Feature Engineering | charges_per_tenure, nb_services, risque_financier |
| 5. Modélisation | Train/test split → SMOTE → LightGBM |
| 6. Seuil | Optimisation précision/rappel selon contexte métier |
| 7. SHAP | Explicabilité globale et individuelle |
| 8. CV | Cross-validation 5-fold, résultats finaux |
| 9. Prochaines étapes | FastAPI, Streamlit, Power BI |

## Résultats attendus
- AUC-ROC : ~0.84
- F1-Score (classe churn) : ~0.62
- Rappel (classe churn) : ~0.75

## Points clés à comprendre
1. **SMOTE toujours APRÈS le train_test_split** — sinon data leakage
2. **Optimisation du seuil** — 0.5 n'est pas toujours optimal
3. **SHAP** — permet d'expliquer POURQUOI un client va churner
4. **class_weight='balanced'** — alternative légère au SMOTE

## Pour aller plus loin
- [ ] Créer une API FastAPI avec endpoint /predict
- [ ] Déployer un dashboard Streamlit sur Render
- [ ] Connecter à Power BI via un fichier CSV exporté
- [ ] Tester sur d'autres datasets (SaaS, médias, banque)

    
