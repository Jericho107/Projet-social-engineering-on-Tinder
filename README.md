# Speed Dating Analytics

Projet portfolio complet sur le dataset Speed Dating, conçu pour un rendu JEDHA et un usage en entretien Data Analyst Junior.

## Business Problem
Pourquoi certaines rencontres aboutissent-elles à un `YES` puis à un `MATCH` ?

## Questions métier
- Les participants disent-ils réellement ce qu’ils recherchent ?
- Quels facteurs influencent le `YES` ?
- Quels facteurs influencent le `MATCH` ?
- Quel est le poids réel de l’attractivité ?
- Quel est le rôle de l’âge ?
- Quel est le rôle des intérêts communs ?
- Quel est le rôle des variables démographiques ?
- Peut-on prédire un match ?

## Structure
- `data/`: données brutes et données préparées
- `notebooks/`: audit, EDA, statistique, ML, storytelling
- `src/`: pipeline industrialisable
- `dashboard/`: application Streamlit
- `reports/`: livrables exécutifs et figures
- `assets/`: ressources de présentation

## Méthodologie
1. Audit qualité complet
2. EDA démographique et comportementale
3. Tests statistiques et tailles d’effet
4. Feature engineering orientée métier
5. Machine learning avec validation croisée
6. Explainable AI avec SHAP
7. Analyse de fuite de données
8. Dashboard Streamlit
9. Storytelling exécutif

## Installation
```bash
pip install -r requirements.txt
```

## Lancement
```bash
python build_project.py
streamlit run dashboard/app.py
```

## Résultats attendus
- Identification des signaux du `YES` et du `MATCH`
- Comparaison des effets de l’attractivité, de l’âge et des intérêts communs
- Modèle de prédiction interprétable et version réaliste sans fuite de données
