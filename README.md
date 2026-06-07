# Assistant IT — Système RAG

Système de **Retrieval-Augmented Generation (RAG)** appliqué au domaine des **technologies de l'information**.  
Il permet de poser des questions techniques en français, anglais ou arabe et obtenir des réponses précises à partir d'une base de connaissances locale.

---

## Domaine

Support technique IT : Systèmes d'exploitation, Réseaux & Communications, Sécurité informatique.

---

## Architecture

```
Question utilisateur
       ↓
Embedding (all-MiniLM-L6-v2)
       ↓
Recherche FAISS (Top-3 chunks)
       ↓
Prompt Template + Contexte
       ↓
LLM Groq (llama-3.3-70b-versatile)
       ↓
Réponse + Sources
```

---

## Structure du projet

```
rag_project/
├── it_docs/              ← 30 documents TXT (corpus)
├── base_vectorielle/     ← Index FAISS (généré automatiquement)
├── chatbot.py            ← Interface Streamlit + pipeline RAG
├── evaluation.py         ← Évaluation : Precision@k, Recall@k
├── requirements.txt      ← Dépendances Python
├── README.md
└── .env                  ← Clé API Groq (non versionnée)
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/Sidi-med9/rag_project.git
cd rag_project

# 2. Créer et activer l'environnement virtuel
python -m venv rag_project-env
rag_project-env\Scripts\activate   # Windows
source rag_project-env/bin/activate # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API
echo GROQ_API_KEY=votre_cle_ici > .env
```

---

## Utilisation

### Lancer le chatbot
```bash
streamlit run chatbot.py
```
Ouvre automatiquement `http://localhost:8501`

### Lancer l'évaluation
```bash
python evaluation.py
```
Génère `resultats_evaluation.csv` et `graphiques_evaluation.png`

---

## Corpus

| Catégorie | Fichiers | Exemples |
|-----------|----------|---------|
| Systèmes d'exploitation | Guide_01 à Guide_10 | BSOD, GRUB, Linux, macOS |
| Réseaux & Communications | Guide_11 à Guide_20 | DNS, VPN, WiFi, DHCP |
| Sécurité & Comptes | Guide_21 à Guide_30 | MFA, Ransomware, SSH, BitLocker |

---

## Résultats d'évaluation (20 questions)

| Métrique | Valeur |
|----------|--------|
| Precision@3 | 0.65 |
| Recall@3 | **1.00** |
| Temps moyen / question | 0.47s |

---

## Technologies utilisées

| Composant | Technologie |
|-----------|-------------|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Base vectorielle | FAISS |
| LLM | Groq — llama-3.3-70b-versatile |
| Framework RAG | LangChain 0.3 |
| Interface | Streamlit |

---

## Auteur

**Sidi Mohamed Mohamed Lemine** — Master Informatique, FST Université de Nouakchott  
Projet RAG — Juin 2026
