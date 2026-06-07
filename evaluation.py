# ============================================================
# evaluation.py — Évaluation du système RAG
# Métriques : Precision@k, Recall@k, temps de réponse
# Visualisation : matplotlib + plotly
# ============================================================

import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ── 1. Configuration ──────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
DOSSIER_INDEX    = "base_vectorielle"
MODELE_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
MODELE_LLM       = "llama-3.3-70b-versatile"
K                = 3   # nombre de documents récupérés

# ── 2. Jeu de test — 20 questions avec réponses attendues ──
QUESTIONS_TEST = [
    {"id": 1,  "question": "How to fix Windows Blue Screen of Death?",          "doc_attendu": "Guide_01_Windows_BSOD.txt",              "categorie": "OS"},
    {"id": 2,  "question": "How to repair corrupted registry entries?",          "doc_attendu": "Guide_02_Windows_Registry_Fix.txt",       "categorie": "OS"},
    {"id": 3,  "question": "How to fix broken packages in Ubuntu?",              "doc_attendu": "Guide_03_Linux_Package_Management.txt",   "categorie": "OS"},
    {"id": 4,  "question": "How to reset file permissions on macOS?",            "doc_attendu": "Guide_04_Mac_Permission_Reset.txt",       "categorie": "OS"},
    {"id": 5,  "question": "How to rebuild GRUB boot loader?",                   "doc_attendu": "Guide_05_OS_Boot_Loader_Fix.txt",         "categorie": "OS"},
    {"id": 6,  "question": "How to fix stuck Windows Update?",                   "doc_attendu": "Guide_06_Windows_Update_Errors.txt",      "categorie": "OS"},
    {"id": 7,  "question": "How to free disk space on Linux server?",            "doc_attendu": "Guide_07_Linux_Disk_Space_Full.txt",      "categorie": "OS"},
    {"id": 8,  "question": "How to resolve device driver conflicts?",            "doc_attendu": "Guide_08_OS_Driver_Conflict.txt",         "categorie": "OS"},
    {"id": 9,  "question": "How to reset SMC and NVRAM on Mac?",                 "doc_attendu": "Guide_09_Mac_SMC_NVRAM_Reset.txt",        "categorie": "OS"},
    {"id": 10, "question": "How to configure virtual memory paging file?",       "doc_attendu": "Guide_10_OS_Virtual_Memory_Specs.txt",    "categorie": "OS"},
    {"id": 11, "question": "How to configure router security protocols?",        "doc_attendu": "Guide_11_Router_Configuration_Basic.txt", "categorie": "Réseau"},
    {"id": 12, "question": "How to resolve IP address conflicts on network?",    "doc_attendu": "Guide_12_DHCP_IP_Conflict.txt",           "categorie": "Réseau"},
    {"id": 13, "question": "How to fix DNS resolution failures?",                "doc_attendu": "Guide_13_DNS_Resolution_Failure.txt",     "categorie": "Réseau"},
    {"id": 14, "question": "How to prevent VPN connection drops?",               "doc_attendu": "Guide_14_VPN_Connection_Drop.txt",        "categorie": "Réseau"},
    {"id": 15, "question": "How to set up firewall port forwarding?",            "doc_attendu": "Guide_15_Firewall_Port_Forwarding.txt",   "categorie": "Réseau"},
    {"id": 16, "question": "How to fix WiFi signal interference?",               "doc_attendu": "Guide_16_WiFi_Signal_Interference.txt",   "categorie": "Réseau"},
    {"id": 17, "question": "How to diagnose high network latency?",              "doc_attendu": "Guide_17_Network_Ping_Latency.txt",       "categorie": "Réseau"},
    {"id": 18, "question": "How to mount NAS shared storage?",                   "doc_attendu": "Guide_18_NAS_Storage_Mount.txt",          "categorie": "Réseau"},
    {"id": 19, "question": "How to fix proxy server connection errors?",         "doc_attendu": "Guide_19_Proxy_Server_Error.txt",         "categorie": "Réseau"},
    {"id": 20, "question": "How to fix IPv6 transition connectivity issues?",    "doc_attendu": "Guide_20_IPv6_Transition_Issues.txt",     "categorie": "Réseau"},
]


# ── 3. Chargement des outils RAG ──────────────────────────
def charger_outils():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODELE_EMBEDDING,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    index = FAISS.load_local(
        DOSSIER_INDEX,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = index.as_retriever(search_kwargs={"k": K})

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=MODELE_LLM,
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a technical IT assistant.
Use only the context below to answer. Answer in the same language as the question.

Context:
{context}

Question: {question}

Answer:"""
    )

    def formater_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chaine = (
        {"context": retriever | formater_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, chaine


# ── 4. Calcul des métriques ───────────────────────────────
def calculer_precision_recall(docs_recuperes, doc_attendu):
    """
    Precision@k : parmi les k docs récupérés, combien sont pertinents ?
    Recall@k    : parmi tous les docs pertinents, combien sont récupérés ?
    (ici 1 doc attendu par question → recall = 1 si trouvé, 0 sinon)
    """
    noms_recuperes = [
        os.path.basename(doc.metadata.get("source", ""))
        for doc in docs_recuperes
    ]

    pertinents_recuperes = sum(1 for nom in noms_recuperes if nom == doc_attendu)

    precision = pertinents_recuperes / len(docs_recuperes) if docs_recuperes else 0
    recall    = 1.0 if pertinents_recuperes > 0 else 0.0

    return round(precision, 2), round(recall, 2)


# ── 5. Lancer l'évaluation ────────────────────────────────
def lancer_evaluation():
    print("=" * 60)
    print("  ÉVALUATION DU SYSTÈME RAG — 20 questions")
    print("=" * 60)

    retriever, chaine = charger_outils()
    resultats = []

    for item in QUESTIONS_TEST:
        print(f"  [{item['id']:02d}/20] {item['question'][:55]}...")

        # Mesure du temps de réponse
        debut = time.time()
        docs_recuperes = retriever.invoke(item["question"])
        reponse        = chaine.invoke(item["question"])
        temps          = round(time.time() - debut, 2)

        # Calcul des métriques
        precision, recall = calculer_precision_recall(
            docs_recuperes, item["doc_attendu"]
        )

        resultats.append({
            "id":         item["id"],
            "question":   item["question"],
            "categorie":  item["categorie"],
            "doc_attendu": item["doc_attendu"],
            "precision":  precision,
            "recall":     recall,
            "temps_sec":  temps,
            "reponse":    reponse[:200]
        })

        print(f"         Precision@{K}={precision} | Recall@{K}={recall} | Temps={temps}s")

    return resultats


# ── 6. Sauvegarde des résultats ───────────────────────────
def sauvegarder_resultats(resultats):
    df = pd.DataFrame(resultats)
    df.to_csv("resultats_evaluation.csv", index=False, encoding="utf-8")
    print("\n  => resultats_evaluation.csv sauvegardé")
    return df


# ── 7. Affichage du résumé ────────────────────────────────
def afficher_resume(df):
    print("\n" + "=" * 60)
    print("  RÉSUMÉ GLOBAL")
    print("=" * 60)
    print(f"  Precision@{K} moyenne  : {df['precision'].mean():.2f}")
    print(f"  Recall@{K} moyenne     : {df['recall'].mean():.2f}")
    print(f"  Temps moyen / question : {df['temps_sec'].mean():.2f}s")
    print(f"  Temps total            : {df['temps_sec'].sum():.2f}s")
    print("=" * 60)

    print("\n  Résultats par catégorie :")
    print(df.groupby("categorie")[["precision", "recall", "temps_sec"]].mean().round(2))


# ── 8. Graphiques matplotlib ──────────────────────────────
def graphiques_matplotlib(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Évaluation du Système RAG — IT Support", fontsize=14, fontweight="bold")

    couleurs = ["#2196F3" if c == "OS" else "#4CAF50" for c in df["categorie"]]

    # Graphique 1 — Precision@k par question
    axes[0, 0].bar(df["id"], df["precision"], color=couleurs)
    axes[0, 0].set_title(f"Precision@{K} par question")
    axes[0, 0].set_xlabel("Question ID")
    axes[0, 0].set_ylabel("Precision")
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].axhline(y=df["precision"].mean(), color="red", linestyle="--", label=f"Moyenne = {df['precision'].mean():.2f}")
    axes[0, 0].legend()

    # Graphique 2 — Recall@k par question
    axes[0, 1].bar(df["id"], df["recall"], color=couleurs)
    axes[0, 1].set_title(f"Recall@{K} par question")
    axes[0, 1].set_xlabel("Question ID")
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].axhline(y=df["recall"].mean(), color="red", linestyle="--", label=f"Moyenne = {df['recall'].mean():.2f}")
    axes[0, 1].legend()

    # Graphique 3 — Temps de réponse
    axes[1, 0].plot(df["id"], df["temps_sec"], marker="o", color="#FF5722")
    axes[1, 0].set_title("Temps de réponse par question")
    axes[1, 0].set_xlabel("Question ID")
    axes[1, 0].set_ylabel("Secondes")
    axes[1, 0].axhline(y=df["temps_sec"].mean(), color="blue", linestyle="--", label=f"Moyenne = {df['temps_sec'].mean():.2f}s")
    axes[1, 0].legend()

    # Graphique 4 — Moyenne par catégorie
    resume = df.groupby("categorie")[["precision", "recall"]].mean()
    x = range(len(resume))
    largeur = 0.35
    axes[1, 1].bar([i - largeur/2 for i in x], resume["precision"], largeur, label="Precision", color="#2196F3")
    axes[1, 1].bar([i + largeur/2 for i in x], resume["recall"],    largeur, label="Recall",    color="#4CAF50")
    axes[1, 1].set_title("Precision & Recall par catégorie")
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels(resume.index)
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("graphiques_evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  => graphiques_evaluation.png sauvegardé")


# ── 9. Point d'entrée ─────────────────────────────────────
if __name__ == "__main__":
    resultats = lancer_evaluation()
    df        = sauvegarder_resultats(resultats)
    afficher_resume(df)
    graphiques_matplotlib(df)
    print("\n  ÉVALUATION TERMINÉE !")