import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── 1. Variables d'environnement ──────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── 2. Chemins et modèles ─────────────────────────────────
DOSSIER_DOCS     = "it_docs"
DOSSIER_INDEX    = "base_vectorielle"
MODELE_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
MODELE_LLM       = "llama-3.3-70b-versatile"

# ── 3. Titre ──────────────────────────────────────────────
st.title("Assistant IT — Système RAG")
st.caption("Questions techniques : OS, Réseaux, Sécurité")

# ── 4. Chargement de l'index FAISS ────────────────────────
@st.cache_resource
def charger_index():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODELE_EMBEDDING,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(DOSSIER_INDEX):
        return FAISS.load_local(
            DOSSIER_INDEX,
            embeddings,
            allow_dangerous_deserialization=True
        )

    loader = DirectoryLoader(
        DOSSIER_DOCS,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    index = FAISS.from_documents(chunks, embeddings)
    os.makedirs(DOSSIER_INDEX, exist_ok=True)
    index.save_local(DOSSIER_INDEX)
    return index

# ── 5. Construction de la chaîne RAG ──────────────────────
@st.cache_resource
def creer_chaine():
    index = charger_index()

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=MODELE_LLM,
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a technical IT assistant.
Use only the context below to answer. Answer in the same language as the question.



Contexte:
{context}

Question: {question}

Answer:"""
    )

    retriever = index.as_retriever(search_kwargs={"k": 1})

    def formater_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chaine = (
        {"context": retriever | formater_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chaine, retriever

# ── 6. Initialisation de l'historique ─────────────────────
if "historique" not in st.session_state:
    st.session_state.historique = []

# ── 7. Affichage de l'historique ──────────────────────────
for message in st.session_state.historique:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ── 8. Zone de saisie ─────────────────────────────────────
question = st.chat_input("Posez votre question...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.historique.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            chaine, retriever = creer_chaine()

            # Récupérer les sources
            sources = retriever.invoke(question)

            # Générer la réponse
            reponse = chaine.invoke(question)

            st.write(reponse)

            # Afficher les sources
            with st.expander("Sources utilisées"):
                for i, doc in enumerate(sources, 1):
                    st.markdown(f"**Source {i}** — `{doc.metadata.get('source', 'inconnu')}`")
                    st.caption(doc.page_content[:200] + "...")

    st.session_state.historique.append({"role": "assistant", "content": reponse})