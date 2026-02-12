import streamlit as st
from rag import get_index

st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="centered")

st.title("🧠 RAG Chatbot (Vector DB + LLM)")
st.caption("Responde usando tus documentos en /data y muestra fuentes.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Carga índice
index = get_index()
query_engine = index.as_query_engine(similarity_top_k=5)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Pregunta algo sobre tus documentos...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en tus documentos..."):
            response = query_engine.query(prompt)

        st.markdown(str(response))

        # Fuentes (muy importante en un RAG)
        with st.expander("📌 Fuentes / chunks usados"):
            if hasattr(response, "source_nodes") and response.source_nodes:
                for i, sn in enumerate(response.source_nodes, start=1):
                    meta = sn.node.metadata or {}
                    file_name = meta.get("file_name", meta.get("filename", "desconocido"))
                    score = getattr(sn, "score", None)
                    st.write(f"**{i}.** `{file_name}`" + (f" | score: {score:.3f}" if score else ""))
                    st.write(sn.node.get_text()[:700] + ("..." if len(sn.node.get_text()) > 700 else ""))
                    st.divider()
            else:
                st.write("No se encontraron fuentes.")

    st.session_state.messages.append({"role": "assistant", "content": str(response)})
