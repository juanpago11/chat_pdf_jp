import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ======================
# CONFIGURACIÓN DE PÁGINA
# ======================
st.set_page_config(
    page_title="RAG PDF Analyzer",
    page_icon="💬",
    layout="wide"
)

# ======================
# ESTILOS PERSONALIZADOS
# ======================
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #146AEF;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput > div > div > input {
        background-color: #1E222A;
        color: white;
        border-radius: 10px;
        border: 1px solid #146AEF;
    }
    .stTextArea textarea {
        background-color: #1E222A;
        color: white;
        border-radius: 10px;
        border: 1px solid #146AEF;
    }
    .stButton button {
        background-color: #146AEF;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #0d4fc2;
    }
    .block-container {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
col1, col2 = st.columns([1, 3])

with col1:
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, width=120)
    except:
        pass

with col2:
    st.title('Generación Aumentada por Recuperación (RAG) 💬')
    st.caption(f"Python {platform.python_version()}")

st.markdown("---")

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.markdown("## 🧠 Asistente PDF")
    st.markdown("Este agente analiza documentos PDF usando IA.")
    st.markdown("---")
    st.markdown("### ⚙️ Configuración")

# ======================
# API KEY
# ======================
ke = st.text_input('🔑 Ingresa tu Clave de OpenAI', type="password")

if ke:
    os.environ['OPENAI_API_KEY'] = ke
    st.success("API Key cargada correctamente")
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI")

# ======================
# UPLOAD PDF
# ======================
st.markdown("## 📄 Cargar Documento")
pdf = st.file_uploader("Arrastra o selecciona tu PDF", type="pdf")

# ======================
# PROCESAMIENTO
# ======================
if pdf is not None and ke:
    try:
        with st.spinner("Procesando documento..."):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

        st.info(f"📊 {len(text)} caracteres extraídos")

        # Split text
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"🧩 {len(chunks)} fragmentos creados")

        # Embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.markdown("---")
        st.markdown("## ❓ Haz tu pregunta")

        user_question = st.text_area("", placeholder="Ej: ¿Cuál es la idea principal del documento?")

        if user_question:
            with st.spinner("Pensando..."):
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI(
                    temperature=0,
                    model_name="gpt-4o-mini-2024-07-18"
                )

                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)

            st.markdown("---")
            st.markdown("### 🧠 Respuesta")
            st.success(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Necesitas ingresar la API Key primero")

else:
    st.info("👆 Sube un PDF para comenzar")
