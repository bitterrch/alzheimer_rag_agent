import streamlit as st 
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import time

st.set_page_config(page_title="Alzheimer's Research Assistant", layout="wide")

@st.cache_resource
def load_knowledge_base():
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    vector_store = Chroma(
        collection_name='alzheimer_data',
        persist_directory='./chroma_langchain_db',
        embedding_function=embeddings
    )
    return vector_store

def main():
    st.title("Alzheimer's Target Discovery Agent")
    st.markdown("RAG-for rapid search for information on \
                potential drug development targets based on \
                a database of scientific articles")
    with st.sidebar:
        st.header("Settings")
        st.info("Model: Local llama3.2")
        st.info("Database: ChromaDB")

        k_retrieval = st.slider("Documents to retrieve (K)", 1, 10, 5)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # показываем источники если есть
            if "sources" in message:
                with st.expander("Sources & Citations"):
                    for idx, source in enumerate(message["sources"]):
                        st.markdown(f"**{idx+1}. {source['source']}**")
                        st.caption(source['content'][:300] + "...")

    if question := st.chat_input("Ask a question..."):
        # показываем вопрос пользователя
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # генерация ответа
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Searching scientific literature..."):
                try:
                    vector_store = load_knowledge_base()
                    model = OllamaLLM(model='llama3.2')

                    retriever = vector_store.as_retriever(
                        search_type="mmr", 
                        search_kwargs={'k': k_retrieval, 'fetch_k': 20}
                    )
                    
                    template = """You are a specialized AI assistant for Alzheimer's disease research.
                        Use the following pieces of retrieved context to answer the question.
                                
                        Rules:
                        1. Answer strictly based on the provided context.
                        2. If the answer is not in the context, say "I don't have enough information in the provided documents."
                        3. ALWAYS cite the source article names (from metadata) for your statements. Format: [Source: filename.pdf].
                        4. Focus on drug targets, mechanisms of action, and therapeutic potential.
                                
                        Context:
                        {context}
                                
                        Question:
                        {question}
                                
                        Answer:
                    """
                    
                    prompt = ChatPromptTemplate.from_template(template)
                    chain = prompt | model

                    docs = retriever.invoke(question)
                    context_text = "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in docs])
                    
                    response = chain.invoke({"context": context_text, "question": question})
                    
                    # Эффект печатания
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                    # Сохраняем источники для истории
                    sources_data = [{"source": doc.metadata.get('source', 'Unknown'), "content": doc.page_content} for doc in docs]
                    
                    # Добавляем в историю
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources_data
                    })
                    
                    # Отображаем источники сразу под ответом
                    with st.expander("View Retrieved Sources"):
                        for idx, doc in enumerate(docs):
                            st.markdown(f"Source {idx+1}: `{doc.metadata.get('source')}`")
                            st.caption(doc.page_content[:400] + "...")

                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()