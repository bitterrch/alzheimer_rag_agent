# Alzheimer's Target Discovery Agent 

RAG-система для анализа научных статей по болезни Альцгеймера с использованием локальных LLM

## Предварительные требования
1. Установленный Python 3.10
2. Установленная и запущенная Ollama
3. Скачанные модели Ollama:
   ```bash
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   
## Инструкция по запуску
1. Склонируйте репозиторий и в терминале перейдите в папку *alzheimer_rag_agent*
    ```bash
    git clone https://github.com/bitterrch/alzheimer_rag_agent.git
    cd alzheimer_rag_agent
2. Создайте и активируйте виртуальное окружение
    ```bash
    python -m venv venv
    venv\Scripts\activate
3. Установите зависимости 
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
4. Поместите ваши PDF-файлы в папку `data/`
5. Создайте ядро для Jupyter Notebook и запустите его `
    ```bash
    python -m ipykernel install --user --name alzheimer_rag_agent
    jupyter notebook
6. В Jupyter-е откройте `jupyter/01_data_prep_and_eval.ipynb`.
7. Выберите ядро `alzheimer_rag_agent` и выполните весь код. Это создаст векторную базу данных в папке `chroma_langchain_db/`
8. Для запуска интерфейса в терминале с активным `venv` выполните команду
    ```bash
    streamlit run app.py
