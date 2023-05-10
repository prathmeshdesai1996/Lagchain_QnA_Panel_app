import os 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import tempfile


import streamlit as st
from streamlit import file_uploader

def qa(file, query, chain_type, k):
  #load doc
  loader = PyPDFLoader(file)
  documents = loader.load()

  #split doc in chunks
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.split_documents(documents)

  #select embeddings we want to use
  embeddings = OpenAIEmbeddings()

  #create vectorstore to use as the index
  db = Chroma.from_documents(texts,embeddings)

  #expose this index to a retriever interface
  retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

  #create a chain to answer questions
  qa = RetrievalQA.from_chain_type(
      llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
  result = qa({"query": query})
  print(result['result'])
  return result

def qa_result(file, query, chain_type, k):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())

        result = qa(temp_file.name, query, chain_type, k)

        st.markdown(f"**Result:** {result['result']}")

        st.write("Relevant source text:")
        for doc in result["source_documents"]:
            st.write('--------------------------------------------------------------')
            st.write(doc.page_content)

def main():
    st.markdown("""
        ## 🤔 Question Answering with your PDF file
        
        1. Upload a PDF file.
        2. Enter your OpenAI API key.
        3. Type a question and click "Run".
    """)

    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    openaikey = st.text_input("Enter your OpenAI API key:")
    query = st.text_input("Enter your question:")
    chain_type = st.radio('Chain type', ['stuff', 'map_reduce', "refine", "map_rerank"])
    k = st.slider("Number of relevant chunks", 1, 5, 2)
    run_button = st.button("Run")

    if run_button:
        os.environ["OPENAI_API_KEY"] = openaikey
        qa_result(file, query, chain_type, k)

if __name__ == '__main__':
    main()