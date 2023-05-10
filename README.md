# LangChain Question Answering
This project is a Python-based application for Question Answering using PDF documents. It uses the LangChain library which provides an easy-to-use interface for creating Question Answering systems.

[Streamlit Webapp](https://prathmeshdesai1996-lagchain-qna-panel-app-streamlit-nlnp7d.streamlit.app/)
[HuggingFace Webapp](https://huggingface.co/spaces/pd96/QnA_with_custom_pdf_langchain)

## Installation
To use this application, you need to install the following dependencies:
```
pip install -r requirements.txt
```

## Usage
To use the application, you need to upload a PDF file and enter your OpenAI API Key. You can then type a question and click "Run" to get the answer.

## Advanced Settings
You can also adjust the following advanced settings:

- Chain type: Choose the type of chain to use for answering questions. Available options are "stuff", "map_reduce", "refine", and "map_rerank".
- Number of relevant chunks: Choose the number of relevant chunks to consider when answering the question.

## Code Overview
The main code consists of the following sections:

1. Importing libraries and modules.
2. Setting up the Panel widgets for the user interface.
3. Defining the qa() function which loads a PDF, splits it into chunks, creates an index, and uses it to answer a question.
4. Defining the qa_result() function which takes the input from the user interface, calls the qa() function, and displays the output.
5. Setting up the user interface using the Panel library.

## Disclaimer
Note that using the OpenAI API for Question Answering can incur costs. Please set up billing with OpenAI before using this application.
