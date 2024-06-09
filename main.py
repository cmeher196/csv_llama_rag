from flask import Flask, render_template, request, jsonify, g
from flask_cors import CORS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import textwrap

from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain_community import
import sys

from dsxllm import DsxLLM
import os

app = Flask(__name__)

app.config['qa'] = ""

# qa=""

chat_history = []

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


# sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.
#
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """
#
# instruction = """CONTEXT:/n/n {context}/n
#
# Question: {question}"""
# get_prompt(instruction, sys_prompt)


def before_first_request():
    # g.my_qa = qa
    DB_FAISS_PATH = "vectorstore/db_faiss"
    loader = CSVLoader(file_path="data/ormerrors.csv", encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    # print(data)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)

    print(len(text_chunks))

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    print("embedding done!!")

    docsearch = FAISS.from_documents(text_chunks, embeddings)
    print("line 34")
    docsearch.save_local(DB_FAISS_PATH)
    print("line 36")

    docsearch = FAISS.from_documents(text_chunks, embeddings)
    docsearch.save_local(DB_FAISS_PATH)

    query = "What is this generic IT execption error?"
    docs = docsearch.similarity_search(query, k=3)
    print("Result", docs)

    retriever = docsearch.as_retriever(search_kwargs={"k": 5})

    sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using 
    the context text provided. Your answers should only answer the question once and not have any text after the 
    answer is done. And provide info like for the concern error which team is responsible as "Team":"Team Name" and poc assign to be as "POC":"POC Name" and if 
    any resolution has to be taken care off then as "RCA":"Provided RCA", also trim extra information which is not required for the answer.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """

    instruction = """CONTEXT:/n/n {context}/n

    Question: {question}"""
    get_prompt(instruction, sys_prompt)

    prompt_template = get_prompt(instruction, sys_prompt)

    llama_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": llama_prompt}

    # specify the model you want to use
    MODELNAME = "llama-2-13b-chat"

    llm = DsxLLM(model_name=MODELNAME,
                 api_keys=["OWZkN2Y1OWMtZmJiNS00MjVhLWIxZGYtNzczNTNlYmRjYjc5"])  # your DSX Open Source LLM api key(s)

    app.config['qa'] = RetrievalQA.from_chain_type(llm=llm,
                                                   chain_type="stuff",
                                                   retriever=retriever,
                                                   chain_type_kwargs=chain_type_kwargs,
                                                   return_source_documents=True)

    # app.config['qa'] ="data changesss"
    # print("Result", docs)
    # g.my_qa ="updated in context"


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def setup():
    print("Executing setup.")


# Connect the setup function to the got_first_request signal
# app._got_first_request.connect(setup, app)


# got_first_request(setup,app)
CORS(app)

with app.app_context():
    before_first_request()


@app.route("/")
def index():
    print("my data " + app.config['qa'])
    # print("data from qa global "+ g.my_qa)
    return "hello chat gpt"


@app.route("/api/message", methods=['POST'])
def postMessage():
    qa = app.config['qa']
    data = request.get_json()

    # query = "What is the value of  GDP per capita of Finland provided in the data?"
    query = data.get("message")
    print(query)
    if query == 'exit':
        print('Exiting')
        sys.exit()

    # result = qa({"question": query, "chat_history": chat_history})
    result = qa(query)
    print("Response: ", query)
    # return jsonify({'response': query})
    # print("Response: ", result['answer'])
    # return jsonify({'response': result['answer'], 'history': chat_history})
    # print(result.result)
    outputResponse = wrap_text_preserve_newlines(result['result'])
    print(outputResponse)

    return jsonify({'response': outputResponse})
    # data = request.get_json()
    # print(data)
    # return jsonify(data)


if __name__ == '__main__':
    app.run()
