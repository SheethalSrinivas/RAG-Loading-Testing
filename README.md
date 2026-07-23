# RAG-Loading-Testing
!pip -q uninstall -y google-adk langgraph-prebuilt \
  opentelemetry-sdk opentelemetry-api \
  opentelemetry-exporter-otlp-proto-http opentelemetry-exporter-otlp-proto-common opentelemetry-proto \
  opentelemetry-exporter-gcp-logging
# Install required libraries
!pip install -q langchain_community==0.3.27 \
              langchain==0.3.27 \
              chromadb==1.0.15 \
              pymupdf==1.26.3 \
              tiktoken==0.9.0 \
              datasets==4.0.0 \
              langchain_openai==0.3.30
# Import core libraries
import os                                                                       # Interact with the operating system (e.g., set environment variables)
import json                                                                     # Read/write JSON data
import requests  # type: ignore                                                 # Make HTTP requests (e.g., API calls); ignore type checker

# Import libraries for working with PDFs and OpenAI
from langchain.document_loaders import PyMuPDFLoader                            # Load and extract text from PDF files
from langchain_community.document_loaders import PyPDFLoader                    # Load and extract text from PDF files
from openai import OpenAI                                                       # Access OpenAI's models and services

# Import libraries for processing dataframes and text
import tiktoken                                                                 # Tokenizer used for counting and splitting text for models
import pandas as pd                                                             # Load, manipulate, and analyze tabular data

# Import LangChain components for data loading, chunking, embedding, and vector DBs
from langchain.text_splitter import RecursiveCharacterTextSplitter              # Break text into overlapping chunks for processing
from langchain.embeddings.openai import OpenAIEmbeddings                        # Create vector embeddings using OpenAI's models  # type: ignore
from langchain.vectorstores import Chroma                                       # Store and search vector embeddings using Chroma DB  # type: ignore

from datasets import Dataset                                                    # Used to structure the input (questions, answers, contexts etc.) in tabular format
from langchain_openai import ChatOpenAI                                         # This is needed since LLM is used in metric computation

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/AI/WEEK1-PROMPTENGINEERING')
out_dir = '/content/drive/MyDrive/AI/WEEK1-PROMPTENGINEERING'

# Load the JSON file and extract values
file_name = "/content/drive/MyDrive/AI/config.json"                                                       # Name of the configuration file
with open(file_name, 'r') as file:                                              # Open the config file in read mode
    config = json.load(file)                                                    # Load the JSON content as a dictionary
    OPENAI_API_KEY = config.get("OPENAI_API_KEY")                                             # Extract the API key from the config
    OPENAI_API_BASE = config.get("OPEN_API_BASE")                             # Extract the OpenAI base URL from the config

# Store API credentials in environment variables
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY                                          # Set API key as environment variable
os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE                                 # Set API base URL as environment variable

# Initialize OpenAI client
#client = OpenAI()                                                               # Create an instance of the OpenAI client

pdf_path = "/content/drive/MyDrive/AI/WEEK1-PROMPTENGINEERING/HBR_How_Apple_Is_Organized_For_Innovation.pdf"
pdf_loader = PyMuPDFLoader(pdf_path)
pdf = pdf_loader.load()
for i in range(11):
    print(f"\nPage Number : {i+1}",end="\n")
    print(pdf[i].page_content,end="\n")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='cl100k_base',
    chunk_size=256,
    chunk_overlap= 20
   )
    document_chunks = pdf_loader.load_and_split(text_splitter)
    len(document_chunks)
    print("Number of Chunks:", len(document_chunks))

# Define a function to get a response
def response(user_prompt, max_tokens=1000, temperature=0.75, top_p=0.95):   # Complete the code to set default paramenters
    # Create a chat completion using the OpenAI client
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Corrected model name from gpt-40-mini to gpt-4o-mini
        messages=[
            {"role": "user", "content": user_prompt}                            # User prompt is the input/query to respond to
        ],
        max_tokens=max_tokens,                                                  # Max number of tokens to generate in the response
        temperature=temperature,                                                # Controls randomness in output
        top_p=top_p                                                             # Controls diversity via nucleus sampling
    )
    return completion.choices[0].message.content                                # Return the text content from the model's reply
    question_1 = "Who are the authors of this article and who published this article ?"
base_prompt_response_1=response(question_1)
base_prompt_response_1

question_2 = "List down the three leadership characteristics in bulleted points and explain each one of the characteristics under 2 lines" #Complete the code to define the question #2
base_prompt_response_2=response(question_2) #Complete the code to pass the user input
base_prompt_response_2

question_3 = "Can you explain specific examples from the article where apple's approach to leadership has led to successful innovations?" #Complete the code to define the question #3
base_prompt_response_3=response(question_3)  #Complete the code to pass the user input
base_prompt_response_3
   # Initialize the OpenAI Embeddings model with API credentials
embedding_model = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,                                                     # Your OpenAI API key for authentication
    openai_api_base=OPENAI_API_BASE                                             # The OpenAI API base URL endpoint
)

# Generate embeddings (vector representations) for the first two document chunks
embedding_1 = embedding_model.embed_query(document_chunks[0].page_content)      # Embedding for chunk 0
embedding_2 = embedding_model.embed_query(document_chunks[1].page_content)      # Embedding for chunk 1

# Check and print the dimension (length) of the embedding vector
print("Dimension of the embedding vector ", len(embedding_1))                   # Typically 1536 or 2048 depending on model

# Verify if both embeddings have the same dimension (should be True)
len(embedding_1) == len(embedding_2)

# Return/display the two embedding vectors for further inspection or use
embedding_1, embedding_2

vectorstore = Chroma(
    persist_directory=out_dir,
    embedding_function=embedding_model
)


out_dir = 'Apple_db'#'/content/drive/MyDrive/AI/WEEK1-PROMPTENGINEERING'    # complete the code to define the name of the vector database

if not os.path.exists(out_dir):
  os.makedirs(out_dir)

 # Building the vector store and saving it to disk for future use
vectorstore = Chroma.from_documents(
    document_chunks,                                                            # Documents to index
    embedding_model,                                                            # Embedding model for converting text to vectors
    persist_directory=out_dir                                                   # Save vector DB files here
)

vectorstore = Chroma(
    persist_directory=out_dir,
    embedding_function=embedding_model
)
vectorstore.embeddings

vectorstore.similarity_search("Who are the authors of this article and who published this article",k=3) #Complete the code to pass a query and an appropriate k value

retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3} #Complete the code to pass an appropriate k value
)

qna_system_message = """
You are an AI Assistant Designed to answer to user question based on given context and trusted source.
When Crafting your answer use only the provided context.

You are a helpful business analyst assistant.

Important rules:
1) If the user refers to "this article" and no article text is provided, ask for the article title or relevant excerpt.
2) Do not invent facts. If you are unsure, say "I don't have enough information to answer reliably."
3) Keep answers concise and structured.
4) For Question 2, use bullet points and keep each explanation under two lines.
 #Complete the code to define the system prompt

"""
qna_user_message_template = """
###Context
Who are the authors of this article and who published this article
{context}
###Question
{question}
"""  #Complete the code to define the user message
def generate_rag_response(user_input,k=3,max_tokens=1000,temperature=0.75,top_p=0.95):
    global qna_system_message,qna_user_message_template
    # Retrieve relevant document chunks
    relevant_document_chunks = retriever.get_relevant_documents(query=user_input,k=k)
    context_list = [d.page_content for d in relevant_document_chunks]

    # Combine document chunks into a single context
    context_for_query = ". ".join(context_list)

    user_message = qna_user_message_template.replace('{context}', context_for_query)
    user_message = user_message.replace('{question}', user_input)

    # Generate the response
    try:
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": qna_system_message},
            {"role": "user", "content": user_message}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
        )
        # Extract and print the generated text from the response
        response = response.choices[0].message.content.strip()
        response
    except Exception as e:
        response = f'Sorry, I encountered the following error: \n {e}'

    return response

    question_1 = "Who are the authors of this article and who published this article?"
response_with_rag_1 = generate_rag_response(question_1)
response_with_rag_1

pdf_path = "/content/drive/MyDrive/AI/WEEK1-PROMPTENGINEERING/HBR_How_Apple_Is_Organized_For_Innovation.pdf"
pdf_loader = PyMuPDFLoader(pdf_path)
pdf = pdf_loader.load()
for i in range(11):
    print(f"\nPage Number : {i+1}",end="\n")
    print(pdf[i].page_content,end="\n")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='cl100k_base',
    chunk_size=256,
    chunk_overlap= 20
   )
    document_chunks = pdf_loader.load_and_split(text_splitter)
    len(document_chunks)
    print("Number of Chunks:", len(document_chunks))
    
    response_with_rag_1 = generate_rag_response("Who are the leading experts of this article and who published this article?")
response_with_rag_1

response_with_rag_2 = generate_rag_response("List down the three leadership characteristics in bulleted points and explain each one of the characteristics under 2 lines") #Complete the code to pass the user input
response_with_rag_2

response_with_rag_3 = generate_rag_response("Can you explain specific examples from the article where apple's approach to leadership has led to successful innovations?") #Complete the code to pass the user input
response_with_rag_3
....
