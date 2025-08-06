import pandas as pd, numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv('books_with_emotions.csv')
books['large_thumbnail'] = books['thumbnail'] + '&fife=w800'
books['large_thumbnail'] = np.where(books['large_thumbnail'].isna(), 'cover-not-found.jpg', books['large_thumbnail'])

raw_documents = TextLoader('tagged_description.txt').load()
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())