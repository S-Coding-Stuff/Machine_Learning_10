from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document

import numpy as np, pandas as pd
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

movies = pd.read_csv('movies_semantic.csv')

documents = []
with open('movies_semantic.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line:
            documents.append(Document(page_content=line))

db_movies = Chroma.from_documents(documents, OpenAIEmbeddings())

def retrieve_semantic_recommendations(query,
                                      initial_top_k=50,
                                      final_top_k=10,):

    recs = db_movies.similarity_search(query, k=initial_top_k)
    movies_list = [
        int(rec.page_content.strip('"').split(' ', 1)[0]) for rec in recs
    ]

    movie_recs = movies.set_index('movie_id').loc[movies_list].reset_index()

    results = ""
    for _, row in movie_recs.iterrows():
        results += f"**{row['Title']}** ({row['Release Year']}) \n{row['Plot']}\n\n"
    return results

with gr.Blocks() as demo:
    gr.Markdown('# Movie Recommendations System')
    query = gr.Textbox(label='Movie title, theme or idea')
    output = gr.Markdown()

    gr.Markdown('## Recommendations')
    query.submit(fn=retrieve_semantic_recommendations, inputs=query, outputs=output)

if __name__ == '__main__':
    demo.launch()