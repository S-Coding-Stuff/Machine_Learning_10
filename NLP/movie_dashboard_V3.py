from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document

import numpy as np, pandas as pd
from dotenv import load_dotenv
import gradio as gr
from rapidfuzz import process, fuzz

load_dotenv()

movies = pd.read_csv('movies_semantic.csv')
movies['Plot'] = movies['Plot'].str.slice(0, 1000)

documents = [Document(page_content=row['Plot'],
                      metadata={
                          'movie_id': row['movie_id'],
                          'Title': row['Title'],
                          'Release Year': row['Release Year'],
                          'Genre': row['Genre']
                      }
                      ) for _, row in movies.iterrows()]

# Simply loading embeddings again
db_movies = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='sem_movies')


def retrieve_semantic_recommendations(query,
                                      top_k=10,):
    """Matching the title of the movie through fuzzy match to then find movies within the same time period
    that are similar. Otherwise, for any move 'idea,' perform a simple semantic similarity search."""

    initial_top_k = 100
    titles = [doc.metadata['Title'] for doc in documents]
    match, score, index = process.extractOne(query, titles, scorer=fuzz.token_sort_ratio)
    if score >= 80:
        base_doc = documents[index]
        targeted_year = base_doc.metadata['Release Year']

        recs = db_movies.similarity_search(query, k=initial_top_k)

        rank = []

        for doc, similarity_score in recs:
            mov_id = doc.metadata['movie_id']
            if mov_id == base_doc.metadata['movie_id']:
                continue
            year = doc.metadata['Release Year']
            year_dist = abs(int(year) - int(targeted_year)) # Very simple distance calculation

            complete_score = similarity_score - 0.1 * year_dist
            rank.append((complete_score, doc))

            if len(rank) == top_k:
                break

        rank =  sorted(rank, key=lambda x: x[0], reverse=True)[:top_k]
        selection = [doc.metadata['movie_id'] for _, doc in rank]


    else:
        recs = db_movies.similarity_search(query, k=initial_top_k)
        selection = [rec.metadata['movie_id'] for rec in recs]

    movie_recs = movies.set_index('movie_id').loc[selection].reset_index()

    results = ""
    for _, row in movie_recs.iterrows():
        results += f"**{row['Title']}** ({row['Release Year']}) \n{row['Plot']}\n\n"
    return results


with gr.Blocks() as demo:
    gr.Markdown('# Movie Recommendations System')
    query = gr.Textbox(label='Movie title, theme or idea')

    gr.Markdown('## Recommendations')
    output = gr.Markdown()
    query.submit(fn=retrieve_semantic_recommendations, inputs=query, outputs=output)

if __name__ == '__main__':
    demo.launch()
