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

documents = [
    Document(
        page_content=row['Plot'],
        metadata={
            'movie_id': row['movie_id'],
            'Title': row['Title'],
            'Release Year': row['Release Year'],
            'Genre': row['Genre']
        }
    )
    for _, row in movies.iterrows()
]

# Simply loading embeddings each time rather than recreating each time
db_movies = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='sem_movies')


# Fuzzy match above 90% match threshold. Used to check titles are equal to query text
def match_title(query, documents, threshold=85):
    titles = [doc.metadata['Title'] for doc in documents]
    match, score, index = process.extractOne(query, titles, scorer=fuzz.token_sort_ratio)

    if score >= threshold:
        return documents[index].metadata, score
    return None, None


def retrieve_semantic_recommendations(query,
                                      top_k=10):
    match, score = match_title(query, documents)
    results = []
    seen = set()

    if match:
        if match['movie_id'] not in seen:
            results.append(match)
            seen.add(match['movie_id'])

        recs = db_movies.similarity_search(match['Title'], k=top_k + 1)

        for doc in recs:
            if doc.metadata['movie_id'] not in seen:
                results.append(doc.metadata)
                seen.add(doc.metadata['movie_id'])

    else:
        recs = db_movies.similarity_search(query, k=top_k)
        for doc in recs:
            if doc.metadata['movie_id'] not in seen:
                results.append(doc.metadata)
                seen.add(doc.metadata['movie_id'])

    string_result = ""
    for result in results:
        string_result += f"**{result['Title']}** ({result['Release Year']}) \n\nGenres: {result['Genre']}\n\n"

    return string_result


with gr.Blocks() as demo:
    gr.Markdown('# Movie Recommendations System')
    query = gr.Textbox(label='Movie title, theme or idea')

    gr.Markdown('## Recommendations')
    output = gr.Markdown()
    query.submit(fn=retrieve_semantic_recommendations, inputs=query, outputs=output)

if __name__ == '__main__':
    demo.launch()
