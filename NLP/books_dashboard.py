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
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    'cover-not-found.jpg',
    books['large_thumbnail'])

raw_documents = TextLoader('tagged_description.txt').load()
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

def retrieve_semantic_recommendations(query,
                                      category,
                                      tone,
                                      initial_top_k=50,
                                      final_top_k=16):

    recommendations = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(recommendation.page_content.strip('"').split()[0]) for recommendation in recommendations]
    book_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != 'All':
        book_recs = book_recs[book_recs['categories'] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == 'Happy':
        book_recs.sort_values(by='happiness', ascending=False, inplace=True)
    elif tone == 'Surprising':
        book_recs.sort_values(by='surprise', ascending=False, inplace=True)
    elif tone == 'Angry':
        book_recs.sort_values(by='anger', ascending=False, inplace=True)
    elif tone == 'Suspenseful':
        book_recs.sort_values(by='fear', ascending=False, inplace=True)
    elif tone == 'Sad':
        book_recs.sort_values(by='sadness', ascending=False, inplace=True)

    return book_recs

def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        desc = row['description']
        truncated_desc_split = desc.split()
        truncated_desc = " ".join(truncated_desc_split[:30]) + '...' # Will show a small amount of the description

        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            sentence = f'{authors_split[0]} and {authors_split[1]}'
        elif len(authors_split) > 2:
            sentence = f'{', '.join(authors_split[:-1])} and {authors_split[-1]}'
        else:
            sentence = row['authors']

        caption = f'{row['title']} by {sentence}: {truncated_desc}'
        results.append((row['large_thumbnail'], caption))

    return results

categories = ['All'] + sorted(books['categories'].unique())
tones = ['All'] + ['Happy', 'Suspenseful', 'Angry', 'Surprising', 'Sad']

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Book Recommender")

    with gr.Row():
        query = gr.Textbox(label='Please enter a description of a book:', placeholder='e.g. a story about love')
        category_dropdown = gr.Dropdown(choices=categories, label='Select a Category:', value='All')
        tone_dropdown = gr.Dropdown(choices=tones, label='Select a Tone:', value='All')
        submit_button = gr.Button('Find Recommendations')

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label='Recommended Books', columns=8, rows=2)

    submit_button.click(fn=recommend_books, inputs=[query, category_dropdown, tone_dropdown], outputs=output)


if __name__ == '__main__':
    dashboard.launch()