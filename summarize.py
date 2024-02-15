from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
from warnings import simplefilter
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import numpy as np
import os
import pandas as pd
import ast

class BookSummarizer:
    def __init__(self, pdf_path, num_clusters=10):
        self.pdf_path = pdf_path
        self.num_clusters = num_clusters
        self.book_page_number = None
        load_dotenv()

    def load_book(self):
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        self.book_page_number = len(pages)
        
        text = ""
        for page in pages:
            text += page.page_content
        text = text.replace('\t', ' ')

        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=5000, chunk_overlap=2500)
        docs = text_splitter.create_documents([text])
        return docs

    def perform_clustering(self, docs):
        embeddings = OpenAIEmbeddings()
        vectors = embeddings.embed_documents([x.page_content for x in docs])

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42).fit(vectors)

        return kmeans, vectors

    def plot_clusters(self, reduced_data, labels):
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Book Embeddings Clustered')
        plt.show()

    def find_closest_embeddings(self, vectors, kmeans):
        closest_indices = []

        for i in range(self.num_clusters):
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)

        return closest_indices

    def get_summaries(self, docs, closest_indices):
        llm3 = ChatOpenAI(temperature=0, max_tokens=1000, model='gpt-4')

        map_prompt = """
        You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
        Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
        Your response should be at least three paragraphs and fully encompass what was said in the passage.

        ```{text}```
        FULL SUMMARY:
        """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        map_chain = load_summarize_chain(llm=llm3, chain_type="stuff", prompt=map_prompt_template)

        selected_docs = [docs[doc] for doc in closest_indices]
        summary_list = []

        for i, doc in enumerate(selected_docs):
            chunk_summary = map_chain.run([doc])
            summary_list.append(chunk_summary)
            print(f"Summary #{i} (chunk #{closest_indices[i]}) - Preview: {chunk_summary[:250]} \n")

        summaries = "\n".join(summary_list)
        return Document(page_content=summaries)

    def combine_summaries(self, summaries):
        llm4 = ChatOpenAI(temperature=0, max_tokens=3000, model='gpt-4')

        combine_prompt = """
        You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
        Your goal is to give a verbose summary of what happened in the story.
        The reader should be able to grasp what happened in the book.

        ```{text}```
        VERBOSE SUMMARY:
        """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
        reduce_chain = load_summarize_chain(llm=llm4, chain_type="stuff", prompt=combine_prompt_template, verbose=False)

        output = reduce_chain.run([summaries])
        return output
    
    def summary_embeddings(self, output):
        embeddings = OpenAIEmbeddings()
        vectors = embeddings.embed_documents([output])
        return vectors

    def run_pipeline(self):
        docs = self.load_book()
        kmeans, vectors = self.perform_clustering(docs)
        simplefilter(action='ignore', category=FutureWarning)
        closest_indices = self.find_closest_embeddings(vectors, kmeans)
        summaries = self.get_summaries(docs, closest_indices)
        output = self.combine_summaries(summaries)
        
        return output
    
    def document_classification(self, summary):
        
        template =''''
        You are given the {summary} of a book. Your task is to classify the book into one of the following genres:
        [report, letter, handbook, specification document, poem, journal, science book, course outline].

        If it is not present in the array suggest any relevant genre.

        GENRE:
        '''

        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(temperature=0, model_name='gpt-4')

        chain = prompt | llm
        doc_class = chain.invoke({'summary': summary})
        return doc_class.content
    
def process_single_file(file_path):
    summarizer = BookSummarizer(pdf_path=file_path, num_clusters=3)
    output = summarizer.run_pipeline()
    vector = summarizer.summary_embeddings(output)[0]
    doc_class = summarizer.document_classification(output)

    return {
        'book_name': os.path.basename(file_path),
        'vector': vector,
        'page_number': summarizer.book_page_number,
        'summary': output,
        'doc_class': doc_class
    }
    
if __name__ == "__main__":
    
    filename = "docs_copy/ros.pdf"
    data = "books.csv"
    number_of_clusters = 3
    reduced_data_tsne = None
    kmeans = None

    if not os.path.exists(data):
        columns = ['vector', 'x', 'y', 'book_name', 'page_number', 'summary', 'labels', 'doc_class']
        df = pd.DataFrame(columns=columns)
        df.to_csv(data, index=False)
    
    df = pd.read_csv(data)
    file_data = process_single_file(filename)

    vector = file_data['vector']
    book_name = file_data['book_name']
    page_number = file_data['page_number']
    summary = file_data['summary']
    doc_class = file_data['doc_class']

    new_row = {'vector': vector, 'book_name': book_name,
               'page_number': page_number, 'summary': summary,
               'doc_class': doc_class,
               }
    
    df = df.append(new_row, ignore_index=True)
    df.to_csv(data, index=False)

    df = pd.read_csv(data)
    
    if len(df) > 1:
        df['vector'] = df['vector'].apply(ast.literal_eval)
        all_books_vectors = np.array(df['vector'].tolist())
        
    else:
        all_books_vectors = vector
    
    if len(df) > 4:
        
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=42).fit(all_books_vectors)
        closest_indices = []

        for i in range(number_of_clusters):
            distances = np.linalg.norm(all_books_vectors - kmeans.cluster_centers_[i], axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)
       
        tsne = TSNE(n_components=2, random_state=42,perplexity=4)
        reduced_data_tsne = tsne.fit_transform(np.array(all_books_vectors))
        
    
    if reduced_data_tsne is not None:
        x = reduced_data_tsne[:, 0]
        y = reduced_data_tsne[:, 1]
        labels = kmeans.labels_

        df['x'] = x
        df['y'] = y
        df['labels'] = labels

    df.to_csv(data, index=False)