"""
Created on Thu Feb 16 19:05:51 2023

@author: Shane Tomasello and Charlie Evert
@note: This script takes pdf's from a directory, scrappes pdf's, creates emmbeddings using OpenAI's
    embedding engine then buts the embbedings and data with in a mivus vector database that can be
    through a docker container locally. It then creates a web Gradio chatbot that will last around
    72 hours before it gets disconnected. This project was to test document scrapping and an open
    source vector database to see how well it will hold up and better understand how to query said
    database for information.
    You can use the question_answer.csv if you dont want to do document scrappinf and just want to skip
    to embeddings.
"""
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee.dc2 import pipe, ops, DataCollection
import pandas as pd
import numpy as np
import gradio
import PyPDF2
import openai
import os
import pandas as pd
import re
import tiktoken
import time
from openai.embeddings_utils import get_embedding


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

t = time.localtime()
start_time = time.time()

openai.api_key = '<OpenAI Key>' 
COMPLETIONS_MODEL = "<AI model you wish to use>"

pdf_dir = '<Your directory of pdfs>'

embeddings = []

# Need to iterate through doc types, use the label below to differentiate which one is which
document_type = '<Doc Type>'

for filename in os.listdir(pdf_dir):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, filename)
        with open(pdf_path, 'rb') as pdf_file:
            
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            #so we can get 3 layers to each embedding for best performance
            prev_paragraph = ''
            prev_prev_paragraph = ''
            
            for page_num in range(len(pdf_reader.pages)):
                
                text = pdf_reader.pages[page_num].extract_text()
                text = re.sub(r'^\s+|\s+?$', '', text) #remove leading/trailing spaces
                text = re.sub("\n\n", "randomwordforlookingup", text) #replace the \n\n
                text = re.sub("\n \n", "randomwordforlookingup", text) #replace the \n \n
                text = re.sub("\n", " ", text) #replace the \n that are left over
                text = re.sub(' +', ' ', text) #remove the extra spaces
                text = text.replace(".", "").replace(".", ".") #remove the extra periods
                text = text.replace(" s ", "") #remove the extra s that are scattered about
                text = text.replace(" d ", "") #remove the extra d that are scattered about
                
                paragraphs = text.split("randomwordforlookingup")
                
                paragraph_number = 1
                
                for paragraph in paragraphs:
                    
                    if paragraph != "":
                        
                        published_paragraph = paragraph
                        
                        #add chunks of previous context if the paragraph is too small
                        if len(published_paragraph) < 100:
                            published_paragraph = prev_paragraph + " " + published_paragraph
                            
                            if len(published_paragraph) < 200:
                                published_paragraph = prev_prev_paragraph + " " + published_paragraph
                            
                        embedding = {
                             'doctype': document_type,
                             'title': filename,
                             'page': page_num + 1, # no start on page 0
                             'paragraph': paragraph_number,
                             'tokens': num_tokens_from_string(published_paragraph, "cl100k_base"),
                             'text': published_paragraph,
                             'embedding': get_embedding(published_paragraph, engine='<Embbeding engine here>')                             
                        }
                        
                        embeddings.append(embedding)                     
                        
                        #so we can get 3 layers to each embedding for best performance if we have a small paragraph
                        prev_prev_paragraph = prev_paragraph
                        prev_paragraph = published_paragraph
                        
                        paragraph_number += 1
                        
#Convert list of dicts to a df
df = pd.DataFrame(embeddings)
embedding_vector = df["embedding"]

#Connects to running milvus container. You should be running your docker at this point.
connections.connect(
    alias="default",
    host='localhost',
    port='19530'
)

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        
    fields = [
    FieldSchema(name='id', dtype=DataType.INT64, description='ids', is_primary=True, auto_id=False),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length= 2000),
    FieldSchema(name="text", dtype=DataType.VARCHAR,max_length= 20000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim)
    ]
    
    schema = CollectionSchema(fields=fields, description='Q&A search')
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":1536}
    }
    
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

collection = create_milvus_collection('<collection _name>', 1536)

insert_p = (
    pipe.input('id','title','text')
    .map(
        'id',  # Input columns
        'vec',  # Output columns
        lambda id: [embedding_vector[id]]  # Load pre-existing embedding
    )
    .map(
        ('id','title', 'text', 'vec'),
        (), 
        ops.ann_insert.milvus_client(
            host='localhost', 
            port='19530', 
            collection_name='Case_Study'
        )
    )
    .output()
)

for index, row in df.iterrows():
    insert_p(index, row.title, row.text)

collection.load()

def chat(message, history):
    history = history or []
    question_embedding = get_embedding(message, engine='<insert text engine>')
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 5}  
    results = collection.search(
        data=[question_embedding],
        anns_field="embedding",
        param=search_params,
        limit=3, 
    	expr=None,
    	consistency_level="Strong"
        )
    
    res = []
    
    for result_key in results[0].ids:
        res.append(collection.query(
          expr = "id in [{}]" .format(result_key),
          offset = 0,
          limit = 10, 
          output_fields = ["title", "text"],
          consistency_level="Strong"
        ))
        
    context = []
    for row in res:
        sorted_res = sorted(row, key=lambda k: k['id'])
        context_text = sorted_res[0]['text']
      
        chunks = [context_text[i:i+1500] for i in range(0, len(context_text), 1500)] #chunk the text into blocks of 1k characters
      
        for chunk in chunks:
              
            time.sleep(0.5)
            
            prompt = f"""Summarize the Context below in one sentence. Gear the summary towards answering the Question below. If you don't know, say "I don't know."
          
            Context:
            {chunk}
          
            Q: {message}
            A:"""
          
            summarized_context = openai.Completion.create(
                prompt=prompt,
                temperature=0,
                max_tokens=50,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                model="text-curie-001"
            )["choices"][0]["text"].strip(" \n")
              
            context.append(summarized_context)
    
    #get an answer given the question & context from above
    prompt = f"""Answer the following question using only the context below. If you don't know the answer for certain, say I don't know. Explain your answer.
    
    Context:
    {context}
    
    Q: {message}
    A:"""
    
    response = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model='text-davinci-003'
    )["choices"][0]["text"].strip(" \n")
        
    history.append((message, response))
    return history, history

collection.load()
chatbot = gradio.Chatbot(color_map=("green", "gray"))
interface = gradio.Interface(
    chat,
    ["text", "state"],
    [chatbot, "state"],
    allow_screenshot=False,
    allow_flagging="never",
)
interface.launch(inline=True, share=True)

        
        