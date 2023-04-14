""" 
Created on Tue Feb 21 17:25:14 2023

@author: Shane Tomasello
"""

import pandas as pd
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


df = pd.read_csv(r"<Directory for wine csv>")


def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
    FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', is_primary=True, auto_id=False),
    FieldSchema(name='country', dtype=DataType.VARCHAR, descrition='country'),
    FieldSchema(name='description', dtype=DataType.VARCHAR, descrition='wine discription'),
    FieldSchema(name='designation', dtype=DataType.VARCHAR, descrition='designation'),
    FieldSchema(name='winery', dtype=DataType.VARCHAR, descrition='winery'),
    FieldSchema(name='price', dtype=DataType.VARCHAR, descrition='price'),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

collection = create_milvus_collection('question_answer', 768)
