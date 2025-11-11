# load documents from file
import os 
import numpy as np
from pathlib import Path 
from typing import Dict , List, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import CSVLoader



def load_all_document(data_dirs: str) -> List[Any]:
    # use project root data folder
    data_path = Path(data_dirs).resolve()
    print(f"[Debug] data path: {data_path}")
    documents = []

    # pdf files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[Debug] found {len(pdf_files)} pdf files")
    for pdf_file in pdf_files:
        print(f"[Debug] loading {pdf_file}")
        try: 
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[Debug] loaded {len(loaded)} pdf documents from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] failed to load {pdf_files} due to {e}")
    

    # text files
    text_files = list(data_path.glob("**/*.txt"))
    print(f"[Debug] found {len(text_files)} text files")
    for text_file in text_files:
        print(f"[Debug] loading {text_file}")
        try:
            loader = TextLoader(str(text_file))
            loaded = loader.load()
            print(f"[Debug] loaded {len(loaded)} text documents from {text_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] failed to load {text_file} due to {e}")
    
    # json file
    json_files = list(data_path.glob("**/*.json"))
    print(f"[Debug] found {len(json_files)} json files")
    for json_file in json_files:
        print(f"[Debug] loading {json_file} file")
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            print(f"[Debug] loaded {len(loaded)} json files from {json_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] failed to load {json_file} due to {e}")
        
    return documents





    
