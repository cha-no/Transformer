from typing import (
    Tuple, List
)

import pandas as pd
import numpy as np
import unicodedata
import re

import urllib3
import zipfile
import shutil
import os

import warnings
warnings.filterwarnings(action='ignore')

# 프랑스어 처리
def unicode_to_ascii(s : str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# 특수문자 제거 
def preprocess(sentence : str) -> str:
    sentence = unicode_to_ascii(sentence.lower())
    
    sentence = re.sub(r"([?.!,¿])", r" \1", sentence)

    sentence = re.sub(r"[^a-zA-Z!.?]+", r" ", sentence)

    sentence = re.sub(r"\s+", " ", sentence)
    return sentence

# dog_breed dataset
def download_data() -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    http = urllib3.PoolManager()
    url ='http://www.manythings.org/anki/fra-eng.zip'
    
    DATA_PATH = "data"
    filename = 'fra-eng.zip'

    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    path = os.getcwd()
    zipfilename = os.path.join(path, DATA_PATH, filename)
    
    with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:
        shutil.copyfileobj(r, out_file)

    with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(path, DATA_PATH))

    print('\nSuccess Download eng-fra file!!\n')

    df = pd.read_csv(DATA_PATH + '/fra.txt', names=['eng', 'fra', 'lic'], sep='\t')
    del df['lic']
    df = df.sample(frac = 0.1)
    
    eng_input = [preprocess(sentence).split() for sentence in df['eng']]
    fra_input = [['<SOS>'] + preprocess(sentence).split() for sentence in df['fra']]
    fra_target = [preprocess(sentence).split() + ['<EOS>'] for sentence in df['fra']]

    return eng_input, fra_input, fra_target