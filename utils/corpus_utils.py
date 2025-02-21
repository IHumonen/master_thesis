import pymorphy2
import torch

import pandas as pd

from collections import defaultdict

from corus import load_lenta2
import gensim.downloader
from razdel import tokenize
from torch_geometric.data import Data

class Morpher:
    """pymorphy with cash"""
    def __init__(self, pymorphy_morpher):
        self.morpher = pymorphy_morpher
        self.cash = dict()
        self.freq = defaultdict(int)

    def __call__(self, word):   
        if word in self.cash:
            return self.cash[word]
        else:
            full_info = self.morpher.parse(word)[0]
            lemma = full_info.normal_form
            pos = full_info.tag.POS
            self.cash[word] = (lemma, pos)
            return (lemma, pos)
    # def max_freq(self):
    #     return max(self.freq.values())+1

class Corpus():
    """a container for edges and token-index dictionaries"""
    def __init__(self, graph, token2idx, idx2token):
        self.graph = graph
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.w2v_vectors = None
        
    def create_w2v_vectors(self, w2v_path='word2vec-ruscorpora-300'):
        w2v = gensim.downloader.load(w2v_path)
        
        w2v_size = w2v[0].shape[0]
        max_id = max(self.token2idx.values())+1
        w2v_vectors = torch.zeros((max_id, w2v_size))
        for key in w2v.key_to_index.keys():
            word = key.split('_')[0]
            if word in self.token2idx:
                index = self.token2idx[word]
                w2v_vectors[index] = torch.tensor(w2v[key])

        self.w2v_vectors = w2v_vectors

class CorpusMaker():
    """creates a Corpus objects from a collection of texts"""
    def __init__(self, tokenizer='razdel', morpher='pymorphy2', window_size=3):
        self.tokenizer = tokenizer
        if self.tokenizer == 'razdel':
            self.tokenizer = razdel_tokenize

        self.morpher = morpher
        if self.morpher is not None:
            self.morpher = Morpher(pymorphy2.MorphAnalyzer())

        self.window_size = window_size
       
    def texts2tokens(self, raw_texts, create_dicts=True, freq_threshold=0.7, texts_number=10000, print_every=10):

        if type(raw_texts) == list:
            iterator = raw_texts
        elif raw_texts == 'lenta':
            path = './data/lenta-ru-news.csv.bz2'
            iterator = load_lenta2(path)


        texts = []
        i = 0
        for text in iterator:
            if i < texts_number:
                if i % print_every == 0:
                    print(f'step {i}')
                i+=1
                if raw_texts == 'lenta':
                    text = text.text

                if self.tokenizer is not None:
                    tokenized = self.tokenizer(text)
                else:
                    tokenized = raw_texts
                if self.morpher is not None:
                    lemmas = []
                    for token in tokenized:
                        lemma, pos = self.morpher(token)
                        if pos in ['NOUN', 'INFN', 'VERB', 'ADJF']:
                            self.morpher.freq[lemma] += 1 
                            lemmas.append(lemma)

                    texts.append(lemmas)
                else:
                    texts.append(tokenize)
            else:
                break

        if create_dicts:

            df = pd.DataFrame({'word' : self.morpher.freq.keys(), 'freq' : self.morpher.freq.values()})
            print(f'{df.shape[0]} вершин было')
            
            if freq_threshold < 1: #type float
                quantile = 1 - freq_threshold
                freq_threshold = df.quantile(q=quantile)['freq']
            else: #type int
                pass
            print(f'порог {freq_threshold}')
            
            df = df[df['freq'] > freq_threshold].reset_index()
            df.to_csv('corpus_freq.csv')

            token2idx = dict(zip(df['word'], df.index))
            idx2token = dict(zip(df.index, df['word']))

            print(f'{df.shape[0]} вершин стало')

            return texts, token2idx, idx2token
        else:
            return texts

    def tokens2graph(self, texts, token2idx, idx2token):
        edges = []
        for text in texts:
            for i, token in enumerate(text):
                if token in token2idx:
                    left = max(i-self.window_size, len(text))
                    right = min(i+self.window_size, len(text))
                    neighbours = text[left: i] + text[i+1: right]
                    for neighbour in neighbours:
                        if neighbour in token2idx:
                            edge = [token2idx[token], token2idx[neighbour]]
                            if edge != [] and edge[0] != edge[1] and not (edge in edges):
                                edges.append(edge)

        x = torch.tensor(list(idx2token.keys()))
        if edges == []:
            edges = [[0,1]]
            print('empty ===========================')
        edge_index = torch.tensor(edges).transpose(0, 1)
        graph = Data(x = x.long(), edge_index = edge_index)

        print(f'{len(edges)} рёбер' )

        return Corpus(graph, token2idx, idx2token) 

    def create_graph(self, raw_texts, dicts=None, print_every=10):
        if dicts is not None:
            token2idx, idx2token = dicts[0], dicts[1]
            tokens = self.texts2tokens(raw_texts, create_dicts=False)
        else:
            tokens, token2idx, idx2token = self.texts2tokens(raw_texts, create_dicts=True, print_every=print_every)

        return self.tokens2graph(tokens, token2idx, idx2token)
    

def razdel_tokenize(text):
    tokens = []
    tokens_raw = list(tokenize(text))
    for token in tokens_raw:
        tokens.append(token.text)

    return tokens

