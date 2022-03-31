import pymorphy2
import torch

from corus import load_lenta2
from razdel import tokenize
from torch_geometric.data import Data

class Morpher:
    """pymorphy with cash"""
    def __init__(self, pymorphy_morpher):
        self.morpher = pymorphy_morpher
        self.cash = dict()
        
    def __call__(self, word):        
        if word in self.cash:
            return self.cash[word]
        else:
            full_info = self.morpher.parse(word)[0]
            lemma = full_info.normal_form
            pos = full_info.tag.POS
            self.cash[word] = (lemma, pos)
            return (lemma, pos)

class Corpus():
    """a container for edges and token-index dictionaries"""
    def __init__(self, graph, token2idx, idx2token):
        self.graph = graph
        self.token2idx = token2idx
        self.idx2token = idx2token

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
       
    def texts2tokens(self, raw_texts, create_dicts=True):
        if create_dicts:
            unique_tokens = set()
        if type(raw_texts) == list:
            iterator = raw_texts
        elif raw_texts == 'lenta':
            path = './data/lenta-ru-news.csv.bz2'
            iterator = load_lenta2(path)

        texts = []
        for text in iterator:
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
                        lemmas.append(lemma)
                        if create_dicts:
                            unique_tokens.add(token)
                texts.append(lemmas)
            else:
                texts.append(tokenize)

        if create_dicts:    
            token2idx = dict(zip(unique_tokens, range(len(unique_tokens))))
            idx2token = dict(zip(range(len(unique_tokens)), unique_tokens))

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

        return Corpus(graph, token2idx, idx2token) 

    def create_graph(self, raw_texts, dicts=None):
        if dicts is not None:
            token2idx, idx2token = dicts[0], dicts[1]
            tokens = self.texts2tokens(raw_texts, create_dicts=False)
        else:
            tokens, token2idx, idx2token = self.texts2tokens(raw_texts, create_dicts=True)

        return self.tokens2graph(tokens, token2idx, idx2token)
    

def razdel_tokenize(text):
    tokens = []
    tokens_raw = list(tokenize(text))
    for token in tokens_raw:
        tokens.append(token.text)

    return tokens

