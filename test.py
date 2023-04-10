import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import pandas as pd

# Tokenize the sentences into words
# sentences = [
#     "বিশেষ বিয়ে আইন",
#     "মানুষ আলাদা নতুন শিক্ষা আছে",
#     "খেজুর ভিতর আলাদা বিশেষ ভিটামিন আছে",
#     "ব্যাংক-এর বিশেষ অফিস",
#     "পুলিশ-এর বিশেষ বিভাগ",
#     "বিশেষ নিরাপততা বাহিনী",
#     "বিশেষ খবর",
#     "বিশেষ খবর মিটিং",
# ]

excel_file = 'Sentence_Wise_SignSentence.xlsx'

# Read the data from the Excel file from column 'SIGN SENTENCE'
df = pd.read_excel(excel_file)
sentences = df['SIGN SENTENCE'].tolist()

sentences = sentences[:100]

words = []
for sentence in sentences:
    words.extend(str(sentence).split())

# Create a vocabulary of unique words
vocab = sorted(set(words))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# Create one hot vectors for each sentence
one_hot_vectors = []
for sentence in sentences:
    one_hot_vector = np.zeros(len(vocab))
    for word in str(sentence).split():
        one_hot_vector[word_to_idx[word]] = 1
    one_hot_vectors.append(one_hot_vector)

# Compute the co-occurrence matrix
co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))
for one_hot_vector in one_hot_vectors:
    co_occurrence_matrix += np.outer(one_hot_vector, one_hot_vector)

font_dirs = ['/home/navid/Desktop/sentence_analyze/solaimanlipi']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams["font.family"] = "SolaimanLipi"

# Visualize the co-occurrence network using a graph
G = nx.DiGraph(co_occurrence_matrix)
labels = {idx: word for word, idx in word_to_idx.items()}
nx.relabel_nodes(G, labels, copy=False)
# nx.draw(G, with_labels=True)
# set font properties
font_path = "solaimanlipi/SolaimanLipi.ttf"  # replace with the path to your font file
font_prop = FontProperties(fname=font_path, size=16)

# plot networkx graph with custom font
nx.draw(G, with_labels=True, font_family="SolaimanLipi", font_size=16)
plt.show()