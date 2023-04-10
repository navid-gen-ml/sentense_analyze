import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from bltk.langtools.banglachars import punctuations

excel_file = 'Sentence_Wise_SignSentence.xlsx'

# Read the data from the Excel file from column 'SIGN SENTENCE'
df = pd.read_excel(excel_file)
sentences = df['SIGN SENTENCE'].tolist()
# sentences = sentences[:1]

# sentences = [
#     "বিশেষ বিয়ে আইন",
#     "মানুষ আলাদা নতুন শিক্ষা আছে",
#     "খেজুর ভিতর আলাদা বিশেষ ভিটামিন আছে",
#     "ব্যাংক-এর বিশেষ অফিস",
#     "পুলিশ-এর বিশেষ বিভাগ",
#     "বিশেষ নিরাপততা বাহিনী",
#     "বিশেষ খবর",
#     "বিশেষ খবর মিটিং"
# ]

# spark = sparknlp.start(gpu=False)

# document_assembler = DocumentAssembler() \
# .setInputCol("text") \
# .setOutputCol("document")

# tokenizer = Tokenizer()\
# .setInputCols(["document"]) \
# .setOutputCol("token")

# # lemmatizer = LemmatizerModel.load('lemma_bn')
# lemmatizer = LemmatizerModel.pretrained("lemma", "bn") \
# .setInputCols(["token"]) \
# .setOutputCol("lemma")

# nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, lemmatizer])
# light_pipeline = LightPipeline(
#     nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

# def remove_punctuations(text: str):
#     for p in punctuations:
#         text = str(text).replace(p, "")
#     return text

# def lemmatize(text):
#     results = light_pipeline.annotate(text)
#     return results['lemma']

# def sentence_list_to_lemma(sentence_list: list):
#     return [lemmatize(remove_punctuations(sentence)) for sentence in sentence_list]

# sentences = sentence_list_to_lemma(sentences)

vocab = set()
for sentence in sentences:
    words = str(sentence).split()
    for word in words:
        if word not in punctuations:
            vocab.add(word)

vocab = list(vocab)
vocab_size = len(vocab)

# load the co-occurrence matrix from a file
# co_occurrence_matrix = np.loadtxt('co-occurrence_matrix.txt', dtype=int)

co_occurrence_matrix = np.zeros((vocab_size, vocab_size))

# for plotting
matrix = []

for sentence in sentences:
    words = str(sentence).split()
    for i in range(len(words)):
        for j in range(max(0, i - 2), min(len(words), i + 3)):
            if i != j:
                if words[i] in punctuations or words[j] in punctuations:
                    continue
                word1_index = vocab.index(words[i])
                word2_index = vocab.index(words[j])
                co_occurrence_matrix[word1_index][word2_index] += 1
                if co_occurrence_matrix[word1_index][word2_index] > 0:
                    if len(matrix) > 0:
                        if [words[i], words[j]] in matrix:
                            index = matrix.index([words[i], words[j]])
                            matrix[index][2] += 1
                        else:
                            matrix.append([words[i], words[j], co_occurrence_matrix[word1_index][word2_index]])
                    else:
                        matrix.append([words[i], words[j], co_occurrence_matrix[word1_index][word2_index]])

# save the matrix to a file
np.savetxt('matrix.txt', matrix, fmt='%s')

# plot the matrix
df = pd.DataFrame(matrix, columns=['word1', 'word2', 'count'])
df.to_csv('matrix.csv', index=False)

# save the co-occurrence matrix to a file
# np.savetxt('co-occurrence_matrix.txt', co_occurrence_matrix, fmt='%d')

# Create a heatmap using Seaborn
# sns.set()

# font_dirs = ['/home/navid/Desktop/sentence_analyze/solaimanlipi']
# font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

# for font_file in font_files:
#     font_manager.fontManager.addfont(font_file)

# plt.rcParams["font.family"] = "SolaimanLipi"


# # if co_occurrence_matrix values greather than 100 then count for visualization
# co_occurrence_matrix = np.where(co_occurrence_matrix > 100, 100, co_occurrence_matrix)
# plt.figure(figsize=(20, 8))
# sns.heatmap(co_occurrence_matrix, cmap='YlGnBu', xticklabels=vocab, yticklabels=vocab)
# plt.title('Co-occurrence Matrix')
# plt.xlabel('Words')
# plt.ylabel('Words')
# plt.show()