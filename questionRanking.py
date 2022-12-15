import pdfText
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import copy
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizerGPT2 = GPT2Tokenizer.from_pretrained("gpt2")
modelGPT2 = GPT2LMHeadModel.from_pretrained("gpt2")
from loggerLog import *


def sentence_split(paragraph):
    return nltk.sent_tokenize(paragraph)


def sentence_score(sentence_list, df):
    sentences_scores = [0] * len(sentence_list)

    for sentence in range(len(sentence_list)):
        count_words = 0
        sum_values = 0
        for value in range(len(df.iloc[[0]].values[0])):
            num = df.iloc[[sentence]].values[0][value]
            if (num != 0):
                count_words += 1
                sum_values += num
        if count_words != 0:
            sentences_scores[sentence] =  sum_values/count_words

    return sentences_scores


def summarizeTF_IDF(text, maxSentences):
    #sentences are docs
    try:
        sentence_list = sentence_split(text)
    except Exception as e:
        logger.error("Exception in function sentence_split in summarizeTF_IDF (questionRanking.py): %s", e)

    if len(sentence_list) < maxSentences:
        maxSentences = len(sentence_list)

    stopWords = set(stopwords.words('english'))

    vectorizer = TfidfVectorizer(stop_words=stopWords)
    vectors = vectorizer.fit_transform(sentence_list)
    feature_names = vectorizer.get_feature_names()

    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)

    try:
        sentences_scores = sentence_score(sentence_list, df)
    except Exception as e:
        logger.error("Exception in function sentence_score in summarizeTF_IDF (questionRanking.py): %s", e)

    sentences_scores_sorted = copy.deepcopy(sentences_scores)
    sentences_scores_sorted.sort(reverse=True)
    threshold = sentences_scores_sorted[maxSentences-1]

    summary = []
    for i in range(len(sentences_scores)):
        if sentences_scores[i] >= threshold:
            summary.append(sentence_list[i])

    result = ''
    for i in range(len(summary)):
        result += '{} '.format(summary[i])

    return result


def scoreGPT2(text):
    input_ids = torch.tensor(tokenizerGPT2.encode(text)).unsqueeze(0)
    output = modelGPT2(input_ids, labels=input_ids)
    loss = output.loss
    logits = output.logits
    return loss.item()


# acrescentar gpt2 para só ter uma pergunta por frase
def selectQuestionGPT2(questions, answer):
    scores = []
    for question in questions:
        scores.append([question, scoreGPT2(tokenizerGPT2, modelGPT2, question+" "+answer)])
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
    return scores[0][0]