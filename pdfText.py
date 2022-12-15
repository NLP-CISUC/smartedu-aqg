import re
import fitz
from nltk.tokenize import sent_tokenize
from loggerLog import *

def readPdf(filename):
    documentText = ""
    with fitz.open(filename) as pdfDoc:
        for page in pdfDoc:
            pageText = page.get_text()
            pageText = pageText.replace('\n', ' ')
            documentText += " " + pageText
    try:
        documentText = processPdfText(documentText)
    except Exception as e:
        logger.error("Exception in function processPdfText in readPdf (pdfText.py): %s", e)
    return documentText


def processPdfText(text):
    text = re.sub('Page[0-9]+in[0-9]+', ' ', text)
    text = re.sub(' [0-9]+\. ', ' ', text)
    text = text.replace('•', ' ')
    text = text.replace('➢', ' ')
    text = text.replace('‣', ' ')
    text = text.replace('›', ' ')
    text = re.sub(' +', ' ', text)
    textSentences = sent_tokenize(text)
    textSentences = [sentence for sentence in textSentences if sentence[-1] not in ["?","!"]]
    text = " ".join(textSentences)
    return text


def readTxt(filename):
    filename += ".txt"
    data = ""
    with open(filename, 'r', encoding='utf8') as f:
        data = f.read()
    return data


def writeTxt(filename, mode, data):
    filename += ".txt"
    with open(filename, mode, encoding='utf8') as f:
        f.write(data)


def writePdfToTxt(filename):
    pdfText = readPdf(filename)
    writeTxt(filename, 'w', pdfText)
