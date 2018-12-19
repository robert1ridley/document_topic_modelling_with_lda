import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import load_file
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import numpy
from sklearn.feature_extraction.text import CountVectorizer


class Lda_sklearn (object):

    def __init__(self):
        pass


def remove_stop_and_punc(articles):
    smart_punct = [u"\u2018", u"\u2019", u"\u201c", u"\u201d"]
    stop = stopwords.words('english')
    stop_and_punc_removed_articles = []
    for article in articles:
        no_stop_punc = []
        article_tokens = word_tokenize(article)
        for token in article_tokens:
            if token not in punctuation and token not in smart_punct and token not in stop:
                no_stop_punc.append(token)
        #         IF PERFORMANCE POOR, THINK ABOUT ADDING LOWER_CASE AND 还原
        stop_and_punc_removed_articles.append(" ".join(no_stop_punc))
    return stop_and_punc_removed_articles


def main(args):
    articles_doc = args[0]
    articles = load_file(articles_doc)
    stop_and_punc_removed = remove_stop_and_punc(articles)
    vectorised = CountVectorizer()
    bow = vectorised.fit_transform(stop_and_punc_removed).todense()
    print(bow.shape)


if __name__ == '__main__':
    main(sys.argv[1:])
