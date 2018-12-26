import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import _pickle
from gensim.models import LdaModel
from gensim import corpora
from nltk import word_tokenize


def tokenize_articles(articles):
    tokenized_articles = []
    for article in articles:
        article = article.lower()
        tokenized_articles.append(word_tokenize(article))
    return tokenized_articles


def main(args):
    articles_filepath = args[0]
    number_of_categories = int(args[1])
    with open(articles_filepath, 'rb') as infile:
        articles = _pickle.load(infile)
    articles = tokenize_articles(articles)
    dictionary = corpora.Dictionary(articles)
    corpus = [dictionary.doc2bow(article) for article in articles]
    lda = LdaModel(corpus, num_topics=number_of_categories, id2word=dictionary, passes=2)
    for topic, words in lda.print_topics(-1):
        words_split = words.split(' ')
        while '+' in words_split:
            words_split.remove('+')
        print('Topic: {}'.format(topic))
        for term in words_split:
            word_val_split = term.split('*')
            word = word_val_split[1]
            probs = lda.get_term_topics(word[1:-1])
            for prob in probs:
                if prob[0] == topic:
                    probability = prob[1]
            print('Word: {} Probability: {}'.format(word, probability))


if __name__ == '__main__':
    main(sys.argv[1:])
