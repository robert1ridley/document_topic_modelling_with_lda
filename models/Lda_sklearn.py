import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import _pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def main(args):
    articles_filepath = args[0]
    number_of_categories = int(args[1])
    number_of_words = int(args[2])
    with open(articles_filepath, 'rb') as infile:
        articles = _pickle.load(infile)
    vectorised = CountVectorizer()
    bow = vectorised.fit_transform(articles).todense()
    feature_names = vectorised.get_feature_names()
    lda = LatentDirichletAllocation(n_components=number_of_categories, max_iter=5, learning_method='online',
                                    learning_offset=50., random_state=0).fit(bow)
    display_topics(lda, feature_names, number_of_words)


if __name__ == '__main__':
    main(sys.argv[1:])
