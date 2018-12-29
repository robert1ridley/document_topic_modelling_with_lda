import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import _pickle
from nltk import word_tokenize
import random
import bisect


class Lda(object):
  def __init__(self, in_number_of_categories, in_number_of_passes):
    self.word_lookup = {'word_count': 0}
    self.documents = []
    self.alpha = 0.1
    self.beta = 0.0002
    self.number_of_categories = in_number_of_categories
    self.number_of_passes = in_number_of_passes
    self.words_per_topic = 10
    self.n_words = None
    self.topic_and_words = None


  def word_counts(self, articles):
    for article in articles:
      document = []
      for word in article:
        if word not in self.word_lookup:
          n_words = self.word_lookup["word_count"]
          self.word_lookup[word] = n_words
          self.word_lookup[n_words] = word
          self.word_lookup["word_count"] += 1
        ind = self.word_lookup[word]
        document.append(ind)
      self.documents.append(document)


  def random_choice(self, probs):
    partials = []
    psum = 0.
    for p in probs:
      psum += p
      partials.append(psum)
    choice = random.random() * psum
    return bisect.bisect_right(partials, choice)


  def probs(self, v, nkm, nkr, nk):
    n_words = self.word_lookup["word_count"]
    res = [0] * self.number_of_categories
    for k in range(self.number_of_categories):
      res[k] = (nkm[k] + self.alpha) * (nkr[k][v] + self.beta) / (nk[k] + n_words * self.beta)
    return res


  def get_topics(self, w_counts):
    self.n_words = self.word_lookup["word_count"]
    number_of_categories = self.number_of_categories
    number_of_passes = self.number_of_passes
    documents = self.documents
    zs = []
    nkr = [[0] * self.n_words for _ in range(number_of_categories)]
    nkm = [[0] * number_of_categories for _ in range(len(documents))]
    nk = [0] * number_of_categories

    n_words_total = 0
    for i in range(len(documents)):
      zs.append([])
      for j in range(len(documents[i])):
        topic = random.randint(0, number_of_categories - 1)
        zs[i].append(topic)
        ind = documents[i][j]
        nkm[i][topic] += 1
        nkr[topic][ind] += 1
        nk[topic] += 1
        n_words_total += 1
    for it in range(number_of_passes):
      print ("Pass", it)
      for i in range(len(documents)):
        for j in range(len(documents[i])):
          ind = documents[i][j]
          k = zs[i][j]
          nkm[i][k] -= 1
          nkr[k][ind] -= 1
          nk[k] -= 1
          ps = self.probs(ind, nkm[i], nkr, nk)
          newk = self.random_choice(ps)
          nkm[i][newk] += 1
          nkr[newk][ind] += 1
          nk[newk] += 1
          zs[i][j] = newk

    for k in range(number_of_categories):
      for v in range(self.n_words):
        nkr[k][v] /= nk[k] + 0.
        nkr[k][v] -= (w_counts[v] + 0.) / n_words_total
    self.topic_and_words = [[nk[k], nkr[k]] for k in range(number_of_categories)]


  def display_topics(self):
    words_per_topic = self.words_per_topic
    n_words = self.word_lookup["word_count"]
    topics_and_words = self.topic_and_words
    topic_num = 1
    for nk, t in topics_and_words:
      topic = [[-1, 0]] * words_per_topic
      for i in range(n_words):
        for j in range(words_per_topic):
          if t[i] > topic[j][1]:
            topic[j + 1:] = topic[j:-1]
            topic[j] = [self.word_lookup[i], t[i]]
            break
      print ("\nTopic: {}".format(topic_num))
      topic_num += 1
      for word_prob_pair in topic:
        print ('Word: "{}" Probability: {}'.format(word_prob_pair[0], word_prob_pair[1]))


def tokenize_articles(articles):
  tokenized_articles = []
  for article in articles:
    article = article.lower()
    tokenized_articles.append(word_tokenize(article))
  return tokenized_articles


def main(args):
  articles_filepath = args[0]
  iterations = 2
  number_of_categories = int(args[1])
  with open(articles_filepath, 'rb') as infile:
    articles = _pickle.load(infile)
  articles = tokenize_articles(articles)
  lda = Lda(number_of_categories, iterations)
  lda.word_counts(articles)
  word_count = lda.word_lookup['word_count']
  count = [0]*word_count
  for document in lda.documents:
    for word in document:
      count[word] += 1
  lda.get_topics(count)
  lda.display_topics()


if __name__ == '__main__':
  main(sys.argv[1:])