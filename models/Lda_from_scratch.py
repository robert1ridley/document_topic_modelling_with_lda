import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import _pickle
from nltk import word_tokenize
import random
import bisect


class Lda(object):
  def __init__(self):
    self.word_lookup = {'word_count': 0}
    self.documents = []
    self.alpha = 0.1
    self.beta = 0.0002


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


  def probs(self, v, nkm, nkr, nk, n_topics):
    n_words = self.word_lookup["word_count"]
    res = [0] * n_topics
    for k in range(n_topics):
      res[k] = (nkm[k] + self.alpha) * (nkr[k][v] + self.beta) / (nk[k] + n_words * self.beta)
    return res


  def get_topics(self, iters, w_counts, docs, n_topics):
    n_words = self.word_lookup["word_count"]
    zs = []
    nkr = [[0] * n_words for _ in range(n_topics)]
    nkm = [[0] * n_topics for _ in range(len(docs))]
    nk = [0] * n_topics

    n_words_total = 0
    for i in range(len(docs)):
      zs.append([])
      for j in range(len(docs[i])):
        topic = random.randint(0, n_topics - 1)
        zs[i].append(topic)
        ind = docs[i][j]
        nkm[i][topic] += 1
        nkr[topic][ind] += 1
        nk[topic] += 1
        n_words_total += 1
    for it in range(iters):
      print ("Iteration", it)
      for i in range(len(docs)):
        for j in range(len(docs[i])):
          ind = docs[i][j]
          k = zs[i][j]
          nkm[i][k] -= 1
          nkr[k][ind] -= 1
          nk[k] -= 1
          ps = self.probs(ind, nkm[i], nkr, nk, n_topics)
          newk = self.random_choice(ps)
          nkm[i][newk] += 1
          nkr[newk][ind] += 1
          nk[newk] += 1
          zs[i][j] = newk

    for k in range(n_topics):
      for v in range(n_words):
        nkr[k][v] /= nk[k] + 0.
        nkr[k][v] -= (w_counts[v] + 0.) / n_words_total
    return [[nk[k], nkr[k]] for k in range(n_topics)]


  def display_topics(self, topics):
    relevant = 10
    n_words = self.word_lookup["word_count"]
    for nk, t in topics:
      top = [[-1, 0]] * relevant
      for i in range(n_words):
        for j in range(relevant):
          if t[i] > top[j][1]:
            top[j + 1:] = top[j:-1]
            top[j] = [self.word_lookup[i], t[i]]
            break
      print ("\n" + "==TOPIC==", " with number of words =", nk)
      for rank in top:
        print (rank)


def tokenize_articles(articles):
  tokenized_articles = []
  for article in articles:
    article = article.lower()
    tokenized_articles.append(word_tokenize(article))
  return tokenized_articles


def main(args):
  articles_filepath = args[0]
  iterations = 10
  number_of_categories = int(args[1])
  with open(articles_filepath, 'rb') as infile:
    articles = _pickle.load(infile)
  articles = tokenize_articles(articles)
  lda = Lda()
  lda.word_counts(articles)
  word_count = lda.word_lookup['word_count']
  count = [0]*word_count
  for document in lda.documents:
    for word in document:
      count[word] += 1
  topics = lda.get_topics(iterations, count, lda.documents, number_of_categories)
  lda.display_topics(topics)



if __name__ == '__main__':
  main(sys.argv[1:])