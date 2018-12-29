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
    self.number_of_words = None
    self.topic_and_words = None
    self.individual_word_count_for_single_topic = None
    self.document_and_topic_co_occurrence_count = None
    self.topic_word_counts = None
    self.zs = []


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


  def probs(self, v, i):
    n_words = self.word_lookup["word_count"]
    updated_probability = [0] * self.number_of_categories
    for k in range(self.number_of_categories):
      updated_probability[k] = (self.document_and_topic_co_occurrence_count[i][k] + self.alpha) * \
               (self.individual_word_count_for_single_topic[k][v] + self.beta) / \
               (self.topic_word_counts[k] + n_words * self.beta)
    return updated_probability


  def iterate_and_update(self, w_counts):
    for it in range(self.number_of_passes):
      print ("Pass", it)
      for i in range(len(self.documents)):
        for j in range(len(self.documents[i])):
          entity = self.documents[i][j]
          k = self.zs[i][j]
          self.document_and_topic_co_occurrence_count[i][k] -= 1
          self.individual_word_count_for_single_topic[k][entity] -= 1
          self.topic_word_counts[k] -= 1
          updated_probabilities = self.probs(entity, i)
          updated_k = self.random_choice(updated_probabilities)
          self.document_and_topic_co_occurrence_count[i][updated_k] += 1
          self.individual_word_count_for_single_topic[updated_k][entity] += 1
          self.topic_word_counts[updated_k] += 1
          self.zs[i][j] = updated_k

    for k in range(self.number_of_categories):
      for v in range(self.number_of_words):
        self.individual_word_count_for_single_topic[k][v] /= self.topic_word_counts[k]
        self.individual_word_count_for_single_topic[k][v] -= (w_counts[v]) / self.n_words_total
    self.topic_and_words = [[self.topic_word_counts[k], self.individual_word_count_for_single_topic[k]]
                            for k in range(self.number_of_categories)]


  def initial_random_topic_assignment(self):
    self.n_words_total = 0
    for i in range(len(self.documents)):
      self.zs.append([])
      for j in range(len(self.documents[i])):
        topic = random.randint(0, self.number_of_categories - 1)
        self.zs[i].append(topic)
        ind = self.documents[i][j]
        self.document_and_topic_co_occurrence_count[i][topic] += 1
        self.individual_word_count_for_single_topic[topic][ind] += 1
        self.topic_word_counts[topic] += 1
        self.n_words_total += 1


  def initialize_word_and_category_counts(self):
    self.number_of_words = self.word_lookup["word_count"]
    self.individual_word_count_for_single_topic = []
    for x in range(self.number_of_categories):
      nkr_intermediate = []
      for _ in range(self.number_of_words):
        nkr_intermediate.append(0)
      self.individual_word_count_for_single_topic.append(nkr_intermediate)

    self.document_and_topic_co_occurrence_count = []
    for x in range(len(self.documents)):
      nkm_intermediate = []
      for _ in range(self.number_of_categories):
        nkm_intermediate.append(0)
      self.document_and_topic_co_occurrence_count.append(nkm_intermediate)

    self.topic_word_counts = []
    for _ in range(self.number_of_categories):
      self.topic_word_counts.append(0)


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
  iterations = 10
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
  lda.initialize_word_and_category_counts()
  lda.initial_random_topic_assignment()
  lda.iterate_and_update(count)
  lda.display_topics()


if __name__ == '__main__':
  main(sys.argv[1:])