import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import _pickle

def load_file(filepath):
  infile = open(filepath, 'r')
  data = infile.readlines()
  new_list = []
  for line in data:
    line = line.strip()
    new_list.append(line)
  infile.close()
  return new_list


def remove_stop_and_punc(articles):
  smart_punct = [u"\u2018", u"\u2019", u"\u201c", u"\u201d"]
  stop = stopwords.words('english')
  stop_and_punc_removed_articles = []
  for article in articles:
    no_stop_punc = []
    article_tokens = word_tokenize(article)
    for token in article_tokens:
      if token not in punctuation and token not in smart_punct and token.lower() not in stop:
        no_stop_punc.append(token.lower())
    stop_and_punc_removed_articles.append(" ".join(no_stop_punc))
  return stop_and_punc_removed_articles


def main(args):
  articles_doc = args[0]
  articles = load_file(articles_doc)
  stop_and_punc_removed = remove_stop_and_punc(articles)
  with open('./data/no_punc_stop.txt', 'wb') as outfile:
    _pickle.dump(stop_and_punc_removed, outfile)


if __name__ == '__main__':
    main(sys.argv[1:])
