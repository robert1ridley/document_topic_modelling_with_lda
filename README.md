# Document Topic Modelling with Latent Dirichlet Allocation

This repository contains code for LDA (Latent Dirichlet Allocation) for document topic modelling. 
There exist two different LDA implementations in the repository, a gensim implementation – which can be found at
`models/Lda_gensim.py` – and an LDA written from scratch – which can be found at `models/Lda_from_scratch.py`.
These datasets used are a selection of news articles (8888 in total), which can be found in the `/data` folder, 
and are labelled `news.txt`.

## Requirements

* Python version: 3.5.1

## Start Developing

After cloning the repository:

* Setting up the environment：
    - `cd document_classification_with_lda`
    - Create a virtual environmnet: `python3 -m venv venv`
    - `source venv/bin/activate`
    - Install the project dependencies：`pip install –r requirements.txt`

* Create a Pickle file with of news articles with stopwords and punctuation removed:
    - Ensure that you are inside `/document_classification_with_lda` and that your virtual environment is running
    - Enter `python utils/file_utils.py`
    - This will generate a pickle file of the news articles with stopwords and punctuation removed. 
    The file will be at `data/no_punc_stop.txt
    - Deactivate your virtual environment by entering `deactivate`.
    
* Run the gensim version of LDA:
    - Ensure that you are inside `/document_classification_with_lda` and that your virtual environment is running. Also, ensure that you 
    have already generated the pickle file `no_punc_stop.txt`
    - Enter `python models/Lda_gensim.py data/no_punc_stop.txt <number of topics>`.
    For example, to return the top words for five topics, enter `python models/Lda_gensim.py data/no_punc_stop.txt 5`. 
    This will output the top 10 words for five topics in the terminal.
    - Deactivate your virtual environment by entering `deactivate`.
    
 * Run the from-scratch version of LDA:
    - Ensure that you are inside `/document_classification_with_lda` and that your virtual environment is running. Also, ensure that you 
    have already generated the pickle file `no_punc_stop.txt`
    - Enter `python models/Lda_from_scratch.py data/no_punc_stop.txt <number of topics>`.
    For example, to return the top words for five topics, enter `python models/Lda_from_scratch.py data/no_punc_stop.txt 5`. 
    This will output the top 10 words for five topics in the terminal.
    - Deactivate your virtual environment by entering `deactivate`.
