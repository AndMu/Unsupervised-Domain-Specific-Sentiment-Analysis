# Wikiled.Sentiment

## *pSenti* Lexicon system

* Download [*pSenti*](AndMu/Wikiled.Sentiment/releases/tag/2.6.55) lexicon based utility

* Download [Sentiment resources](AndMu/Wikiled.Sentiment/src/Resources) 

## Python requirements

* Python 2.7
* Keras >= 2.0.5
* CNTK 2.2
* Unit Test [Data](http://datasets.azurewebsites.net/other/test.zip)

## Word2Vec Embeddings

* [IMDB Reviews](http://datasets.azurewebsites.net/Word2Vec/Imdb.zip)
* [SemEval-2017](http://datasets.azurewebsites.net/Word2Vec/SemEval.zip)
* [Amazon Reviews](http://datasets.azurewebsites.net/Word2Vec/Amazon.zip)
* [Amazon Kitchen](http://datasets.azurewebsites.net/Word2Vec/Kitchen.zip)
* [Amazon Video](http://datasets.azurewebsites.net/Word2Vec/Video.zip)
* [Amazon Electronics](http://datasets.azurewebsites.net/Word2Vec/Electronics.zip)

## IMDB Domain

### Datasets

Can be downloaded from (http://ai.stanford.edu/~amaas/data/sentiment/)

### Process

```
python DiscoverLexicon.py -d imdb -c 0.7 -b  # Induce Sentiment Lexicon
Wikiled.Sentiment.ConsoleApp.exe bootimdb -Words="[path]\words_imdb.csv" -Path="[path]" -Destination="[path]" -BalancedTop=0.8 # Bootstrap Training dataset
Python Sentiment.py -d imdb -a lstm -n 2 -p # Train sentiment classifier
```

## SemEval-2017

### Datasets

* [Unlabelled](http://datasets.azurewebsites.net/SemEval/all.zip)
* [Test](http://datasets.azurewebsites.net/SemEval/test.zip)

### Process

In brackets listed options for binary and multiclass classification. If necesary you can bootstrap neutral class with *pSenti*
```
python DiscoverLexicon.py -d semeval -c 0.7 -b # Induce Sentiment Lexicon
Wikiled.Sentiment.ConsoleApp.exe semboot -Words="[path]\words_semeval.csv" -Path="[path]" -Destination="[path]\train.txt" -InvertOff [-Neutral] # Bootstrap Training dataset
Python Sentiment.py -d semeval -a lstm -n [2 or 5] -p # Train sentiment classifier
```
