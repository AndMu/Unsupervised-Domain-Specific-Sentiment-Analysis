# Wikiled.Sentiment

## *pSenti* Lexicon system

* Download [*pSenti*](https://github.com/AndMu/Wikiled.Sentiment/releases/tag/2.6.55) lexicon based utility

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

## Different embeddings

* [Extracted Lexicons](http://datasets.azurewebsites.net/Amazon/words_Amazon.zip)
* [Test Kitchen domain](http://datasets.azurewebsites.net/Amazon/Kitchen_exp.zip)

```
python DiscoverLexicon.py -d amazon_kitchen  # to extract full kitchen domain lexicon
python DiscoverLexicon.py -d amazon_video  # to extract full kitchen domain lexicon
python DiscoverLexicon.py -d amazon_electronics  # to extract full kitchen domain lexicon
Wikiled.Sentiment.ConsoleApp.exe test -Articles="[Path]\Kitchen_exp.xml" -Out=.\kitchen_psenti
Wikiled.Sentiment.ConsoleApp.exe test -Articles="[Path]\Kitchen_exp.xml" -Out=.\kitchen_own -Weights="[Path]\words_Amazon_Kitchen.csv" -FullWeightReset
Wikiled.Sentiment.ConsoleApp.exe test -Articles="[Path]\Kitchen_exp.xml" -Out=.\kitchen_t_el -Weights="[Path]\words_Amazon_Electronics.csv" -FullWeightReset
Wikiled.Sentiment.ConsoleApp.exe test -Articles="[Path]\Kitchen_exp.xml" -Out=.\kitchen_t_vi -Weights="[Path]\words_Amazon_Video.csv" -FullWeightReset
```

## IMDB Domain

### Datasets

Can be downloaded from (http://ai.stanford.edu/~amaas/data/sentiment/)

### Process

```
python DiscoverLexicon.py -d imdb -c 0.7 -b  # Induce Sentiment Lexicon
Wikiled.Sentiment.ConsoleApp.exe bootimdb -Words="[path]\words_imdb.csv" -Path="[path]" -Destination="[path]" -BalancedTop=0.8 # Bootstrap Training dataset
Python Sentiment.py -d imdb -a lstm -n 2 -p # Train sentiment classifier
```

## Amazon Domain

### Datasets

* [Test](http://datasets.azurewebsites.net/Amazon/Test.zip)
* [Unlabeled](http://datasets.azurewebsites.net/Amazon/unlabel.zip)

### Process

```
python DiscoverLexicon.py -d amazon -c 0.7 -b 
Wikiled.Sentiment.ConsoleApp.exe boot -Words="[path]\words_amazon.csv" -Path="[path]\unlabel.txt" -Destination="[path]" -BalancedTop=0.8
Python Sentiment.py -d imdb -a lstm -n 2 -p 
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
