# How much value a college adds?

It is widely believed that people's success is invariant under a choice of college. 
In other words, smart people will succeed no matter which college they go to. 

In this project we attempt to support/falsify this hypothesis via a very simple analysis. 
We use College Scorecard Data https://collegescorecard.ed.gov/data/. It is a big data set containing 
various information (1825 features) about 6478 colleges in the USA over 19 years (1996-97 to 2015-16).

The idea is to factor out students' qualities, which in case of this dataset are represented as SAT scores. 
To do so we build a simple linear regression model predicting earnings based on SAT scores. 
This allows us to generate a very simple "normalized" college ranking via dividing an actual earnings 
by a predicted via linear regression earnings based on SAT.

# Usage

Download and unzip dataset:
```
wget https://ed-public-download.app.cloud.gov/downloads/CollegeScorecard_Raw_Data.zip; unzip CollegeScorecard_Raw_Data
```
Trian a student model:
```
python models.py --task
```

Generate ranking (creates `scores.csv`):
```
python models.py --task scores
```

Train a college model:
```
python models.py --task college
```
