# NLP-MIT-Media-Fatuality-Bias

This repository contains natural language processing model improvement and comparison in order to find out a more cost-efficient and faster computation model to realize the function of real-time prediction. 

## 1 Inspiration From
* 'Predicting Factuality of Reporting and Bias of News Media Sources'
Ramy Baly, Georgi Karadzhov, Dimitar Alexandrov, James Glass, Preslav Nakov MIT Computer Science and Artificial Intelligence Laboratory, MA, USA Qatar Computing Research Insti- tute, HBKU, Qatar; Sofia University, Bulgaria
* Language-Features-for-News
Benjamin DHorne
https://github.com/BenjaminDHorne/Language-Features-for-News

## 2 Data Overview and Classification.py

### 2.1 Corpus
The corpus is created by retrieving websites and factuality/bias labels from the Media Bias/Fact Check (MBFC) [website](http://mediabiasfactcheck.com/). The corpus is stored **data/corpus.csv**, which contains the following fields:
* **source_url**: the URL to each website (example: http://www.who.int/en/)
* **source_url_processed**: a shortened version of the *source_url* (example: who.int-en). These will be used as IDs to split the data into 5 folds of training and testing (in **data/splits.txt**) 
* **URL**: the link to the page in the MBFC website analyzing the corresponding website (example: http://mediabiasfactcheck.com/world-health-organization-who/)
* **fact**: the factuality label of each website (low, mixed, or high)
* **bias**: the bias label of each website (extreme-right, right, center-right, center, center-left, left, extreme-left)

### 2.2 Features
In addition to the corpus, we provide the different features that we used to obtain the results in our EMNLP paper, as well as a script to run the classification and re-generate the results.

Here is the list of features categorized by the source from which they were extracted:
* Traffic: *alexa_rank*
* Twitter: *has_twitter*, *verified*, *created_at*, *has_location*, *url_match*, *description*, *counts*
* Wikipedia: *has_wiki*, *wikicontent*, *wikisummary*, *wikicategories*, *wikitoc*
* Articles: *title*, *body*

Each of these features is stored as a numpy file in **data/features/**. The 1st column corresponds to the **source_url_processed** to ensure alignment with the corpus, and the last two columns correspond to the factuality and bias labels.

### 2.3 Classification
To run the classification script, use a command-line argument of the following format:

```
python3 classification.py --task [0] --features [1]
```
where
* [0] refers to the prediction task: fact, bias or bias3way (an aggregation of bias to a 3-point scale), and
* [1] refers to the list of features (from the list above) that will be used to train the model. features must separated by "+" signs (example: has_wiki+has_twitter+title)


### 2.4 Citation
For more details about the dataset, the features and the results, please refer to EMNLP paper:

```
@InProceedings{baly:2018:EMNLP2018,
  author    = {Baly, Ramy  and  Karadzhov, Georgi  and  Alexandrov, Dimitar and  Glass, James  and  Nakov, Preslav},
  title     = {Predicting Factuality of Reporting and Bias of News Media Sources},  
  booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language Processing},
  series = {EMNLP~'18},
  NOmonth     = {November},
  year      = {2018},
  address   = {Brussels, Belgium},
  NOpublisher = {Association for Computational Linguistics}
}
```

## 3 Results
After comparison, using: Logistic regression + Lasso regularization can reach similar F1 score and prediction by just analyzing:
* Wikipedia related variables
* Twitter account verification
* Article title

## 4 Business Use
It is more cost efficient and faster computation model to predict article’s bias and factuality. 
So this can solve Real-time source prediction function with ability to extend to all news websites. 
Potential application develop for small startup whose information sources are limited and more cost-sensative, to build their websites more believable and decrease bias.


