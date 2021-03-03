#### Authors Ryan Cooper, Cuong Nguyen, Evan Downs, Rhythm M. Syed, Hamna Khan

### Presentation Video:

https://drive.google.com/file/d/18xwnCenKkasaJt0dNkja31UJ53BJolkV/view

### Introduction/Background  

The approval process for academic conference papers is a manual task requiring hours of reading. Initial pre-screening of papers must be done to determine a paper’s relevancy to a conference. This project focuses on automating the pre-screening process, such that relevant papers receive further review while irrelevant papers are disregarded.

### Problem definition  

For supervised learning: Given an abstract X, perform a classification for Y relevant topics and one not-relevant. For unsupervised, generate K clusters representing topics found in documents accepted to a conference. Given this unsupervised model and a new input text, use a similarity score to determine acceptance threshold.

### Methods  

- Data Collection  

    The basis of our data collection will involve developing our own datasets with samples drawn from [arXiv dataset](https://www.kaggle.com/Cornell-University/arxiv), [semantic scholar corpus](http://s2-public-api-prod.us-west-2.elasticbeanstalk.com/corpus/), and [kaggle](https://www.kaggle.com/nikhilmittal/research-paper-abstracts?select=data_input.csv). The supervised learning dataset will require manual text labeling with Y labels for a conference. Other methods will involve processing text and creating semantic word representations.

- Data Preprocessing  

    A variety of language models will be applied to featurize the data. Processing techniques such as bag-of-words, vector-based word representations, and BERT and its variants will be explored. [1]

- Supervised Learning 

    Simple supervised learning methods can be used on engineered features extracted from the abstracts. These engineered features could be placed in tabular structures which allow for basic implementations of logistic regressions or support vector machines given an input of 1 x F and a multi class output of 1 x C. F is the number of engineered features and C is the number of key word classes.
 
    While engineered feature methods can be useful allowing for simple and interpretable models, the extracted features lose the contextual meaning of the text. Deep learning methods such as RNNs trained on word vectors can be applied to capture contextual information [2].

- Unsupervised Learning  

    A variety of clustering techniques will be applied in extracting useful topic clusters out of a given conference corpus. With each of the found clusters, similarity distances to topical centroids will be used to determine acceptance. Since k, the number of clusters, is not necessarily defined for a given conference, algorithms will be chosen where k is not a parameter to the model, such as hierarchical clustering, or some density-based approaches like DBSCAN, or HDBSCAN [3].

### Potential results  

We expect that our methodology will reveal an automated approach to accepting papers by using language models to generate features, and extract useful topical similarities to an established corpus. Metrics like Hamming Loss, F1-score, recall@k will be useful for dealing with multilabel classification. In unsupervised learning methods, since we will have the actual labelled data available to us, we can directly evaluate the quality of topic cluster.


### Discussion  

One of the major challenges for the proposed project will be collecting the data samples to use in the supervised learning approaches. It can be difficult to identify paper's abstracts that list keywords that match the conference paper keywords. Also care needs to be taken in identifying negative samples. The samples need to be distinguishable from the positive samples, but they shouldn't be completely irrelevant so that the trained model is unbiased.
 

### References  

*[1] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, <i>arXiv e-prints</i>, 2018.*
 
*[2] Minaee, S., Kalchbrenner, N., Cambria, E., Nikzad, N., Chenaghlu, M., and Gao, J., “Deep Learning Based Text Classification: A Comprehensive Review”, arXiv e-prints, 2020.*  

*[3] Jelodar, H., “Latent Dirichlet Allocation (LDA) and Topic modeling: models, applications, a survey”, arXiv e-prints, 2017.*




