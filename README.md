### Authors Ryan Cooper, Cuong Nguyen, Evan Downs, Rhythm M. Syed, Hamna Khan

# Introduction/Background  

The current review process of submissions to academic conferences is a manual task, typically done by fellow academics and requires long hours of reading. Initial pre-screening of papers must be done to determine if a paper’s content is relevant to a particular conference’s list of relevant topics. For this project, we are interested in automating this screening process to expedite the paper selection procedure, such that relevant papers receive priority for further review by the committee and that papers are sent to researchers with relevant expertise on the paper’s topic.

We will develop this work in two phases of supervised and unsupervised learning. For Supervised Learning we can define the problem: Given an abstract X, perform a classification from Y labels containing the list of relevant topics and a category of not-relevant. For Unsupervised, generate K clusters that represent topics found in a historical set of documents that were accepted to a particular conference in the past. Given this unsupervised model and a new input text, use an embedding-based  similarity score to determine the basis of acceptance. We propose directly comparing supervised and unsupervised learning in order to solve the task of automating conference paper screening.


# Problem definition  

# Methods  

## Data Collection  

The basis of our data collection will involve developing our own datasets with samples drawn from arXiv dataset, semantic scholar corpus, and kaggle. To develop a supervised model, we will be manually labeling our dataset with Y labels for a specific conference. All other data analysis will involve processing text and creating semantic word representations in the processing step.

## Data Preprocessing  

In order to work with the text data, we will apply a variety of language models to the data in order to featurize it. We will explore and compare the various processing techniques, from the classic bag-of-words model (BoW), vector-based word representations such as word2vec, to the cutting edge BERT and its many variants[1].

## Supervised Learning 

For supervised learning, our planned approach to the problem is as follows: Take in an abstract of a paper and classify what conference key words are most present in the abstract. Simple supervised learning methods can be used on engineered features that can be extracted from the abstract. These engineered features could be placed in a tabular structure which would allow for basic implementations of logistic regressions or support vector machines given an input of 1 x F and a multi class output of 1 x C where F is the number of engineered features and C is the number of key word classes.
 
While engineered feature methods can be useful allowing for simple and interpretable models, the extracted features lose the contextual nature of the text within the abstract. In order to capture the contextual nature of the words, deep learning methods such as RNNs trained on word vector representations of the abstracts will be tested [2].

## Unsupervised Learning  

We plan to benchmark a variety of clustering techniques for extracting useful topic clusters out of a given conference corpus. With each of the found clusters, we plan to use similarity distance to topical centroids as a method to determine acceptance. Since k, the number of clusters, is not necessarily defined for a given conference, we will focus on algorithms where k is not a parameter to the model, such as hierarchical clustering, or some density based approaches like DBSCAN, or HDBSCAN [3].

# Potential results  

We expect that our methodology will reveal an automated approach to accepting papers by using language models to generate features, and extract useful topical similarities to an established corpus. Metrics like Hamming Loss, F1-score, recall@k will be useful for dealing with multilabel classification. For unsupervised learning, since we will have the actual labelled data available to us, we can directly evaluate the quality of topic cluster.


# Discussion  

One of the major challenges for the proposed project will be collecting the data samples to use in the supervised learning approaches. Sometimes it is difficult to identify paper's abstracts that actually list their keywords and that have keywords that match the keywords from the conference paper that they were submitted to. Also care needs to be taken in identifying negative samples. The samples need to be different enough from the positive samples that the model can distinguish them, but they also shouldn't be the complete opposite of the positive samples or else the model will have a biased performance.
 


# References  

*[1] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, <i>arXiv e-prints</i>, 2018.*
 
*[2] Minaee, S., Kalchbrenner, N., Cambria, E., Nikzad, N., Chenaghlu, M., and Gao, J., “Deep Learning Based Text Classification: A Comprehensive Review”, arXiv e-prints, 2020.*  

*[3] Jelodar, H., “Latent Dirichlet Allocation (LDA) and Topic modeling: models, applications, a survey”, arXiv e-prints, 2017.*




