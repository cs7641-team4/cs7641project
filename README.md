
## **Authors: Ryan Cooper, Cuong Nguyen, Evan Downs, Rhythm M. Syed, Hamna Khan**

## **Introduction**
### **Introduction and Background:**
The current review process of submissions to academic conferences is a manual task, typically done by fellow academics and requires long hours of reading. Initial pre-screening of papers must be done to determine if a paper’s content is relevant to a particular conference’s list of relevant topics. For this project, we are interested in automating this screening process to expedite the paper selection procedure, such that relevant papers receive priority for further review by the committee and that papers are sent to researchers with relevant expertise on the paper’s topic.

### **Problem Definition:**
We will develop this work in two phases of supervised and unsupervised learning. For Supervised Learning we can define the problem: Given an abstract X, perform a classification from Y labels containing the list of relevant topics and a category of not-relevant. For Unsupervised, generate K clusters that represent topics found in a historical set of documents that were accepted to a particular conference in the past. Given this unsupervised model and a new input text, use an embedding-based  similarity score to determine the basis of acceptance. We propose directly comparing supervised and unsupervised learning in order to solve the task of automating conference paper screening.

## **Methods**
### **Data Collection:**

A major part of our midterm progress has been devoted to developing our own custom dataset of research papers from specific conferences. With this effort, we focused on collecting papers specifically from the EMNLP conference. We chose EMNLP over other conferences because of the focus on NLP related topics. Had we chosen a conference such as ICML which focuses more on a broad variety of ML topics, the separability between topics would become much blurrier and therefore harder to learn. With the EMNLP conference, we collected the topics of interest from previous years and reduced the number of classifications down to 11 classes including: Linguistics, Text Generation, Sentiment Analysis, ML for NLP, Q&A, NLP Applications, Social Science, Information Extraction, Speech, Resources and Evaluation, and Negative Samples. We provide further detail about the dataset in the Appendix Section. Given these classes, we hand labeled datasets of papers from EMNLP 2020 and 2019, choosing 2 labels that best represent a paper’s abstract. To collect the papers for both year’s conferences, we developed a scraper that collected data from aclweb.org such as title, abstract, author list, and pdf url. To generate the negative samples for our dataset, we utilized the arXiv dataset and collected papers from all topics excluding Information Retrieval, Computational Linguistics, Computers and Society, Social and Information Networks, and General Literature. Altogether, our final dataset contained 1450 samples, 700 of which are negative samples. Given this dataset, the next step is to generate word embeddings and train our models (supervised and unsupervised).

### **Data Preprocessing**
The dataset's explanatory variable of interest is the given abstract and the response variable is the label indicating the given abstract's class. Before featurizing the abstract's text, extensive cleaning is necessary. While the type of text cleaning can differ between models, all the techniques will be introduced here and the use cases will be explained in the following sections. One method is ensuring that all the text within each abstract only includes alphabetic characters and spaces. All punctuation and special characters are removed in order to reduce the complexity of the representation. For further reducing the size of the abstracts, stop words such as the, an, our, etc. are removed from each abstract. Lemmatization is also used in some cases to reduce words to their root in order to be able to map similar words to the same representation. Tagging is also used when it is necessary to only focus on a word's part of speech. This technique can be helpful in reducing the size of each abstract's representation.  
- Cleaning/Stopword
- Stemming/Lemmatizer
- Implementation

### **Language Models**
#### *BoW*
        
Bag of words is a classical approach to creating sparse feature vectors of documents by generating counts of each word in the sentence, where each index in the vector represents a word in the corpus and the value at this index is the count in the document. Since each vector's dimensionality is equal to the size of the vocabulary in the corpus, these vectors can be quite unruly to work with and also suffer from the fact that distance in the feature space has no tangible meaning.

#### *Trained W2V*
Word2vec models are helpful in capturing a representation of a word based upon the context that the word is used in. A famous linguist Firth explains the motivation for this technique by stating "You shall know a word by the company it keeps". There are many pre-trained word2vec models that can be utilized, but they are often biased to the topics in the corpus they are trained on. In order to ensure words have domain specific meaning to the corpus used, a word2vec model trained on the corpus may provide a more clear representation of corpus specific terms. A skipgram word2vec model built using negative sampling loss and optimized with stochastic gradient descent is implemented on the entire corpus specific to the dataset used.

#### *GloVe*
GloVe is a set of word embeddings trained by researchers at Stanford in 2014 using Wikipedia, Twitter, and a Common Crawl dataset [4]. These vectors are trained by aggregating global word-word co-occurrence statistics from a corpus and the resulting representations show linear substructures in the semantic space, allowing for analogies such as: man to woman as to king and queen, to be represented in the semantic space. GloVe is a fairly standard pre-trained Word2Vec model and serves as a good starting place for many NLP classification tasks.

#### *Fine-tuned GloVe*
To further improve upon the pre-trained word embeddings, we thought that possibly fine-tuning GloVe with our corpus could prove valuable as it would include out-of-vocabulary words, and further align the feature space with the co-occurrences in the training corpus. Utilizing a package called mittens, fine-tuning GloVe embeddings becomes easy and generating this language model aligns the feature space to be task-specific.


#### *LDA*
LDA (Latent Dirichlet Allocation) is an unsupervised learning method that is based on treating each document as a mixture of topics and each topic as a mixture of words. At a high level the method updates the probability of a word being assigned to a topic and a document being assigned to a topic.  Two required hyperparameters for this model are alpha which specifies the level of topic mixtures that should be present in a documents' classification, and beta which represents the amount of words that should be used in a topic. The number of topics also needs to be specified before running the model. This model is implemented over the text abstracts to see if topics are distinguishable. Since it is known that there are 11 labels, LDA can be tested to see if it can separate out the document into 11 classes. This method can validate how distinguishable 11 classes are. LDA can also be used for dimensionality reduction. Since each abstract is assigned a probability of belonging to each class, the distribution over the number of classes can represent an engineered feature of the abstract. The dimensionality of this feature is the number of topics chosen.

## **Supervised Learning**
### *A. Task Introduction*
The formulation of supervised learning on our dataset is as follows: Given N abstracts (N = 1450) (which is featurized in the previous steps as an N-dimensional embedding), we build model that could classify future paper abstracts as belonging to either one of the positive NLP topics (1-10) or is a negative sample (11). We split the data 80-20 into a training set (N = 1160) and a test set (N = 290), stratifying by label classes so that there are similar amounts of samples per class between the training set and test set. 
### **B. Supervised Learning Models**
For this task, we consider the following supervised learning models:
#### *SVM*
Support Vector Machine is a linear model that can be utilized for both binary and multiclass classification, in addition to regression. It differentiates classes with a line or hyperplane and is effective in high dimensional spaces.   
#### *Gaussian Process*
Gaussian processes are utilized to solve both regression and classification problems. They are defined by the Gaussian distribution, can be molded by the covariance matrix, and output probabilistic predictions (soft classification).
#### *Decision Tree*
Decision trees are models utilized for both classification and regression. They primarily learn from data features in order to create decision rules for future prediction.  
#### *Random Forest*
Random Forest is a type of bagging ensemble learning method that can be applied to both classification and regression problems. It works by fitting multiple decision tree classifiers on different samples of the dataset and averages the outputs to improve predictions.
#### *Neural Networks (MLP)*
Neural networks consist of connected neurons organized into layers, featuring non-linear activation functions. They are trained using backpropagation techniques and can distinguish non-linearly separable data.
#### *AdaBoost*
AdaBoost is a boosting ensemble method that trains weak individual decision tree models sequentially in order to adapt to misclassified data points. More accurate classifiers are given larger weight to contribute to the output prediction, thus capitalizing the accuracy of the overall system. 
#### *Naive Bayes*
Naive Bayes classifiers work by applying Bayes’ theorem with the assumption that there exists conditional independence between features of the data. Because of this “naive” assumption, these classifiers are fast and work well with high dimensional data.
#### Bagging 
Bagging is a technique utilized to reduce overfitting of data. It involves fitting multiple classifiers on various subsets of the dataset and building an overall prediction from a combination of the outputs of these individual models.
### **C. Tasks**
For each of these models detailed in section B, we use grid search with 10-fold cross-validation in order to find the best-performing hyperparameter combination. We report the hyperparameters considered for each model in the results section below. We perform supervised learning for the following 3 tasks:
- Binary Classification (BC): We treat all 10 positive classes as a single label, and perform classification with that label (1) and negative samples (0) as the other label.
- Multiclass Classification - Positive Only (MC-PO): We discarded all samples that was labeled as a negative sample, and performed classification on the remaining data that was labeled as one of the NLP topics (1 - 10)
- Multiclass Classification - Full (MC-F): We perform classification on the dataset without any modifications (1 - 11)

### **D. Balancing the dataset**
Because our dataset is imbalanced, especially among the positive classes, we experimented with SMOTE as a method to fix this imbalance. In summary, SMOTE oversamples classes that are in the minority by taking samples from that class and consider their k nearest neighbors in the feature space. Synthetic data points are then created by randomly taking a vector between one of those k neighbors and the current data point, multiplying it by a factor between 0 and 1, and adding it to the original data point. We executed sklearn’s SMOTE implementation on the dataset and re-ran the MC-PO task for further analysis

## **Results**
### **Unsupervised Learning**
#### **A. LDA Topic modeling**

The visualizations for the topic distributions of LDA with 3 topics and for LDA with 11 topics is seen in the figures below. 11 topics are chosen since it is known a-priori that there are 11 topics within the abstracts. The visualization for 11 topics does show that there is some overlap between words assigned to various topics. This observation shows that 11 topics may be too difficult to distinguish effectively in the dataset. 3 topics is used since it appears visually that there are 3 clear distinctions within the topics. This observation is reinforced through calculating coherence scores for each LDA model. The 11 topic model has a coherence score of 0.3349 and the 3 topic model has a coherence score of 0.3482. While they are similar, the 3 topic model's coherence score is slightly higher showing that the 3 topics are more human readable than the 11 topics generated.

![](https://drive.google.com/uc?export=view&id=183yOC0g_Dg37KwQZQFr9Q4Lzj4B0GwSe)

![](https://drive.google.com/uc?export=view&id=1011KtcF0oTQBH3t2RZzZVUazxeU7DhrO)


#### **B. LDA Classification**

LDA's generated 11 topic distribution over each abstract is tested briefly as an input for an SVM classifier. It is shown to reach a test set accuracy of 55.55% for multi-class labeling and a test accuracy of 86.50% for binary class labeling. While these scores are not as strong as the supervised classification using GloVe embeddings, they still hold up well in comparison for how simple their representations are. Since each feature represents an abstracts' distribution over topics, the engineered features can be more explainable than GloVe embeddings. Further exploration and optimization of this technique will be performed in the future.

#### **C. Language Model Visualization**
The word2vec model trained on the corpus is found to capture the contextual meaning of the words well. The word plot in the figure below shows a two dimensional representation of the trained 13 dimensional vectors. While some contextual meaning is lost in the dimensionality reduction, the plot still shows similar words grouped closer together and more unrelated words further apart. The words cache, memory, time, and latency are expected to be located around similar words contextually so it makes sense they are closer. Language, learning, bert, and machine are all heavily related to machine learning terminology so it makes sense that they are grouped closer together as well. Paper means something different from the other words so it is farther away which agrees with common sense. A more robust comparison involving all the dimensions is to use cosine similarity to compare two vectors. The vectors for cache and memory achieve a cosine similarity score  of 0.99 which is close to one indicating that the vectors are in a similar plane. However, time and paper have a cosine similarity score of 0.56 showing that they are not as related in the same dimension space.

![](https://drive.google.com/uc?export=view&id=1cSRrIoN58694oNvjAiV3NcMgO07vNGNN)


These 2d and 3d visualizations are intended to represent a projection from the high dimensional semantic space, colored by binary class to illustrate the separability in the feature space. We find that the space is extremely separable for binary classification, but for multiclass, becomes more convoluted.


- *BOW*

![](https://drive.google.com/uc?export=view&id=1lJqAQFYRhFsIAJeRMmBDnsvvQMiSNcOn)

- *Trained W2V*

![](https://drive.google.com/uc?export=view&id=13R9mwsisAtxDAIh3RNXDzIHLa-5prnmj)

- *GloVe*

![](https://drive.google.com/uc?export=view&id=1kz4beIDDnruLKneR9PJEoj1INjX5ALt1)

- *GloVe Multi Class*

![](https://drive.google.com/uc?export=view&id=1ytmPj5cFraiIlVb9w715uJv0lVeu3q6Q)

- *Finetuned GloVe*

![](https://drive.google.com/uc?export=view&id=14xUpW-Bl9IjRnCx7UUVMb-LsywijpJn7)

#### **D. Language Model Comparison**

In order to understand the importance of the feature space we choose from various language models, we look at a baseline classification model, support vector machine for classification with rbf kernel, and compare the train and test (33% of the dataset) accuracies.

- *MC-F (Multiclass Classification - Full)*

| Language Model | Training Accuracy | Testing Accuracy |
|---|---|---|
| BOW | 0.84462 | 0.22177 |
| Trained Word2Vec | 0.39047 | 0.25603 |
| GloVe | 0.88333 | 0.37198 |
| Fine-tuned GloVe | 0.87857 | 0.37681 |


- *BC (Binary Classification)*

| Language Model   | Training Accuracy | Testing Accuracy |
|------------------|-------------------|------------------|
| BOW              | 0.98249           | 0.77870          |
| Trained Word2Vec | 0.91324           | 0.93475          |
| GloVe            | 0.98650           | 0.94292          |
| Fine-tuned GloVe | 0.98650           | 0.94063          |

We see that GloVe-based models seem to provide the most separability to the data in both problems. 

## **Supervised Learning**
#### **Model accuracies**
Beginning with the BC task, which encodes whether or not a paper is included in the conference, we ran a logistic regression model yielding high prediction scores on training data and testing data. Moving forward to MC-F task, we experimented with multiple models and utilized grid search to fine-tune hyperparameters for the best fit. Our fitting of multiple models, finetuning of parameters, and prediction accuracies can be summarized in the tables below:

- *MC-F (Multiclass Classification - Full)*

|        Model        | Hyperparameters                                                        | Test Accuracy      |
|:-------------------:|------------------------------------------------------------------------|--------------------|
| SVM                 | {'kernel': ['linear', 'rbf'], 'C':[0.01, 0.1, 1], 'gamma': ['scale']}  | 0.6206896551724138 |
| Gaussian Process    | {'kernel': [1.0 * RBF(1.0)]}                                           | 0.6206896551724138 |
| Logistic Regression | {'random_state': [0], 'max_iter': [5000]}                              | 0.6                |
| Decision Tree       | {'max_depth': [5]}                                                     | 0.496551724137931  |
| Random Forest       | {'max_depth': [5], 'n_estimators': [10], 'max_features': [1]}          | 0.4827586206896552 |
| MLP                 | {'alpha': [1], 'max_iter': [5000]}                                     | 0.6206896551724138 |
| AdaBoost            | {'n_estimators': [100]}                                                | 0.4793103448275862 |
| Naive Bayes         | {}                                                                     | 0.5724137931034483 |
| Bagging Classifier  | {'base_estimator': [SVC()], 'n_estimators': [10], 'random_state': [0]} | 0.4827586206896552 |





- *BC (Binary Classification)*

|        Model        | Hyperparameters                                                        | Test Accuracy       |
|:-------------------:|------------------------------------------------------------------------|---------------------|
| SVM                 | {'kernel': ['linear', 'rbf'], 'C':[0.01, 0.1, 1], 'gamma': ['scale']}  | 0.9137931034482759  |
| Gaussian Process    | {'kernel': [1.0 * RBF(1.0)]}                                           | 0.9379310344827586  |
| Logistic Regression | {'random_state': [0], 'max_iter': [5000]}                              | 0.9172413793103448 |
| Decision Tree       | {'max_depth': [5]}                                                     | 0.7689655172413793 |
| Random Forest       | {'max_depth': [5], 'n_estimators': [10], 'max_features': [1]}          | 0.8862068965517241               |
| MLP                 | {'alpha': [1], 'max_iter': [5000]}                                     | 0.9275862068965517  |
| AdaBoost            | {'n_estimators': [100]}                                                | 0.8931034482758621 |
| Naive Bayes         | {}                                                                     | 0.8482758620689655 |
| Bagging Classifier  | {'base_estimator': [SVC()], 'n_estimators': [10], 'random_state': [0]} | 0.9206896551724137                |


- *MC-PO (Multiclass Classification - Positive Only)*

|        Model        | Hyperparameters                                                        | Test Accuracy       |
|:-------------------:|------------------------------------------------------------------------|---------------------|
| SVM                 | {'kernel': ['linear', 'rbf'], 'C':[0.01, 0.1, 1], 'gamma': ['scale']}  | 0.3933333333333333  |
| Gaussian Process    | {'kernel': [1.0 * RBF(1.0)]}                                           | 0.4066666666666667  |
| Logistic Regression | {'random_state': [0], 'max_iter': [5000]}                              | 0.35333333333333333 |
| Decision Tree       | {'max_depth': [5]}                                                     | 0.24666666666666667 |
| Random Forest       | {'max_depth': [5], 'n_estimators': [10], 'max_features': [1]}          | 0.26                |
| MLP                 | {'alpha': [1], 'max_iter': [5000]}                                     | 0.4066666666666667  |
| AdaBoost            | {'n_estimators': [100]}                                                | 0.19333333333333333 |
| Naive Bayes         | {}                                                                     | 0.3933333333333333  |
| Bagging Classifier  | {'base_estimator': [SVC()], 'n_estimators': [10], 'random_state': [0]} | 0.22                |


## **Discussion**
### **Language Models**

In the latter half of this project, we hope to explore transformer based language models such as BERT [1] and SciBERT [5]. Research has shown that the context attention mechanisms in these models lead to a richer feature space and we hope this will allow for the creation of better classification models.

### **Unsupervised Learning**

The next steps for Unsupervised learning will be finding additional ways to use Unsupervised learning models' output as engineered features for the Supervised learning models. This task will be similar to the example discussed previously involving LDA's output for an input into an SVM classifier. KMeans could be another Unsupervised learning model to test in addition to LDA. Work will also be done to see if it is possible to automatically assign a label to an unsupervised learning model's groups or clusters. This semi-supervised automatic labeling would be helpful in providing interpretability to unsupervised learning groupings' for the dataset. TF-IDF and word embedding summarization combined with distance score metrics will be initally tested for this task.

### **Supervised Learning**
 For the final report, we plan to explore more hyperparameters for each of the classifiers used in the midterm report. We also plan to balance the dataset between the positive classes, through a combination of additional data gathering and exploration with oversampling techniques. Finally, we plan to build a more sophisticated neural network classification model using Pytorch and compare results to traditional machine learning models.

## References
 
[1] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, <i>arXiv e-prints</i>, 2018.
 
[2] Minaee, S., Kalchbrenner, N., Cambria, E., Nikzad, N., Chenaghlu, M., and Gao, J., “Deep Learning Based Text Classification: A Comprehensive Review”, arXiv e-prints, 2020.
[3] Jelodar, H., “Latent Dirichlet Allocation (LDA) and Topic modeling: models, applications, a survey”, arXiv e-prints, 2017.
[4] Pennington, Jeffrey & Socher, Richard & Manning, Christopher. (2014). GloVe: Global Vectors for Word Representation. EMNLP. 14. 1532-1543. 10.3115/v1/D14-1162. 
[5] Iz Beltagy, Kyle Lo, & Arman Cohan (2019). SciBERT: Pretrained Language Model for Scientific Text. In EMNLP.
[6] https://github.com/roamanalytics/mittens

## **Appendix**
### **A. Data Label Frequencies**

| 1  | 2   | 3  | 4   | 5  | 6  | 7  | 8  | 9  | 10 | 11  |
|----|-----|----|-----|----|----|----|----|----|----|-----|
| 81 | 165 | 36 | 139 | 65 | 56 | 28 | 71 | 11 | 98 | 700 |

### **B. NLP Topics** 

| Class | Topic Name | Topic Description | | |
|---------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|---|---|
| Class1  | Linguistics              | Syntax, Phonology, Morphology and Word Segmentation, Linguistic Theories, Cognitive Modeling and Psycholinguistics, Discourse and Pragmatics |   |   |
| Class2  | Text_Generation          | Generation & Summarization & Machine Translation and Multilinguality,                                                                        |   |   |
| Class3  | Sentiment_Analysis       | Sentiment Analysis, Stylistic Analysis, and Argument Mining                                                                                  |   |   |
| Class4  | ML_for_NLP               | Machine Learning for NLP and Interpretability and Analysis of Models for NLP                                                                 |   |   |
| Class5  | Q&A                      | Dialogue, Speech and Interactive Systems + Question Answering                                                                                |   |   |
| Class6  | NLP Applications         | NLP Applications, Language Grounding to Vision, Robotics and Beyond                                                                          |   |   |
| Class7  | Social_Science           | Computational Social Science and Social Media                                                                                                |   |   |
| Class8  | Information Extraction   | Information Extraction & Information Retrieval and Text Mining                                                                               |   |   |
| Class9  | Speech                   | Speech and Multimodality                                                                                                                     |   |   |
| Class10 | Resources and Evaluation | Resources and Evaluation                                                                                                                     |   |   |
| Class11 | Negative Samples         |                                                                                                                                              |   |   |
