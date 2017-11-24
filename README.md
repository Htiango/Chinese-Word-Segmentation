# Chinese-Word-Segmentation
Implement a Chinese Word Segmentation Method based on paper [A Realistic and Robust Model for Chinese Word Segmentation](http://www.aclweb.org/anthology/O08-1009). Extract features and labels from training data and use LogicRegression to train a classification model. And then do prediction on testing set and get the accuracy of this segmentation method.

## Introduction
The intuition behind this technique is to look at 4 consecutive (non-space) characters, or 4-grams, along with a learned model to guess whether or not there should be a word separation between the middle two characters. 

Suppose we have a 4-gram of Chinese characters that we’ll represent by the letters ABCD. Using this sequence, we can define the feature vector x and label y as follows:

+ x = (AB,B,BC,C,CD); y=1 if BC can be separated, y=0 if not.

I set each 1-gram and 2-gram to be a dimension in feature vector. So each feature vector's dimension should be the same as the size of corpus I build. Then I use sparse matrix to represent each feature vector because only 5 dimensions are non-zero. 

The way generate a sparse is shown as following:

```python
row = np.array([0, 2, 3, 1, 0])
col = np.array([0, 10, 20, 30, 40])
data = np.array([1, 1, 1, 1, 1])
mtx = sparse.csr_matrix((data, (row, col)), shape=(4, 50))
```

In this way, the training feature X_train is a *(n_samples, corpus_size)* sparse matrix, while the labeling Y_train is a *n_samples* array. 

Here I use LogicRegression to train the model and get a 89.542% accuracy of segmentation on test set, covering 99.79% of the testing data.


## Process
The process can be listed as follows:

+ Generate Corpus
    + Read from training file, convert to utf-8 and get a list of each convertible line. (Ignore the *UnicodeDecodeError*)
    + Use the training data to generate a corpus of all the 1-gram and 2-gram words. (Remove spaces) 
    +  In order to handle the first 2 and last 2 words, here I use a tricky method: Remain the '\n' in the end and add a '\t' in the beginning of each line. 
+ Get the training features and labels
    + Remove all the separate mark (spaces) and get a list of indexes where there should be a separate mark behind. 
    + Use the index list to set the label.
    + generate the row (represent index in n_sample) and col (represent the index in corpus). In order to speed up feature generating, here I use list to contain the row and col. 
    + Use the method introduced above to generate a sparse matrix, get the feature X_train. Also get the labels Y_train
    + Save features and their relative labels into files. Also save the corpus into a file.
+ Train a LogicRegression Model
    + Load features and their relative labels from files, use the *LogisticRegression* from *sklearn.linear_model*
    + Save the model into file
+ Predict on testing set
    + Load model, corpus from files
    + Read from testing file, convert to utf-8 and get a list of each convertible line.
    + Use the same method as the training stage to generate testing features and the ground-truth labels. 
        + KeyError will be raised when test file has some 1-gram and 2-gram not in corpus. Most of errors come from 2-gram words and I can handle them by using a 1-gram to replace the 2-gram. However, I can't handle the error when 1-gram not in corpus. In the program, I just skip the lines where there are some 1-grams not recorded by corpus. (Fortunately 1-gram missing is very rare and we are able to handle them manually)
    + Use the model we trained to get the predicted model, compare with the ground-truth and get the accuracy.

## Structure
In 'main/main.py', use 3 functions to finish all the processes listed above:

+ **get_feature**: generate the corpus, training features/labels and save them to files.
+ **get_model**: train the LR model and save
+ **get_prediction**: get the accuracy and print to the console

The training and testing data files are in the 'data/' directory. 

The saving files of training features and labels, the corpus, the model are in the 'output/' directory.  

## Result
From the whole 1398 lines in the testing file, there are 3 lines we can't handle. Because they have the characters '.' and '洴' not showing up in the training data set. 

After I ignore the 1-gram errors, the accuracy of segmentation is 89.542% (15446/17250). 

If I ignore both the 1-gram and 2-gram errors, the accuracy can rise up to 91.266% (7795/8541). However, the coverage is less than 50% and it's not a good idea to ignore the 2-gram errors. 

*The way to handle the 2-gram error is shown in Process*


## DataSet
The training data is in a file named training.txt. This file contains 745,817 lines of segmented Chinese words; each word is separated by two space characters. While the testing file contains 1398 lines of segmented Chinese words. Both the training and test file is encode in *big5hkscs*. 

While reading from file, I have to ignore the *UnicodeDecodeError* since there exist some unknown encode type.

## Environment
Python 3.6.2

Python packages:

+ numpy (1.13.3)
+ scipy (1.0.0)
+ scikit-learn (0.19.1)


