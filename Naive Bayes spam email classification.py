# -*- coding: utf-8 -*-

# -- Sheet --

# # DAT405 Introduction to Data Science and AI 
# ## Assignment 4: Spam classification using Naïve Bayes 


# ### Time
# - David Arvidsson 19941029-1414, MPDSC, ardavid@student.chalmers.se 16 hours
# 
# - Johan Hegardt 19970714-6230, MPDSC, johan.hegardt@gmail.com 16 hours


#Download and extract data
#!wget https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2
#!wget https://spamassassin.apache.org/old/publiccorpus/20021010_hard_ham.tar.bz2
#!wget https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2
#!tar -xjf 20021010_easy_ham.tar.bz2
#!tar -xjf 20021010_hard_ham.tar.bz2
#!tar -xjf 20021010_spam.tar.bz2

# *The* data is now in the three folders `easy_ham`, `hard_ham`, and `spam`.


#!ls -lah

# ### 1. Preprocessing: 
# 1.	Note that the email files contain a lot of extra information, besides the actual message. Ignore that and run on the entire text. 
# 2.	We don’t want to train and test on the same data. Split the spam and the ham datasets in a training set and a test set. (`hamtrain`, `spamtrain`, `hamtest`, and `spamtest`) **0.5p**


# Task 1: Convert all emails in the three datasets to strings.
import os
import codecs
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Creating lists for emails and their labels
emails = []
email_labels = []

# Putting the emails into the list as strings, adding labels
for email in os.listdir("easy_ham"):
    with codecs.open("easy_ham/"+email, 'r', encoding='utf-8', errors='ignore') as fdata:
        emails.append(fdata.read())
        email_labels.append(0)

# Putting the emails into the list as strings, adding labels
for email in os.listdir("spam"):
    with codecs.open("spam/"+email, 'r', encoding='utf-8', errors='ignore') as fdata:
        emails.append(fdata.read())
        email_labels.append(1)

# Task 2: Vectorize the words in the email strings and splitting the dataset into test and train data. 
# Stratification used on labels to keep proportions
# emails_vector containing easy_ham and spam
count_vectorizer = CountVectorizer()
emails_vector = count_vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(emails_vector, email_labels, random_state=1, test_size=0.25, stratify = email_labels)

# Putting the emails into the list as strings, adding labels
for email in os.listdir("hard_ham"):
    with codecs.open("hard_ham/"+email, 'r', encoding='utf-8', errors='ignore') as fdata:
        emails.append(fdata.read())
        email_labels.append(2)

# Task 2: Vectorize the words in the email strings and splitting the dataset into test and train data. 
# Stratification used on labels to keep proportions
# emails_vector2 containing hard_ham, easy_ham and spam
count_vectorizer = CountVectorizer()
emails_vector2 = count_vectorizer.fit_transform(emails)

X_train2, X_test2, y_train2, y_test2 = train_test_split(emails_vector2, email_labels, random_state=1, test_size=0.25, stratify = email_labels)

# Method for converting both hard_ham and easy_ham to 0. Here after, spam = 1 and ham = 0.
def transform(y_train,y_test):
    for i in range(0,len(y_train)):
        if y_train[i] == 2:
            y_train[i] = 0

    for i in range(0,len(y_test)):
        if y_test[i] == 2:
            y_test[i] = 0

# Transform label data to 2D
transform(y_train,y_test)
transform(y_train2,y_test2)

# ### 2. Write a Python program that: 
# 1.	Uses four datasets (`hamtrain`, `spamtrain`, `hamtest`, and `spamtest`) 
# 2.	Trains a Naïve Bayes classifier (e.g. Sklearn) on `hamtrain` and `spamtrain`, that classifies the test sets and reports True Positive and True Negative rates on the `hamtest` and `spamtest` datasets. You can use `CountVectorizer` to transform the email texts into vectors. Please note that there are different types of Naïve Bayes Classifier in SKlearn ([Documentation here](https://scikit-learn.org/stable/modules/naive_bayes.html)). Test two of these classifiers that are well suited for this problem
#     - Multinomial Naive Bayes  
#     - Bernoulli Naive Bayes. 


# Necessary imports
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# Method for Multinomial Naive Bayes classification
def MNBClassification(X_train,y_train,X_test,y_test):
    
    # Create a multinomial Naive Bayes model
    mnb = MultinomialNB()

    #Train the model using the training sets
    mnb.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = mnb.predict(X_test)

    # Create a confusion matrix to visualize predictions and true negative/true positive absolute numbers
    titles_options = [("Confusion matrix", None)]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(mnb, X_test, y_test, display_labels=["non-spam","spam"], cmap=plt.cm.Blues, normalize=normalize)   
        plt.grid(False) 
        disp.ax_.set_title(title)
    plt.grid(False)
    plt.show()

    # Find and print the true positive/true negative rates
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    tpRate = tn/(fp+tn)
    tnRate = tp/(fn+tp)
    print ("True positive rate: " + str(tpRate))
    print ("True negative rate: " + str(tnRate))

# Method for Bernoulli Naive Bayes classification
def BNBClassification(X_train,y_train,X_test,y_test):
    
    # Create a Bernoulli Naive Bayes model
    bnb = BernoulliNB()
    
    #Train the model using the training sets
    bnb.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = bnb.predict(X_test)

    # Create a confusion matrix to visualize predictions and true negative/true positive absolute numbers
    titles_options = [("Confusion matrix", None)]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(bnb, X_test, y_test, display_labels=["non-spam","spam"], cmap=plt.cm.Blues, normalize=normalize)
        plt.grid(False) 
        disp.ax_.set_title(title)
    plt.grid(False)
    plt.show()

    # Find and print the true positive/true negative rates
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    tpRate = tn/(fp+tn)
    tnRate = tp/(fn+tp)
    print ("True positive rate: " + str(tpRate))
    print ("True negative rate: " + str(tnRate))

# a) Explain how the classifiers differ. What different interpretations do they have? **1p** 
# 
# **Multinomial Naïve Bayes classifier:**
# The Multionomial classifier classify a data point based on the counts it finds of multiple features, for example the number of occurances of each word in this assignment. It does not only take into consideration if a word exist or not in the given document, but also looks at how many times each word occurs. 
# 
# The Multionomial classifier is a Naïve Bayes classifier and therefore it is based on the assumption that there is no inter-dependece between features. It is also based on the assumption that each feature in the given data point is multinomial distributed, which works well for data that can easily be turned into counts, such as word counts in a document. 
# 
# **Bernoulli Naïve Bayes classifier:**
# Bernoulli Naive Bayes classify data points based on binary values only. For example, in this assignment it checks if a given email contains a word or not, and each value in the classifier will be either 1 or 0, which represents if the word exists or not. It is a very simplified model and could be used in cases where word frequency is less important, and also suitable for large datasets problem because it is computationalluy better than other traditional algorithms.
# 
# **Differences between the two models:**
# By summarizing the two classifiers and their characteristics, it becomes clear that they are useful in different situations and to solve different types of problems. The Berneoulli Naïve Bayes classifier iterates through the words in an email and binarily provides information if a word has occured or not. That may be useful in situations where the classification is binary, if a specific word is contained in the email or not. If, for example, we would like to classify emails that are discussing the course DAT405 and emails that are not, the Berneoulli Naïve Bayes classifer would do that effictively. However, in the case of email classification it is of interest not only if specific words are detected in the emails, but also the **frequency** of their occurances. Words such as "free", "offer" or "click" might show up in a regular email once or twice without it being a problem, but if they appear 5-10 times in an email it should probably be categorized as spam. The binary Berneoulli model fails to identify those multiple occurences of specific words, whereas the Multinomial model, which expects a multinomial distribution of features instead of binary, is keeping track of the frequency of the words. That is why the Multinomial model is expected to generate a higher accuracy of spam classification in this specific task.


# ### 3. Run your program on 
# -	Spam versus easy-ham 
# -	Spam versus (hard-ham + easy-ham). 
# -   Discuss your results **2.5p** 


#Multinomial Naïve Bayes classification - Spam versus easy-ham
MNBClassification(X_train,y_train,X_test,y_test)

print("------------------------------------------")

#Multinomial Naïve Bayes classification - Spam versus (hard-ham + easy-ham)
MNBClassification(X_train2,y_train2,X_test2,y_test2)

#Bernoulli Naïve Bayes classification - Spam versus easy-ham
BNBClassification(X_train,y_train,X_test,y_test)

print("------------------------------------------")

#Bernoulli Naïve Bayes classification - Spam versus (hard-ham + easy-ham)
BNBClassification(X_train2,y_train2,X_test2,y_test2)

# **True positive rates**
# - Multinomial Naïve Bayes classification - Spam versus easy-ham: **99.7%**
# - Multinomial Naïve Bayes classification - Spam versus (hard-ham + easy-ham): **98.7%**
# - Bernoulli Naïve Bayes classification - Spam versus easy-ham: **99.4%**
# - Bernoulli Naïve Bayes classification - Spam versus (hard-ham + easy-ham): **98.1%**
# 
# **True negative rates**
# - Multinomial Naïve Bayes classification - Spam versus easy-ham: **92.8%**
# - Multinomial Naïve Bayes classification - Spam versus (hard-ham + easy-ham): **93.6%**
# - Bernoulli Naïve Bayes classification - Spam versus easy-ham: **61.6%**
# - Bernoulli Naïve Bayes classification - Spam versus (hard-ham + easy-ham): **34.4%**
# 
# The first analysis point is that, as expected, Multinomial Naïve Bayes classification generates higher accuracies measured by true positive and true negative rates than Bernoulli Naïve Bayes classification in all cases. This means that it is likely that the frequency of word occurences are of importance in spam/non-spam classification. Some words that occur more often, which the Multinomial takes into consideration while Bernoulli only binary counts the occurence of a word.
# 
# The obtained true positive and true negative rates are higher for both Multinomial and Bernoulli when considering only easy-ham rather than both easy-ham and hard-ham. The reason is most likely that there is a bigger distinction between spam and easy-ham, where the words that occur in the two types are fairly seperated and easily distinguished. From the results it is clear that both Bernoulli and Multionomial have a hard time detecting spam, and incorrectly labels about 10% of spam email as ham. 
# 
# When also considering hard-ham, both the true negative and true positve rates decreases, especially for Bernoulli Naïve Bayes classification. This effect might be explained by the fact that the characteristics of emails categorized as hard-ham are more similar to spam emails, where the words that are present in many cases are the same as the ones used in spam mails. This effect is obvious for both Multinomial and Bernoulli but slightly more devastating for Bernoulli, probably because Bernoulli only checks for word occurance and not frequency.
# 
# Worth mentioning is that in this specific case, it could be assumed that the accuracy measured by true positive rate is of higher importance than the true negative rate. This is due to that a spam email failed to be categorized as spam and therefore ending up in the regular inbox is probably more acceptable than a non-spam ending up in the spam inbox. From that perspective, the Multinomial model has a very satisfiable accuracy.


# # 4.	To avoid classification based on common and uninformative words it is common to filter these out. 
# 
# **a.** Argue why this may be useful. Try finding the words that are too common/uncommon in the dataset. **1p** 
# 
# The simplest way to explain why it may be advantageous to remove the most common words is that they don't give us much information. In this case when classifying an email as spam or ham words like "the", "and", "from", etc. might not help to distinguish between spam and ham and therefore act as noise which negatively impacts performance. It is not certain that the removal of the n most popular words will guarantee that the model will be more accurate measured by true positive and true negative rates, but by exploring different exclusion rates it is possible to find the optimal one. 
# 
# Also, words that only occur once in the document does not provide any useful information to the Naive Bayes classifiers and therefore could be removed. This will probably not effect the accuracy in major ways, since these words did not influence the classifiers greatly before, but it will help to improve performance and make the matrix of vectors more dense and only include words that actually play a role in determining if the emails are spam or not.  
# 
# **b.** Use the parameters in Sklearn’s `CountVectorizer` to filter out these words. Update the program from point 3 and run it on your data and report and discuss your results. You have two options to do this in Sklearn: either using the words found in part (a) or letting Sklearn do it for you. Argue for your decision-making. **1p** 
# 
# We chose to let Sklearn do it for us by utilizing the min_df and max_df parameters in CountVectorizer. Our decision were based on the reasoning that the most important measurement is not how many times a word occurs in total, rather how many distinct emails that contain a certain word. Since CountVectorizer min_df and max_df are suitable parameters for removing words based on that exact measurement, we saw no advantage of developing our own method for removing the exact words found in part(a). For example, by removing words that only occur in one email, by definition we also remove all words that only occur once, which is what we have defined as least common in part (a).
# 
# To then find the optimal percentage of words that should be removed before analysis of the emails, regarding both most and least common words, every value of max_df and min_df respectively were investigated with respect to their impact to true positive accuracy. True positive and not true negative accuracy is chosen due to the assumption that it is more important to not spam-classify emails that are not really spam, as discussed in Task 3. The parameters and their respective true positive accuracy are plotted below. Here we can see that the true positive accuracy increases with increasing max_df and a constant min_df. The plot of the min_df value and the true positive accuracy shows that, the other way around, a higher min_df value generates lower true positive accuracies with a constant max_df (the values around min_df=0.9 are just noise with low amount of data). However, no measured accuracy from the plots when varying the removal of common and uncommon words are giving higher true positive accuracies than without them. However some of the gave higher true negative rates. That is of course a trade-off, but to keep the true positive rate as high as possible with a multinomial model, no common or uncommon words are removed. It is important to note that we have not measured all possible combinations of min_df and max_df, each parameter have only been evaluated with the other kept constant. Therefore, there is a possibility that there is a combination between min_df and max_df that results in a higher true positve rate than min_df = 0 and max_df = 1, but we have not found such a combination. Therefore, min_df and max_df will be set to 0 and 1 respectively for the reminder of this assignment. 


from collections import Counter

# Create a counter and find the frequency of words in the emails
counter = Counter()
for email in emails:
    for word in email.split():
        counter[word] += 1

# Find the 50 most common words
most_common = counter.most_common(50)

# Find least common words in counter
counterSort = sorted(counter, key=counter.get, reverse=False)
least_common = []
for word in counter:
    if counter[word] == 1: 
        least_common.append(word)

# Print 50 most and least commong words
print(most_common)
print(least_common[0:50])

import numpy as np

# Create list of accuracies and different max_dfs to measure
tpAccuracy = []
count = 1
max_df = np.arange(0.01, 0.99, 0.01)

# Loop through all different max_dfs and add the respective accuracy to the list
for i in max_df:
    count_vectorizer = CountVectorizer()
    count_vectorizer = count_vectorizer.set_params(max_df = i, min_df = 0)
    emails_vector2 = count_vectorizer.fit_transform(emails)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(emails_vector2, email_labels, random_state=1, test_size=0.25, stratify = email_labels)

    transform(y_train2,y_test2)
    
    mnb = MultinomialNB()

    mnb.fit(X_train2, y_train2)

    y_pred = mnb.predict(X_test2)

    tn, fp, fn, tp = confusion_matrix(y_test2,y_pred).ravel()

    tpAccuracy.append(tn/(fp+tn))

    count += 1

#Plot figure showing accuracy for varying max_df
plt.figure(figsize=(20,10))
plt.title("True positive accuracy with varying max_df")
plt.grid(True)
plt.xlabel('Value of max_df')
plt.ylabel('True positive accuracy')
plt.plot(max_df,tpAccuracy)
plt.show()

tpAccuracy = []
count = 1
max_df = np.arange(0.01, 0.89, 0.01)

for i in max_df:
    count_vectorizer = CountVectorizer()
    count_vectorizer = count_vectorizer.set_params(max_df = 0.9, min_df = i)
    emails_vector2 = count_vectorizer.fit_transform(emails)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(emails_vector2, email_labels, random_state=1, test_size=0.25, stratify = email_labels)

    transform (y_train2,y_test2)
    
    mnb = MultinomialNB()

    mnb.fit(X_train2, y_train2)

    y_pred = mnb.predict(X_test2)

    tn, fp, fn, tp = confusion_matrix(y_test2,y_pred).ravel()

    tpAccuracy.append(tn/(fp+tn))

    count += 1

# Plot figure showing accuracy for varying min_df
plt.figure(figsize=(20,10))
plt.title("True positive accuracy with varying min_df")
plt.grid(True)
plt.xlabel('Value of min_df')
plt.ylabel('True positive accuracy')
plt.plot(max_df,tpAccuracy)
plt.show()

# ### 5. Eeking out further performance
# **a.**  Use a lemmatizer to normalize the text (for example from the `nltk` library). For one implementation look at the documentation ([here](https://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes)). Run your program again and answer the following questions: 
# 
#   - Why can lemmatization help?
# 
# Lemmatization can increase the classifiers ability to successfully classify emails into either spam or ham. The reason for this is that lemmatization reduces all words into their most "basic" version, known as the *lemma*. For example, *run* and *ran* both get transformed to *run*, and instead of having two features for *ran* and *run*, only the lemma run gets added as a feature. By doing this the classifiers can successfully interpret run and ran as the same word, and mail that contains different forms of the same lemma are likely to belong to the same mail label. By doing this, the vectors also become more dense and the performance should therefore increase.
# 
#   -	Does the result improve from 3 and 4? Discuss. **1.5p** 
# 
# In our case, the result did not improve from Task 3 to Task 5. Since the true positive accuracy was very high in our case to start with, it did not succeed further improve this rate by classifying any of the 9 false classified emails after lemmatizing. The true negative rate was slightly decreased. This could be due to that some words should be left in their non-lemma form and not be interpretated as a lemma since their original form is stronger associated to a characteristic of a spam email. However, if this data manipulation was performed before tuning other parameters of the model, it is likely that it would have affected the model's accuracy positively.


#Download necessary packages for nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

wordnet_lemmatizer = WordNetLemmatizer()

#Get wordnet_pos (grammar class) for a given word.
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

#Create new Tokenizer that seperates words and removes non relevant values. 
tok = nltk.RegexpTokenizer(r"\w+")

#Initilize a new string for suitable datarepresentation for tokenize
str1 = "" 
count1 = 0  

#Fill string with all emails contained in list emails
for email in emails:
    str1 = tok.tokenize(email)
    email = ""
    count2 = 0

    #Substitute uppercase symbols with lowercase and get pos for each word
    for word in str1:
        word = word.lower()
        str1[count2] = wordnet_lemmatizer.lemmatize(word,get_wordnet_pos(word))
        count2 += 1

    #Count each word and update 
    for word in str1:
        email += word + " "
    emails[count1] = email
    count1 += 1

#Vectorize emails with max_df and min_df and new emails list with lemmatized words
count_vectorizer = CountVectorizer()
emails_vector3 = count_vectorizer.fit_transform(emails)

#Create train test split with stratify
X_train3, X_test3, y_train3, y_test3 = train_test_split(emails_vector3, email_labels, random_state=1, test_size=0.25, stratify = email_labels)

#Set label 2 to label 0 in training and test set. Reason is that in the end, we only want to determine if spam or ham.  
transform(y_train3,y_test3)

#Multinomial Naïve Bayes classification - Spam versus (hard-ham + easy-ham)
MNBClassification(X_train3,y_train3,X_test3,y_test3)

# **b.** The split of the data set into a training set and a test set can lead to very skewed results. Why is this, and do you have suggestions on remedies? 
# What do you expect would happen if your training set were mostly spam messages while your test set were mostly ham messages?  **1p** 
# 
# If the data in the dataset is unbalanced in any direction, meaning that either spam emails or ham emails are overrepresented, this could be problematic when the data is split into training and test data. If the training data is overrepresented by for example spam messages, the model could be overfitted to this kind of data and assume that most words contained in a spam email are triggering spam filters. Especially if common words are not cleaned before, then the model could be trained into believing that common words such as "and", "from" or "the" are characteristics of a spam email. When the model is tested on the test data, the system is then likely to classify emails are spam when they are really not. This will lead to a lower true positive rate which is very undesirable for this type of model, as dicussed in Task 3.
# 
# 
# One possible remedy is to stratify the split, which ensures that the proportion of each label is the same in both the test set and training set respectively. This has been done in every train/test split so far in the assignment. Another possibly remidy is to resample the training dataset, which can be applied when there is a severe skrew in labels distribution, for example 1:1000 examples of the minority label in relation to the majority label. This resampling can be done by two different methods;*Undersampling* which means to delete random examples from the majority label, and *Oversampling* which is to duplicate examples from the minorty label. When undersampling it is possible to loose important information by dropping examples that would otherwise provide useful information for the model.


# **c.** Re-estimate your classifier using `fit_prior` parameter set to `false`, and answer the following questions:
#   - What does this parameter mean?
#  
# Setting this parameter to true or false alters if the classifier takes into account the fitted distribution of labels when predicting new data points. For example, if it is true and there are two labels spam and ham, and the training data contained 70% spam and 30% ham the model take this into account when classifying new mails. In effect, an email that the classifier is keen to label as ham have a chance to instead be classified as spam because based on the probability of the training data, a new datatpoint is more likely to be spam than ham. If this parameter instaead is set to false, the classifier does not take into account the distribution of labels in the training data, and new datapoints are entirely classified based upon the features of that data point, without taking into consideration the relative occurance of different labels. 
# 
#   - How does this alter the predictions? Discuss why or why not. **0.5p** 
# 
# In this case, it seems like change the fit_prior from True (default) to False did not improve the predictions and there for not the true positive nor the true negative rates. Even though the prioritization now is uniform, it does not affect the outcome of the predictions positively. That might be due to that the features of a spam respectively a ham email are strong enough by themselves for prediction and that a class prior does not change that strength.


# New method for Multinomial Naive Bayes classification with fit_prior = False
def MNBClassification(X_train,y_train,X_test,y_test):
    
    # Create a multinomial Naive Bayes model
    mnb = MultinomialNB(fit_prior = False)

    #Train the model using the training sets
    mnb.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = mnb.predict(X_test)

    # Create a confusion matrix to visualize predictions and true negative/true positive absolute numbers
    titles_options = [("Confusion matrix", None)]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(mnb, X_test, y_test, display_labels=["non-spam","spam"], cmap=plt.cm.Blues, normalize=normalize)   
        plt.grid(False) 
        disp.ax_.set_title(title)
    plt.grid(False)
    plt.show()

    # Find and print the true positive/true negative rates
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    tpRate = tn/(fp+tn)
    tnRate = tp/(fn+tp)
    print ("True positive rate: " + str(tpRate))
    print ("True negative rate: " + str(tnRate))

#Multinomial Naïve Bayes classification - Spam versus (hard-ham + easy-ham)
MNBClassification(X_train3,y_train3,X_test3,y_test3)

# **d.** The python model includes smoothing (`alpha` parameter ), explain why this can be important. 
# 
# The alpha parameter can be important because it complements the calculation for maximum likelyhood with a small number if the number of occurences of a certain feature in a given label is 0. If smoothing is not done, an email containing a word that is not present in any other emails with that label, the frequency-based probability for that word, and any probabilities containing that word, will be zero. Smoothing is a way to add a pseudocount to the probability estimators of all words, including them not occuring with a specific label, to avoid the zero probability issue. In the Multinomial Naive Bayes classification, setting a value to alpha that is not 0 will create a pseudocount like this. Using higher alpha values will force the probability estimator towards 0.5. The value of alpha that best fit our data, measured in higher true negative accuracy with retained true positive accuracy, is alpha = 0.5 which increased the true negative accuracy from **93.6% to 96.0%**
# 
# - What would happen if in the training data set the word 'money' only appears in spam examples? What would the model predict about a message containing the word 'money'? Does the prediction depend on the rest of the message and is that reasonable? Explain your reasoning  **1p** 
# 
# If the word *money* only appears in spam examples in the training data set the probability for that word in label ham would be set to zero. The probability is frequency-based and without smoothing it is zero if a given label and word never occur together in training data. This would mean that a message/mail containing *money* would automatically have probability zero to be a ham mail, and would therefore be predicted as a spam mail. In the case of no smoothing this prediction does not depend at all on the rest of the message as the probability for *money* will be zero, resulting in that the dot product calculation for all words in that label will be zero. This is not reasonable and highlights the importance of smoothing. It is very likely that new mails will contain words that don't occur in the training data and predecting labels for these words would be impossible without smoothing. If smoothing is applied, the probability would not equal zero for label ham just because a message contains *money*, but instead a small probability will be set to this label/word combination because of smoothing and if the rest of the message indicates that it should be predicted as ham it still could. 


# New method for Multinomial Naive Bayes classification with alpha = 0.5
def MNBClassification(X_train,y_train,X_test,y_test):
    
    # Create a multinomial Naive Bayes model
    mnb = MultinomialNB(alpha = 0.5)

    #Train the model using the training sets
    mnb.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = mnb.predict(X_test)

    # Create a confusion matrix to visualize predictions and true negative/true positive absolute numbers
    titles_options = [("Confusion matrix", None)]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(mnb, X_test, y_test, display_labels=["non-spam","spam"], cmap=plt.cm.Blues, normalize=normalize)   
        plt.grid(False) 
        disp.ax_.set_title(title)
    plt.grid(False)
    plt.show()

    # Find and print the true positive/true negative rates
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    tpRate = tn/(fp+tn)
    tnRate = tp/(fn+tp)
    print ("True positive rate: " + str(tpRate))
    print ("True negative rate: " + str(tnRate))

#Multinomial Naïve Bayes classification - Spam versus (hard-ham + easy-ham) after adjusting for alpha to 0.5
MNBClassification(X_train3,y_train3,X_test3,y_test3)

