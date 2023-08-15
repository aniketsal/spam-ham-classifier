# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# %%


# %%
path=str(os.getcwd())
spam_directory = os.listdir(path+ "/spam/")
ham_directory = os.listdir(path + "/ham/")

# %%
def preprocessFile(filename):
	words = []
	with open(filename, "r", errors="ignore") as file:
		filedata = file.readlines()
		a=['\\r','\\"','\\n']
		b=["won\'t","can\'t","n\'t","\'re","\'s","\'d","\'ll","\'t","\'ve","\'m"]
		c=["will not","can not","not","are","is","would","will","not","have","am"]
		for line in filedata:
			phrase=line
			for i in range(len(b)):
				phrase=re.sub(r"b[i]",c[i],phrase)
			sent = phrase
			for i in a:
				sent=sent.replace(i,' ')
			sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
			sent = ' '.join(e.lower() for e in sent.split())
			words += list(sent.strip().split())
	return words


# %%
def vocab(isspam,vocabulary):
    if (isspam):
        path=str(os.getcwd())+ "/spam/"
        directory=spam_directory
    else:
        path=str(os.getcwd()) + "/ham/"
        directory=ham_directory
    
    for file in directory:
        for word in preprocessFile(path+file):
            if word not in vocabulary:
                vocabulary[word]=0

    return vocabulary


def calculate_probabilty(isspam,spam_vocabulary,ham_vocabulary):
    if (isspam):
        path=str(os.getcwd())+ "/spam/"
        directory=spam_directory
        _vocabulary=spam_vocabulary

    else:
        path=str(os.getcwd())+ "/ham/"
        directory=ham_directory
        _vocabulary=ham_vocabulary
    
    for file in directory:
        for word in preprocessFile(path+file):
            _vocabulary[word]+=1
    for x in _vocabulary:
        _vocabulary[x]+=1
    count=0
    for word in _vocabulary:
        if(_vocabulary[word]!=0):
            count+=1
    
    for word in _vocabulary:
        _vocabulary[word] /= count
    return _vocabulary



# %%
vocabulary={}
vocabulary=vocab(1,vocabulary)
vocabulary=vocab(0,vocabulary)

# %%
word_list=[]
for word in vocabulary:
    word_list.append(word)
spam_vocabulary=vocabulary.copy()
ham_vocabulary=vocabulary.copy()
spam_vocabulary=calculate_probabilty(1,spam_vocabulary,ham_vocabulary)
ham_vocabulary=calculate_probabilty(0,spam_vocabulary,ham_vocabulary)

# %%
length=len(ham_directory)+len(spam_directory)
phat = len(spam_directory)/length
phat=0.5

# %%
def check(x,vocabulary):
    if x in vocabulary:return False
    else: return True
def compute_feature(spam,x_test,ham,isspam):
    f=[]
    if (isspam==1):
        words=spam
    else: words=ham
    for x in words:
        if (check(x,x_test)):
              f.append(0)
        else: f.append(1)
    return f

def Compute_Label(x_test,flag):
    x_test=preprocessFile(x_test)
    f=[]
    word_list_spam = []
    word_list_ham=[]
    for x in x_test:
        
        if (check(x,vocabulary)):
            spam_vocabulary[x] = 1/(len(spam_vocabulary)+1)
            ham_vocabulary[x] = 1/(len(ham_vocabulary)+1)
            vocabulary[x] = 0
    for x in vocabulary:
        word_list.append(x)
    

    temp_spam_vocabulary = dict(sorted(spam_vocabulary.items(), key=lambda kv: (kv[1], kv[0]))[::-1])
    temp_ham_vocabulary = dict(sorted(ham_vocabulary.items(), key=lambda kv: (kv[1], kv[0]))[::-1])
    
    for x in temp_spam_vocabulary:
        word_list_spam.append(x)
    
    for x in temp_ham_vocabulary:
        word_list_ham.append(x)

    feature_spam=compute_feature(temp_spam_vocabulary,x_test,temp_ham_vocabulary,1)
    feature_ham=compute_feature(temp_spam_vocabulary,x_test,temp_ham_vocabulary,0)

    ham_probability=1
    spam_probability=1
    for x in range(len(feature_ham)):
        ham_probability  *= ((temp_ham_vocabulary[word_list_ham[x]]**feature_ham[x]) *  ((1-temp_ham_vocabulary[word_list_ham[x]])**(1-feature_ham[x])))
        spam_probability  *= ((temp_spam_vocabulary[word_list_spam[x]]**feature_spam[x]) * ((1-temp_spam_vocabulary[word_list_spam[x]])**(1-feature_spam[x])))
    ham_probability *= (1-phat)
    spam_probability *= phat


    
    if(ham_probability > spam_probability):
        if flag:
            print("0")
        return 0
    else:
        if flag:
            print("+1")
        return 1


# %%
count_spam = 0
path=str(os.getcwd())+ "/spam/"
for filename in spam_directory:
    if(Compute_Label(path+filename,0)==1):
        count_spam+=1
print(" Training accuracy on spam emails:")
print(count_spam/len(spam_directory)*100)

# %%
count_ham = 0
path=str(os.getcwd())+ "/ham/"
for filename in ham_directory:
    if(Compute_Label(path+filename,0)==0):
        count_ham+=1
print("Training accuracy on ham emails:")
print(count_ham/len(ham_directory)*100)

# %%
path=str(os.getcwd()) + '/test/'
dir=os.listdir(path)
print("Classification for email files")
for filename in dir:
    Compute_Label(path+filename,1)

# %%



