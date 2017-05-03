
# coding: utf-8

# # ViCare QnA specialty classification
# ## Comparing text classification techniques with Cross-Validation

# - Title: QnAtag_cross-validation_f1score.py
# - Description: compare techniques used in text classification for Vietnamese questions from ViCare.vn
# - Author: Dinh Van Quy | <quy.dinh3195@gmail.com>
# - Date created: 2017-04-27
# - Version: 0.1
# - Usage:
# - Notes:
# - Python_version: 2.7.13

# ### Set up params

# In[11]:

data_fpath = 'all_thread_1count-and-stop-words-removal.csv'
vect_obj_fName = 'QnAtag_logit_TfidfVectorizer_no-preprocessing_lbfgs_vect-export.pkl'
model_fName = 'QnAtag_logit_TfidfVectorizer_no-preprocessing_lbfgs_model.pkl'
logit_solver = 'lbfgs'


# ### Prepare data

# In[12]:

# %config InlineBackend.figure_format = 'retina'
get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (10, 12)
import os
import re
import pandas as pd
import time
from slack_privateMes import slack_message
import traceback #use this to print traceback errors why using try except
import matplotlib.pyplot as plt

#Import unprocessed data
#Populate questions into pandas DataFrame and insert column headers
df = pd.read_csv('all_thread.csv', header=-1)
df = df.rename(columns={0: 'question', 1: 'org_label', 2: 'label'})

#Remove the first row which seems to be nonsense
df = df.drop(df.index[0])
df.head()
# In[13]:

#Import processed data
df = pd.read_csv(data_fpath, index_col=0)
df.shape


# In[14]:

df.dropna(axis=0, how='any', inplace=True) #drop na values
df.head()


# In[15]:

# Define how many data can be used to train the models
# test_df = df.sample(n=5000) # trying to get randomly 5k entries for the cross-validation to run quicker
test_df = df.copy() # otherwise input all the data into training models
test_df.index.name = 'id'
test_df = test_df.drop('label', axis=1)

# Group specialties whose counts < 5
label_counts = test_df['org_label'].value_counts()
label_5counts = label_counts[label_counts <= 5].index
test_df['org_label'][test_df['org_label'].isin(label_5counts)] = 'Khác'


# ### Split train test
from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(test_df['question'], test_df['label'], train_size=40828, random_state = 10, stratify=test_df['label'])
X_train,X_test,y_train,y_test = train_test_split(test_df['question'], test_df['label'], train_size=40828, random_state = 10)
# In[16]:

# Assign predictors and target variables
X = test_df.iloc[:,0] #predictors
y = test_df.iloc[:,1] #target variables

# Stratified Shuffle Split into 4 folds
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=0)
train_sss = []
test_sss = []

sss_split = sss.split(X,y)
for train_index, test_index in sss_split:
    print("TRAIN:", train_index, "TEST:", test_index)
    train_sss.append(train_index)
    test_sss.append(test_index)

num_of_splits = sss.get_n_splits(X, y) 

# Check intersections between splitted data chunks
for i in range(num_of_splits):
    train_set = set(train_sss[i])
    test_set = set(test_sss[i])
    print('Intersection between train and test of data_chunk%s: %s ' % (i+1, train_set.intersection(test_set)))
    
    # Save each train and test chunk to a csv file
    name_train = 'train%s_rm.csv' % (i+1)
    name_test = 'test%s_rm.csv' % (i+1)
    test_df.ix[train_sss[i]].to_csv(name_train, encoding='utf-8')
    test_df.ix[test_sss[i]].to_csv(name_test, encoding='utf-8')


# In[17]:

# Check for intersections between pairs of train set
def create_ONI_pairs(li):
    # Create all order-non-important pairs of number from a list of number
    pair_li = []
    li_copy = li[:]
    for f in range(len(li)):
        li_copy = list(set(li_copy) - set([li[f]]))
        for c in li_copy:
            temp = [li[f]]
            temp.append(c)
            el = tuple(temp)
            pair_li.append(el)
    return pair_li

list_of_pair = create_ONI_pairs(range(num_of_splits))

for i in list_of_pair:
    train_set_1 = set(train_sss[i[0]])
    train_set_2 = set(train_sss[i[1]])
    print('Intersection between train set %s and train set %s: %s ' % (i[0]+1, i[1]+1, len(train_set_1.intersection(train_set_2))))


# In[18]:

# Number of intersections of all train sets
len(set.intersection(*[set(i) for i in train_sss]))


# ### Check whether the splitted training sets are stratified or not

# In[19]:

# Plot specialty distributions for all data set
temp = test_df['org_label'].value_counts()
temp = temp.sort_values(ascending=True)
temp.plot(kind='barh', figsize=(10,15))


# In[20]:

# Get the top 20 specialties that have the most counts
top_index = test_df['org_label'].value_counts()[:20].index


# In[22]:

# Define a function that returns the bar plot of label distribution for each train and test chunk
list_of_fpath = ['test1_rm.csv',
 'test2_rm.csv',
 'test3_rm.csv',
 'test4_rm.csv',
 'train1_rm.csv',
 'train2_rm.csv',
 'train3_rm.csv',
 'train4_rm.csv',]

def dist_plot(list_of_fpath=list_of_fpath, index=top_index):
    # Set up a 8-frame figure to record plotting
    fig, axs = plt.subplots(2,4)
    
    # Load into pandas df    
    for i in list_of_fpath:
        df = pd.read_csv(i, index_col=0)
        df = df[df['org_label'].isin(index)]
        pop = df['org_label'].value_counts()
        table = pd.Series(pop.values, index=index)
        table.plot(kind='barh')
        plt.show()

dist_plot()


# ### Convert label to a numerical variable

# In[11]:

labels = test_df.org_label.unique()
labels_code = range(len(labels))
label_dict = dict(zip(labels, labels_code)) # concatenate a key array and a value array into a python dictionary


# In[12]:

# Create new column to number-encode label
test_df['label_code'] = test_df.org_label.map(label_dict)


# ### Define a classification function with options to choose model techniques

# In[13]:

# Import models from sklearn
from sklearn import metrics
from sklearn.cross_validation import KFold # For K-fold cross validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


# In[14]:

# Handle cases when input_text is already in unicode
def unicoded(input_text):
    return unicode(input_text, encoding='utf-8', errors='ignore') if not isinstance(input_text, unicode) else input_text

#Import ViTokenizer 
from pyvi.pyvi import ViTokenizer

#Define a function that takes raw VNese words then
def vitokenizer(input_text):
    input_text = unicoded(input_text)
    input_text = ViTokenizer.tokenize(input_text)
    
    #Turn result into usable format to input into Vectorizer(tokenizer=)
    input_text = input_text.split()
    input_text = [x.replace('_', ' ') for x in input_text]
    return input_text
# In[15]:

import pandas as pd
# Define a function to export classification report to csv
def export_report(report, cv_number):
    report_df = pd.concat([pd.Series(report[0]),pd.Series(report[1]),pd.Series(report[2]),pd.Series(report[3])],axis=1)
    report_df = report_df.rename(columns=dict(zip([0,1,2,3],['precision', 'recall', 'f1_score', 'support'])))
    fname = 'report_fold%s.csv' % (cv_number)
    report_df.to_csv(fname, encoding='utf-8')


# In[16]:

#Define predictors and target variables
predictor_var = 'question'
outcome_var = 'org_label'

#Generate a function for building the classification model
def classification_model(model, data, predictors, outcome):
    try:
        # timer
        start = time.time()
        
        # Vectorization
        vect = TfidfVectorizer()
        vect.fit(data[predictors])
        # Export vectorization object
#         joblib.dump(vect, vect_obj_fName)
        
        # Fit the model
        model.fit(vect.transform(data[predictors]), data[outcome])
#         joblib.dump(logit_model, model_fName)
        
        # Make predictions on training set
        predictions = model.predict(vect.transform(data[predictors]))
        
        # Accuracy score
        accuracy = metrics.accuracy_score(y_pred=predictions, y_true=data[outcome])
        print('Accuracy with all training set: %s.' % ('{0:.3%}'.format(accuracy)))
        
        # F1 score
        accuracy = metrics.f1_score(y_pred=predictions, y_true=data[outcome], average='macro')
        end1 = time.time()
        print('F1 score with all training set: %s. Took %ss.' % ('{0:.3%}'.format(accuracy), (end1 - start)))
        
        report = metrics.precision_recall_fscore_support(y_true=data[outcome], y_pred=predictions, average=None)
        
        # Perform k-fold cross-validation with 5 folds
        error = []
        f1_score = []
        for i in range(num_of_splits):
            train = train_sss[i]
            test = test_sss[i]
            print('Cross-validating fold number %s' % (i+1))
            # Filter predictors data
            train_predictors = data[predictors].iloc[train]
            test_predictors = data[predictors].iloc[test]

            # The target we're using to train the problem
            train_target = data[outcome].iloc[train]

            # Extract text feature or something? (must google this later)
            vect = TfidfVectorizer()
            vect.fit(train_predictors)
            train_predictors_df = vect.transform(train_predictors)
            test_predictors_df = vect.transform(test_predictors)

            # Train the algorithm using the predictors and outcome
            model.fit(train_predictors_df, train_target)
            
            predictions = model.predict(test_predictors_df)

            # Record error from each cross-validation run
            error_score = model.score(test_predictors_df, data[outcome].iloc[test])
            print('Fold number %s error_score: %s' % (i+1, error_score))
            error.append(error_score)
            f1_score.append(metrics.f1_score(y_pred=predictions, y_true=data[outcome].iloc[test], average='macro'))
            report_fold = metrics.precision_recall_fscore_support(y_true=data[outcome].iloc[test], y_pred=predictions, average=None)
            export_report(report_fold, i+1)

        end = time.time()
        print('Cross-Validation Score: %s, Run Time: %ss.\n' % ('{0:.3%}'.format(np.mean(error)), (end - end1)))
        print('Cross-Validation macro average F1 Score: %s.' % ('{0:.3%}'.format(np.mean(f1_score))))
        
        # Fit the model again so that it can be referred out side the function:
        model.fit(vect.transform(data[predictors]), data[outcome])
    except:
        try:
            slack_message('Errors occur when running model!')
        except:
            pass
        traceback.print_exc()
#         raise ValueError("Errors!!!")
    
    return report


# In[17]:

# Run Logistic Regression and get the reports
logit_model = LogisticRegression(solver=logit_solver, multi_class='multinomial')
print('- Logistic Regression:')
report = classification_model(logit_model, test_df, predictor_var, outcome_var)
slack_message('Finish training Logistic Regression model!')


# ### Save the best performing model
vect = TfidfVectorizer(tokenizer=vitokenizer)
vect.fit(df[predictor_var])
logit_model = logit_model.fit(vect.transform(df[predictor_var]), df[outcome_var])

from sklearn.externals import joblib
joblib.dump(logit_model, 'QnAtag_logit_pyvitokenizer_1count_words-removal.pkl')
# In[ ]:




# In[ ]:



