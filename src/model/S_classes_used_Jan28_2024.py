#S_classes_used_Jan28_2024.py
'''
Python 3.10.11
import  sklearn
sklearn.__version__
'1.3.0'
https://contrib.scikit-learn.org/lightning/generated/lightning.regression.FistaRegressor.html
pip install sklearn-contrib-lightning
https://contrib.scikit-learn.org/lightning/index.html
'''
import sys
print('python version is ', sys.version)
print('path to python exe ' ,  sys.executable)
import  sklearn
print('sklearn  version' , sklearn.__version__)
from sklearn.feature_extraction.text import TfidfVectorizer
import  numpy as  np
from gensim.parsing.preprocessing import preprocess_documents
from gensim.parsing.porter import PorterStemmer

from lightning.classification import CDClassifier #Estimator for learning linear classifiers by (block) coordinate descent
#import  lightning
#do in parallel on many VMs/instances


class classifier:
    def __init__(self, Xtest, Xtrain, ytest, ytrain):

        self.X_train, self.y_train, self.X_test, self.y_test = self.split_training_test_data(Xtest, Xtrain, ytest, ytrain)
        X_train_vectorised =  self.preprocessing_function(self.X_train, self.X_train )
        #jan28 test_data_vectorised =  self.preprocessing_function(self.X_train, self.X_test )
    
        self.clf_lightning = CDClassifier(loss="squared_hinge",
                                            #penalty="l2",
                                            penalty="l1",
                                            #penalty="l1/l2", #                     
                                            multiclass=False,
                                            max_iter=20,
                                            alpha=1e-4,
                                            C=1.0 / X_train_vectorised.shape[0],
                                            tol=1e-3,
                                            n_jobs =5)

        self.clf_lightning.fit(X_train_vectorised, self.y_train )

 
  
    def split_training_test_data(self, Xtest, Xtrain, ytest, ytrain):
        text_file = open(ytest, "r", errors="ignore")
        y_test = text_file.readlines()
        y_test =  [int(x) for x in y_test]
        text_file = open(ytrain, "r", errors="ignore")
        y_train = text_file.readlines()
        y_train =  [int(x) for x in y_train]
        text_file = open(Xtest, "r", errors="ignore")
        X_test = text_file.readlines()
        text_file = open(Xtrain, "r", errors="ignore")
        X_train = text_file.readlines()
    
        y_train =  np.array( y_train )
        y_test =  np.array( y_test )

        return X_train, y_train, X_test, y_test
                        
    def preprocessing_function(self, X_train, data_to_vectorize ):
        vectorizer = TfidfVectorizer( sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english")    
        #X_train_original =  copy.deepcopy(X_train)
        my_PorterStemmer = PorterStemmer()
        X_train =  my_PorterStemmer.stem_documents( X_train )
        X_train =  preprocess_documents(X_train)
        X_train = [" ".join(x) for x in X_train]
        vectorizer.fit(X_train)
        data_to_vectorize =  my_PorterStemmer.stem_documents( data_to_vectorize )
        data_to_vectorize =  preprocess_documents(data_to_vectorize)
        data_to_vectorize = [" ".join(x) for x in data_to_vectorize]  
        X_train_vectorised = vectorizer.transform(data_to_vectorize)
        return X_train_vectorised

    def  my_inference(self, my_text ):
        test_vectorised =  self.preprocessing_function(self.X_train, [my_text])
        #predict use existing model 
        pred_test_lightning = self.clf_lightning.predict(  test_vectorised    )
        print('for text => ', my_text )
        print('pred_test_lightning', pred_test_lightning[0])
        return pred_test_lightning


if __name__ == '__main__':
    clf_lintening = classifier('X_test.txt', 'X_train.txt', 'y_test.txt', 'y_train.txt')

    #  input your text -> imitate request from any computer or web site 
    my_text =  clf_lintening.X_test[0]

    my_inference = clf_lintening.my_inference(my_text) 