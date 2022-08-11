from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import RidgeClassifier
from utils import toNumpyArray

# You may add more classifier methods replicating this function

# Naive Bayes ignores all the syntactic-semantic rules. The same can be said regarding the other classifiers.
# As such, these models have high bias, but in practice it has low variance.
def applyNaiveBayes(X_train, y_train, X_test):
    '''
    Task: Given some features train a Naive Bayes classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict

def applyLDA(X_train, y_train, X_test):
      trainArray = toNumpyArray(X_train)
      testArray = toNumpyArray(X_test)
    
      clf = LDA()
      clf.fit(trainArray, y_train)
      y_predict = clf.predict(testArray)
      return y_predict

def applyLSVM(X_train, y_train, X_test):
      trainArray = toNumpyArray(X_train)
      testArray = toNumpyArray(X_test)
    
      clf = LinearSVC()
      clf.fit(trainArray, y_train)
      y_predict = clf.predict(testArray)
      return y_predict

def applyMLP(X_train, y_train, X_test):
      trainArray = toNumpyArray(X_train)
      testArray = toNumpyArray(X_test)
    
      clf = MLPClassifier()
      clf.fit(trainArray, y_train)
      y_predict = clf.predict(testArray)
      return y_predict

def applyGBC(X_train, y_train, X_test):
      trainArray = toNumpyArray(X_train)
      testArray = toNumpyArray(X_test)
    
      clf = GBC()
      clf.fit(trainArray, y_train)
      y_predict = clf.predict(testArray)
      return y_predict

def applyRidgeCLF(X_train, y_train, X_test):
      trainArray = toNumpyArray(X_train)
      testArray = toNumpyArray(X_test)
    
      clf = RidgeClassifier()
      clf.fit(trainArray, y_train)
      y_predict = clf.predict(testArray)
      return y_predict