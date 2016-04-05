from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import logging # https://docs.python.org/2/howto/logging.html
from mfunc import lin

# http://paolaelefante.com/2016/03/a-small-guide-to-random-forest-part-2/
# http://blog.yhat.com/posts/random-forests-in-python.html
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
# https://www.quora.com/How-does-randomization-in-a-random-forest-work

# A Get the data
# B,C Clean and prepare the data
# D Prepare the features and samples
# E1 Select the training and testing sets 
# E2 Train the model    
# F Test the model

def selectsets(data,sizeTestSet):
    trainingSet=data[:-sizeTestSet]
    testSet=data[-sizeTestSet:]
    return(trainingSet,testSet)

class analyzeBinaryPreds(object):
    """Calculate and show results for the"""
    """comparison of actual and predicted data vectors."""

    def __init__(self, y_test, preds):
        self.y_test = y_test
        self.preds = preds

    def calculateResults(self):
        """Calculates accuracy, true/false pos/neg. rates."""
        self.totalPos=self.y_test.sum()
        self.totalNeg=self.y_test.size-self.totalPos
        truePos=0
        trueNeg=0
        falsePos=0 # falsely predicted positive
        falseNeg=0 # falsely predicted negative
        for i in np.arange(0,self.y_test.size):
            if(self.preds[i]==1 and self.y_test[i]==1): truePos+=1
            if(self.preds[i]==0 and self.y_test[i]==0): trueNeg+=1
            if(self.preds[i]==1 and self.y_test[i]==0): falsePos+=1
            if(self.preds[i]==0 and self.y_test[i]==1): falseNeg+=1
        self.accuracy=(truePos+trueNeg)/self.y_test.size

        # Make array: tfposneg = [truePos,trueNeg,falsePos,falseNeg]
        self.tfposneg=np.zeros(4)
        self.tfposneg[0]=truePos
        self.tfposneg[1]=trueNeg
        self.tfposneg[2]=falsePos
        self.tfposneg[3]=falseNeg        
        
    def showResults(self):
        # confusion matrix
        print(pd.crosstab(self.y_test, self.preds, rownames=['actual'], colnames=['preds']))
        print('accuracy:',self.accuracy)
        print('error:',1.0-self.accuracy)
        print('true positive rate:',self.tfposneg[0]/self.totalPos)
        print('true negative rate:',self.tfposneg[1]/self.totalNeg)
        print('positive precision:',self.tfposneg[0]/(self.tfposneg[0]+self.tfposneg[2]))
        print('negative precision:',self.tfposneg[1]/(self.tfposneg[1]+self.tfposneg[3]))


def main(): 

#    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    logging.basicConfig(level=logging.WARNING,format='%(asctime)s %(message)s')
    logging.info('Started')
    lin()
    
# A Get the data
# B,C Clean and prepare the data
    rawdata='trainTitanic.csv'
    df = pd.read_csv(rawdata)
    nSamples=len(df)
    if(False): print(df.head(2))
    print('finish reading data, length of data:',nSamples)
    logging.info('Data read, len=%i',nSamples)


# D Select the features and samples
    featureNames=['Sex','Pclass','Parch','SibSp','Embarked']
    print('Feature names:',featureNames)
    nFeatures=len(featureNames)

    features=np.zeros((nSamples,nFeatures))
    i=-1
    for featureName in featureNames:
        i+=1
        features[:,i]=df[featureName].factorize()[0]
    y=df['Survived'].factorize()[0] # Target vector, binary. Values 0 or 1.
    logging.info('Features selected')
    logging.info('y:',y)


# E1 Select the training and testing sets 
    sizeTestSet=89
    features_train, features_test = selectsets(features, sizeTestSet)
    y_train, y_test = selectsets(y, sizeTestSet)
    if (len(y_train) < 51):
            logging.warning('Number of samples < 51')

    
# E2 Train the model
    clf = RandomForestClassifier(n_estimators=1000,max_features=3,oob_score=True,verbose=0)
    clf.fit(features_train, y_train)
    print('oob_score error:',1.0-clf.oob_score_)
    print('feature importances:',clf.feature_importances_)
    dummyTest=False
    if(dummyTest): # Test just using the same training set (not really relevant)
        print('Test just on the training set')
        preds=clf.predict(features_train)  # make confusion table
        print(pd.crosstab(y_train, preds, rownames=['actual'], colnames=['preds']))
    logging.info('Model trained')

    
# F Test the model
    lin()
    print('Test with the true testing set, size:',y_test.size)  
    preds=clf.predict(features_test)  

    instance=analyzeBinaryPreds(y_test,preds)   # print(type(instance))
    instance.calculateResults()
    instance.showResults()

    additionalTests=False
    if(additionalTests):
        j=-1
        for nTrees in np.arange(50,1050,100):
            clf = RandomForestClassifier(n_estimators=nTrees,max_features=3,oob_score=True,verbose=0)
            clf.fit(features, y)
            j+=1
            print('j,i, oob_score error:',j,i,1.0-clf.oob_score_)
    if(additionalTests):
        for depth in np.arange(1,31):
            clf = RandomForestClassifier(n_estimators=400,max_depth=depth,max_features=3,oob_score=True,verbose=0)
            clf.fit(features, y)
            print('i, oob_score error:',i,1.0-clf.oob_score_)

    logging.info('Model tested')
    logging.info('Program finished')
    lin()




if __name__ == '__main__':
    main()


# Dump






