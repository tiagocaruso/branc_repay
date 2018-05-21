import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import linear_model
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class Ml_model(object):
	###########################################################################
	# This class takes a data set, splits in train and test and runs a ml model
	# It outputs probabilites, predicitions, scores and relavant features
	###########################################################################
    def __init__(self, model, data ,var, test_share = 0.2, seed = 10):
        self.model = model
        self.data = data
        self.var = var
        self.test_share = test_share
        self.seed = seed
        self.data_Y = data[var]
        self.data_X = data.drop([var], axis=1)
        self.split_data = self._split_data(self.data_X,self.data_Y,self.test_share,self.seed)
        self.train_X = self.split_data['train_X']
        self.test_X = self.split_data['test_X']
        self.train_Y = self.split_data['train_Y']
        self.test_Y = self.split_data['test_Y']
        self.trained_model = self.model.fit(self.train_X,self.train_Y) # running the model
        self.prob = self._probabilities(self.trained_model, self.test_X)
        self.pred = np.round(self.prob)
        self.f1_score = f1_score(self.test_Y,self.pred)
    
    def _split_data(self,df_X, df_Y,share,seed):
        train_X, test_X, train_Y, test_Y = train_test_split(df_X,df_Y, test_size = share, stratify = df_Y, random_state = seed)
        return {'train_X':train_X , 'test_X':test_X, 'train_Y':train_Y, "test_Y":test_Y}
    
    def _probabilities(self,trained_mod, df_X):
        logi = linear_model.LogisticRegression(verbose = 0, penalty = 'l1')
        lasso  = linear_model.Lasso(alpha=1e-03)
        if isinstance(trained_mod,type(logi)): # method is slithly different for different models
            raw_prob = trained_mod.predict_proba(df_X)
            prob = raw_prob[:,1]
        elif isinstance(trained_mod,type(lasso)):
            raw_prob = trained_mod.predict(df_X)
            prob = np.clip(raw_prob,0,1)
        else:
            prob = trained_mod.predict(df_X)
        return prob
    
    def print_scores(self):
        print 'Model predicted %s positives over %s true positives' %(self.pred.sum(),self.test_Y.sum())
        print 'AUC score %0.3f' % roc_auc_score(self.test_Y,self.prob)
        print 'Precision %0.3f' % precision_score(self.test_Y,self.pred)
        print 'Recall %0.3f' % recall_score(self.test_Y,self.pred)
        print 'F1 score %0.3f' % f1_score(self.test_Y,self.pred)
        
    def print_model_features(self, threshold = 0.00001):
        rf = ensemble.RandomForestRegressor()
        if isinstance(self.trained_model,type(rf)):
            mod_coef = self.trained_model.feature_importances_
        else:
            mod_coef = self.trained_model.coef_
        shaped_coef = np.reshape(mod_coef,(self.train_X.shape[1],))
        features = pd.DataFrame(index=self.test_X.columns, data=shaped_coef)
        features.reset_index(inplace = True)
        features.columns = ['feature','importance']
        features = features[abs(features.importance) > threshold]
        features.sort_values('importance',ascending = False, inplace = True)
        return features


class Voting_Ensemble(object):
	#####################################################################################################
	# This class takes a list of ML models, creates a voting emsemble 
	# Outpurs: voting weigths, probabilites, predicitions, scores, roc curve and precision recall curve
	#####################################################################################################
    def __init__(self, model_list, test_Y):
        self.model_list = model_list
        self.test_Y = test_Y
        self.voting_w_dict = self._voting_w(self.model_list)
        self.prob = self._probabilities(self.model_list, self.voting_w_dict)
        self.pred = np.round(self.prob)
        self.f1_score = f1_score(self.test_Y,self.pred)
    
    def _voting_w(self, mod_list):
        f1_scores_dict = {}
        sum_scores = 0
        for mod in mod_list:
            f1_scores_dict[mod] = mod.f1_score
            sum_scores += mod.f1_score
        voting_w_dict = {x:f1_scores_dict[x]/float(sum_scores) for x in f1_scores_dict.keys()}
        return voting_w_dict
    
    def _probabilities(self,model_list,voting_w_dict):
        prob = 0
        for mod in model_list:
            prob += mod.prob*float(voting_w_dict[mod])
        return prob
    
    def print_scores(self):
        print 'Model predicted %s positives over %s true positives' %(self.pred.sum(),self.test_Y.sum())
        print 'AUC score %0.3f' % roc_auc_score(self.test_Y,self.prob)
        print 'Precision %0.3f' % precision_score(self.test_Y,self.pred)
        print 'Recall %0.3f' % recall_score(self.test_Y,self.pred)
        print 'F1 score %0.3f' % f1_score(self.test_Y,self.pred)
        
        # Plotting precicion recall curve
    def print_precision_recall_curve(self, c = 'b', a = 0.2):
        precision, recall, _ = precision_recall_curve(self.test_Y, self.prob)
        plt.step(recall, precision, color = c, alpha = a, where='post')
        plt.fill_between(recall, precision, step='post', alpha=a, color=c)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        plt.title('Ensemble - Precision Recall Curve')
        
        # Plotting ROC Curve
    def print_roc_curve(self, c = 'b'):
        fpr, tpr, threshold = roc_curve(self.test_Y,self.prob)
        roc_auc = auc(fpr,tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr,tpr,c,label = 'AUC = %0.2f' %roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

# Getting the data
df_repaid = pd.DataFrame.from_csv('df_repaid')

# Logistic Model
logistic_model = linear_model.LogisticRegression(verbose = 0, penalty = 'l2')
logi_ml = Ml_model(logistic_model,df_repaid,var = 'repaid')

# Random Forest
rf = ensemble.RandomForestRegressor(n_estimators=20, verbose=1, min_impurity_split=1e-09)
rf_ml = Ml_model(rf,df_repaid,var = 'repaid')

# Lasso Model
lasso  = linear_model.Lasso(alpha=1e-03,fit_intercept=True, normalize=True)
lasso_ml = Ml_model(lasso,df_repaid,var = 'repaid')

# Voting Emsemble
df_Y = logi_ml.test_Y
ve = Voting_Ensemble([logi_ml,rf_ml,lasso_ml],df_Y)

### Looking at all the results
# Logistic Model
logi_ml.print_scores()
logi_ml.print_model_features()

# Random Forest
rf_ml.print_scores()
rf_ml.print_model_features()

# Lasso
lasso_ml.print_scores()
lasso_ml.print_model_features()

# Viewing Voting Emsemble Results
ve.print_scores()

# Plotting Graphics
ve.print_precision_recall_curve()
ve.print_roc_curve()
