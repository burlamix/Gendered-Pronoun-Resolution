import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from model9 import model9
from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import LinearSVC

from brew.base import Ensemble, EnsembleClassifier
from brew.combination.combiner import Combiner

import pandas as pd



val_pathx = "../datasets/gap-light.tsv"

mode_9 = model9("model_9/weights")


val_path = pd.read_csv(val_pathx, delimiter="\t")#pd.read_csv(test_path, delimiter="\t")

print(val_path.shape)


# create your Ensemble clf1 can be an EnsembleClassifier object too
ens = Ensemble(classifiers=[mode_9, mode_9, mode_9]) 
 
# create your Combiner (combination rule)
# it can be 'min', 'max', 'majority_vote' ...
cmb = Combiner(rule='mean')
 
# and now, create your Ensemble Classifier
ensemble_clf = EnsembleClassifier(ensemble=ens, combiner=cmb)
 
# assuming you have a X, y data you can use
ensemble_clf.fit(val_path, val_path)

print("-----------d-----------")
ensemble_clf.predict(val_path)
ensemble_clf.predict_proba(val_path)
 
# creating a new ensemble of ensembles
ens = Ensemble(classifiers=[clf1,ensemble_clf])
ensemble_ens = EnsembleClassifier(ensemble=ens, combiner=cmb)
 
# and you can use it in the same way as a regular ensemble
ensemble_ens.fit(val_path, val_path)
ensemble_ens.predict(val_path)
ensemble_ens.predict_proba(val_path)


exit()


mode_9 = model9("model_9/weights")

#check_estimator(mode_9)  # passes

val_path = "../datasets/gap-light.tsv"




clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',                          random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()






eclf1 = VotingClassifier(estimators=[        ('lr', mode_9), ('rf', mode_9), ('gnb', mode_9)], voting='hard')

#eclf1 = eclf1.fit(val_path, val_path)

#print(eclf1.predict(X))
'''

np.array_equal(eclf1.named_estimators_.lr.predict(X),               eclf1.named_estimators_['lr'].predict(X))

eclf2 = VotingClassifier(estimators=[        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],        voting='soft')
eclf2 = eclf2.fit(X, y)
print(eclf2.predict(X))

eclf3 = VotingClassifier(estimators=[       ('lr', clf1), ('rf', clf2), ('gnb', clf3)],       voting='soft', weights=[2,1,1],       flatten_transform=True)
eclf3 = eclf3.fit(X, y)
print(eclf3.predict(X))

print(eclf3.transform(X).shape)

+++++++++++++++++++++


from brew.base import Ensemble, EnsembleClassifier
brew.combination.combiner import Combiner 
 
# create your Ensemble clf1 can be an EnsembleClassifier object too
ens = Ensemble(classifiers=[clf1, clf2, clf2]) 
 
# create your Combiner (combination rule)
# it can be 'min', 'max', 'majority_vote' ...
cmb = Combiner(rule='mean')
 
# and now, create your Ensemble Classifier
ensemble_clf = EnsembleClassifier(ensemble=ens, combiner=cmb)
 
# assuming you have a X, y data you can use
ensemble_clf.fit(X, y)
ensemble_clf.predict(X)
ensemble_clf.predict_proba(X)
 
# creating a new ensemble of ensembles
ens = Ensemble(classifiers=[clf1,ensemble_clf])
ensemble_ens = EnsembleClassifier(ensemble=ens, combiner=cmb)
 
# and you can use it in the same way as a regular ensemble
ensemble_ens.fit(X, y)
ensemble_ens.predict(X)
ensemble_ens.predict_proba(X)



'''