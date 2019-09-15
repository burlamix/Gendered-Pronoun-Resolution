import numpy as np
import pandas as pd
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.estimator_checks import check_estimator

from brew.base import Ensemble, EnsembleClassifier
from brew.combination.combiner import Combiner

from model9 import model_squad
from model9 import model_swag
logger = logging.getLogger ( __name__ )


test_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv"
dev_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv"
val_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv"
'''

#per trainare e testare piu velocemente, sono solo 5 esempi
test_path = "../datasets/gap-light.tsv"
dev_path = "../datasets/gap-light.tsv"
val_path = "../datasets/gap-light.tsv"
'''

test_df_prod = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(dev_path, delimiter="\t")
test_df_prod = test_df_prod.copy()
test_df_prod = test_df_prod[['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset', 'URL']]


test_examples_df = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(test_path, delimiter="\t")


logger.info ("building model ")
model_squad_inst = model_squad ("model_9/weights")
model_swag_inst = model_swag ("model_9/weights")

#check_estimator(model_squad_inst)  # passes
#check_estimator(model_swag_inst)  # passes


'''
#sklean
eclf1 = VotingClassifier(estimators=[('squas', model_squad_inst), ('swag', model_swag_inst)], voting='hard')

logger.info ("training ")
test_examples_df_2 = test_examples_df["A"]#    dirty trick
res = eclf1.fit(test_examples_df,test_examples_df_2)

logger.info ("evaluating ")
res = eclf1.predict(test_examples_df)

'''
# create your Ensemble clf1 can be an EnsembleClassifier object too
ens = Ensemble(classifiers=[model_squad_inst, model_swag_inst]) 
 
# create your Combiner (combination rule)
# it can be 'min', 'max', 'majority_vote' ...
cmb = Combiner(rule='mean')
 
# and now, create your Ensemble Classifier
ensemble_clf = EnsembleClassifier(ensemble=ens, combiner=cmb)
 
# assuming you have a X, y data you can use
#ensemble_clf.fit(test_examples_df, test_examples_df_2)

print("-----------d-----------")
res = ensemble_clf.predict(test_examples_df)


val_probas_df_e= pd.DataFrame([test_df_prod.ID, res[:,0], res[:,1], res[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()


val_probas_df_e.to_csv('stage1_ee_my_pred.csv', index=False)


test_path = "../datasets/gap-test.tsv"

print("loss ensambled ")
print(compute_loss("stage1_ee_my_pred.csv",test_path))


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




 # l'altra libreria


# create your Ensemble clf1 can be an EnsembleClassifier object too
ens = Ensemble(classifiers=[mode_9, mode_9]) 
 
# create your Combiner (combination rule)
# it can be 'min', 'max', 'majority_vote' ...
cmb = Combiner(rule='mean')
 
# and now, create your Ensemble Classifier
ensemble_clf = EnsembleClassifier(ensemble=ens, combiner=cmb)
 
# assuming you have a X, y data you can use
ensemble_clf.fit(val_path, val_path)

print("-----------d-----------")
ensemble_clf.predict(val_path)
 

