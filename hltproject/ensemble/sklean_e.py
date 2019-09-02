import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from model9 import model9
from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import LinearSVC


mode_9 = model9("model_9/weights")

#check_estimator(mode_9)  # passes

val_path = "../datasets/gap-light.tsv"




clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',                          random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()






eclf1 = VotingClassifier(estimators=[        ('lr', mode_9), ('rf', mode_9), ('gnb', mode_9)], voting='hard')
eclf1 = eclf1.fit(val_path, val_path)
print(eclf1.predict(X))
'''

np.array_equal(eclf1.named_estimators_.lr.predict(X),               eclf1.named_estimators_['lr'].predict(X))

eclf2 = VotingClassifier(estimators=[        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],        voting='soft')
eclf2 = eclf2.fit(X, y)
print(eclf2.predict(X))

eclf3 = VotingClassifier(estimators=[       ('lr', clf1), ('rf', clf2), ('gnb', clf3)],       voting='soft', weights=[2,1,1],       flatten_transform=True)
eclf3 = eclf3.fit(X, y)
print(eclf3.predict(X))

print(eclf3.transform(X).shape)'''