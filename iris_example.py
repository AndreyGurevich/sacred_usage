from numpy.random import permutation
from sklearn import svm, datasets
from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('iris_rbf_svm')
observer = MongoObserver(
    url='mongodb://user:pass@host/omniboard?authMechanism=SCRAM-SHA-256',
    db_name='omniboard')

@ex.config
def cfg():
  C = 1.0
  gamma = 0.7

@ex.automain
def run(C, gamma):
  iris = datasets.load_iris()
  perm = permutation(iris.target.size)
  iris.data = iris.data[perm]
  iris.target = iris.target[perm]
  clf = svm.SVC(C, 'rbf', gamma=gamma)
  clf.fit(iris.data[:90],
          iris.target[:90])
  ex.log_scalar("Some metric", 0.85)
  return clf.score(iris.data[90:],
                   iris.target[90:])