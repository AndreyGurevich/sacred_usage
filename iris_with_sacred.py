from numpy.random import permutation
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn import svm, datasets

observer = MongoObserver(
    url='mongodb://user:pass@host/omniboard?authMechanism=SCRAM-SHA-256',
    db_name='omniboard')
ex = Experiment('iris_rbf_svm')
ex.observers.append(observer)
ex.captured_out_filter = apply_backspaces_and_linefeeds


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
    return float(clf.score(iris.data[90:],
                           iris.target[90:]))
