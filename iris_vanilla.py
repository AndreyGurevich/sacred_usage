from numpy.random import permutation
from sklearn import svm, datasets





C = 1.0
gamma = 0.7



iris = datasets.load_iris()
perm = permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
clf = svm.SVC(C, 'rbf', gamma=gamma)
clf.fit(iris.data[:90],
        iris.target[:90])
print(clf.score(iris.data[90:],
                iris.target[90:]))