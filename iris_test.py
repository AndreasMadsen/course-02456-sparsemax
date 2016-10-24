from sklearn import datasets
import sparsemax_classifier


iris = datasets.load_iris()
X = iris.data
y = iris.target


test = sparsemax_classifier.SparseMaxClassifier(X, y)

for epoch in range(0, 100):
	test.update_weights()

print(test.predict(X[20,]))
print(y[20,])