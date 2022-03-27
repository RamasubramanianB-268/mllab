from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
iris = datasets.load_iris()
X_train,X_test,Y_train,Y_test = train_test_split(iris.data,iris.target)
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X_train,Y_train)
model.score
acc1=metrics.accuracy_score(Y_test,model.predict(X_test))
print(acc1)
from sklearn.mixture import GaussianMixture
model2 = GaussianMixture(n_components=3)
model2.fit(X_train,Y_train)
model2.score
metrics
acc2=metrics.accuracy_score(Y_test,model.predict(X_test))
print(acc2)
