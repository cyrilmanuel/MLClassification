from sklearn import svm
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
from sklearn import svm

# visualisation des données d'entrée comme point X et Y.
x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

# visualisation des donnes sur un graphe.
# plt.scatter(x, y)
# plt.show()

# restructuration pour les données sous forme X = f(x,y)
temp = [[1, 2],
        [5, 8],
        [1.5, 1.8],
        [8, 8],
        [1, 0.6],
        [9, 11]]

X = np.array(temp)
X.reshape(6, 2)

# creation des étiquettes pour chaque valeur de X
y = [0, 1, 0, 1, 0, 1]

# création du SVC avec le type de noyaux, et la valeur de C.
clf = svm.SVC(kernel='linear', C=1.0)

# passe les données pour le training
clf.fit(X, y)

# teste la classification d'une nouvelle valeur
print(clf.predict(np.array([0.58, 0.76]).reshape(1, 2)))

# visualisation sur le svm
w = clf.coef_[0]
#print(w)

a = -w[0] / w[1]

xx = np.linspace(0, 12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.legend()
plt.show()
