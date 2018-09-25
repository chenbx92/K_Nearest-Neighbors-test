import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors,datasets

n_neighbors=15
iris=datasets.load_iris()
X=iris.data[:,:2]
Y=iris.target

#KNN execute
KNNexe=neighbors.KNeighborsClassifier(n_neighbors,weights='distance')
KNNexe.fit(X,Y)

#plot the decision boundary
xmin=X[:,0].min()-1
xmax=X[:,0].max()+1
ymin=X[:,1].min()-1
ymax=X[:,1].max()+1
x,y=np.meshgrid(np.arange(xmin,xmax,0.05),np.arange(ymin,ymax,0.05))

#predict
Z=KNNexe.predict(np.c_[x.ravel(),y.ravel()])
Z=Z.reshape(x.shape)

#draw background
plt.figure()
cmap_background=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
plt.pcolormesh(x,y,Z,cmap=cmap_background)
#draw training points
cmap_points=ListedColormap(['#FF0000','#00FF00','#0000FF'])
plt.scatter(X[:,0],X[:,1],c=Y,cmap=cmap_points,s=10)
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
plt.show()

#test one point
print('predict one points class')
print('input characteristics:')
print('x=')
testx=input()
print('y=')
testy=input()
testz=KNNexe.predict([[testx,testy]])
print('result: class ')
print(testz)
print('probability is :')
print(KNNexe.predict_proba([[testx,testy]]))

plt.pcolormesh(x,y,Z,cmap=cmap_background)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=cmap_points,s=10)
plt.scatter([float(testx)],[float(testy)],c=testz,cmap=cmap_points,edgecolors='k',s=70)
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))

plt.show()