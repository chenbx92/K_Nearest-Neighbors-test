import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster,datasets
from matplotlib.colors import ListedColormap

samplesn=1000
X,Y=datasets.make_blobs(n_samples=samplesn,centers=4)
kmeansexe1=cluster.KMeans(n_clusters=3,init='k-means++',algorithm='auto')
cmap1=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
y1=kmeansexe1.fit_predict(X)

plt.figure()
plt.subplot(311)
plt.scatter(X[:,0],X[:,1],c=y1,cmap=cmap1,s=10)
plt.colorbar()

kmeansexe2=cluster.KMeans(n_clusters=4,init='k-means++',algorithm='auto')
cmap2=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF','#AAFFFF'])
y2=kmeansexe2.fit_predict(X)

plt.subplot(312)
plt.scatter(X[:,0],X[:,1],c=y2,cmap=cmap2,s=10)
plt.colorbar()

XV,YV=datasets.make_blobs(n_samples=samplesn,centers=4,cluster_std=[1,0.5,3,0.5])
y3=kmeansexe2.fit_predict(XV)
plt.subplot(313)
plt.scatter(XV[:,0],XV[:,1],c=y3,cmap=cmap2,s=10)
plt.colorbar()

plt.show()