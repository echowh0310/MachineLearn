from sklearn.decomposition import PCA
import plotly.express as px

import pandas as pd
# glass = pd.read_csv('D:/wh/AAAproject/Algorithm/Data/irisdata/iris.csv')
glass = pd.read_csv('D:/wh/AAAproject/Algorithm/Data/Breast/BreastTissue.csv')
# Dimensionality reduction to 3 dimensions
pca = PCA(n_components=3)#指定降维的目标维度
# glass_pca = pca.fit_transform(glass.iloc[:, :-1])#选取除最后一列的数据集
glass_pca = pca.fit_transform(glass.iloc[:, 2:])#选取第三列到最后一列的数据集

# 3D scatterplot
fig = px.scatter_3d(x=glass_pca[:, 0],
                    y=glass_pca[:, 1],
                    z=glass_pca[:, 2],
                    color=glass.iloc[:, 1])#选取第二列作为类别列
fig.show()