import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from statsmodels.graphics.regressionplots import influence_plot
# import statsmodels.formula.api as smf
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import pylab as pl
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from  sklearn import preprocessing
from sklearn.tree import  DecisionTreeClassifier



df= pd.read_csv("drug200.csv", delimiter=',')
print(df.head())


X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

y = df["Drug"]
y[0:5]

from sklearn import preprocessing
# set value sex field to 0,1
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

# set value blood per field to 0,1,2
le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])
X[0:5]

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))

# criterion is : index , farsi is meyar
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
# train data with X_trainset,y_trainset
drugTree.fit(X_trainset,y_trainset)


# Prediction
predTree = drugTree.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])

# Evaluation
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# Visualization
from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
import graphviz

dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
out = tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset),
                           filled=True,  special_characters=True, rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')