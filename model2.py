# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd

from IPython.core.display import display, HTML
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio


import seaborn as sns
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib
import warnings

# Configure Jupyter Notebook
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)
display(HTML("<style>div.output_scroll { height: 35em; }</style>"))

reload(plt)
# %matplotlib inline
# %config InlineBackend.figure_format ='retina'

warnings.filterwarnings('ignore')

# configure plotly graph objects
pio.renderers.default = 'iframe'
# pio.renderers.default = 'vscode'

pio.templates["ck_template"] = go.layout.Template(
    layout_colorway = px.colors.sequential.Viridis,
#     layout_hovermode = 'closest',
#     layout_hoverdistance = -1,
    layout_autosize=False,
    layout_width=800,
    layout_height=600,
    layout_font = dict(family="Calibri Light"),
    layout_title_font = dict(family="Calibri"),
    layout_hoverlabel_font = dict(family="Calibri Light"),
#     plot_bgcolor="white",
)

# pio.templates.default = 'seaborn+ck_template+gridon'
pio.templates.default = 'ck_template+gridon'
# pio.templates.default = 'seaborn+gridon'
# pio.templates

df = pd.read_csv('ai4i2020.csv')
# df = pd.read_csv('ai4i2020.csv')

"""First up is just to eyeball the data. It seems that there are two indices: the index and ProductID. We can drop those. There is a Type which is categorical and the remainder are numeric. The last five feastures are all failure modes, so they will not be evaluated in this notebook."""

df.head()

"""There are no apparent missing values, but we'll check these out carefully to make sure"""

df.info()

df.describe(include='all').T

"""making sure that there are no missing values hidden as a question mark"""

df.replace("?",np.nan,inplace=True)

"""turn all columns into float to make processing later easier"""

for column in df.columns:
    try:
        df[column]=df[column].astype(float)
    except:
        pass

"""just check the descriptions for the numeric features. None missing and on apparent outliers"""

# show the numeric characters
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.describe(include='all').T

"""Another verification whether there are any missing features. I see none."""

plt.figure(figsize=(15,15))
plot_kws={"s": 1}
sns.heatmap(df.isna().transpose(),
            cmap='cividis',
            linewidths=0.0,
           ).set_facecolor('white')

"""There are strongly correlated features namely process and air temperature. Torque and rotational speed are also strongly correlated. We can drop one of the temperatures, but the torque to rotational speed difference might be a indication of a failure, so we'll keep both.  """

plt.figure(figsize=(10,10))
threshold = 0.80
sns.set_style("whitegrid", {"axes.facecolor": ".0"})
df_cluster2 = df.select_dtypes(include=np.number).corr()
mask = df_cluster2.where((abs(df_cluster2) >= threshold)).isna()
plot_kws={"s": 1}
sns.heatmap(df_cluster2,
            cmap='RdYlBu',
            annot=True,
            mask=mask,
            linewidths=0.2,
            linecolor='lightgrey').set_facecolor('white')

#!pip install ydata_profiling

from ydata_profiling import ProfileReport

"""The profiling report follows to look for outliers, missing values, and distributions. We can see that the data is imbalanced."""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# profile = ProfileReport(df,
#                         title="Predictive Maintenance",
#                         dataset={"description": "This profiling report was generated for Deepika Ambade"},
#                         explorative=True,
#                        )
# profile

"""Drop the indices as these have no predictive power"""

df.drop(['UDI','Product ID'],axis=1,inplace=True)

"""Drop the failure modes, as we're only interested whether something is a failure. I guess that you'll build a model for each failure mode if it comes down to that."""

df.drop(['TWF','HDF','PWF','OSF','RNF'],axis=1,inplace=True)

"""Drop the type, as this dominates too strongly on type = L."""

df.drop(['Type'],axis=1,inplace=True)

"""The remaining features"""

list(df)

"""turn categorical information into numeric"""

df = pd.get_dummies(df,drop_first=True)

features = list(df.columns)

for feature in features:
    print(feature + " - " + str(len(df[df[feature].isna()])))

"""Just another confirmation of how badly imbalanced the data is. We'll need to oversample in this case to get a better prediction."""

df_group = df.groupby(['Machine failure'])
df_group.count()

df[df['Machine failure'].isna()]
""

df_numeric.fillna(df_numeric.mean(),inplace=True)

for feature in features:
    try:
        df[feature].fillna(df[feature].mean(),inplace=True)
    except:
        try:
            df[feature].fillna(df[feature].mode(),inplace=True)
        except:
            pass

df.describe(include='all').T



# Feature Selection
from sklearn.feature_selection import SelectKBest, chi2

best_features = SelectKBest(score_func=chi2,k='all')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
fit = best_features.fit(X,y)

df_scores=pd.DataFrame(fit.scores_)
df_col=pd.DataFrame(X.columns)

feature_score=pd.concat([df_col,df_scores],axis=1)
feature_score.columns=['feature','score']
feature_score.sort_values(by=['score'],ascending=True,inplace=True)

fig = go.Figure(go.Bar(
            x=feature_score['score'][0:21],
            y=feature_score['feature'][0:21],
            orientation='h'))

fig.update_layout(title="Top 20 Features",
                  height=1200,
                  showlegend=False,
                 )

fig.show()

Selected_Features = []
import statsmodels.api as sm

def backward_regression(X, y, initial_list=[], threshold_out=0.5, verbose=True):
    """To select feature with Backward Stepwise Regression

    Args:
        X -- features values
        y -- target variable
        initial_list -- features header
        threshold_out -- pvalue threshold of features to drop
        verbose -- true to produce lots of logging output

    Returns:
        list of selected features for modeling
    """
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f"worst_feature : {worst_feature}, {worst_pval} ")
        if not changed:
            break
    Selected_Features.append(included)
    print(f"\nSelected Features:\n{Selected_Features[0]}")


# Application of the backward regression function on our training data
backward_regression(X, y)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X.head()
feature_names = list(X.columns)
np.shape(X)

np.shape(X)

len(feature_names)


# import library
from imblearn.over_sampling import SMOTE, SVMSMOTE,RandomOverSampler
oversamp = RandomOverSampler(random_state=0)
# oversamp = SMOTE(n_jobs=-1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 0,
                                                    stratify=y)
# X_train,y_train = oversamp.fit_resample(X_train, y_train)

"""There are no distinct outliers, therefore a simple minmax scaler suffices."""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,roc_auc_score,matthews_corrcoef
from sklearn.metrics import ConfusionMatrixDisplay # Import ConfusionMatrixDisplay instead of plot_confusion_matrix

import time
model_performance = pd.DataFrame(columns=['Accuracy','Recall','Precision','F1-Score','MCC score','time to train','time to predict','total time'])


# Commented out IPython magic to ensure Python compatibility.
# %%time
# from sklearn.linear_model import LogisticRegression
# start = time.time()
# model = LogisticRegression().fit(X_train,y_train)
# end_train = time.time()
# y_predictions = model.predict(X_test) # These are the predictions from the test data.
# end_predict = time.time()

accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')
MCC = matthews_corrcoef(y_test, y_predictions)
# ROC_AUC = roc_auc_score(y_test, y_predictions, average='weighted')
ROC_AUC = roc_auc_score(y_test, model.predict_proba(X_test)[:,1], average='weighted')

print("Accuracy: "+ "{:.2%}".format(accuracy))
print("Recall: "+ "{:.2%}".format(recall))
print("Precision: "+ "{:.2%}".format(precision))
print("F1-Score: "+ "{:.2%}".format(f1s))
print("MCC: "+ "{:.2%}".format(MCC))
print("ROC AUC score: "+ "{:.2%}".format(ROC_AUC))
print("time to train: "+ "{:.2f}".format(end_train-start)+" s")
print("time to predict: "+"{:.2f}".format(end_predict-end_train)+" s")
print("total: "+"{:.2f}".format(end_predict-start)+" s")
model_performance.loc['Logistic'] = [accuracy, recall, precision, f1s,MCC,end_train-start,end_predict-end_train,end_predict-start]

plt.rcParams['figure.figsize']=5,5
sns.set_style("white")

# Instead of plot_confusion_matrix, use ConfusionMatrixDisplay.from_estimator: because it was giving me error
from sklearn.metrics import ConfusionMatrixDisplay  # Ensure this is imported
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)

plt.show()

"""<a id='4_3'></a>
## <p style="padding: 8px;color:white; display:fill;background-color:#aaaaaa; border-radius:5px; font-size:100%"> <b>Decision Tree</b>
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from sklearn.tree import DecisionTreeClassifier
# start = time.time()
# model = DecisionTreeClassifier().fit(X_train,y_train)
# end_train = time.time()
# y_predictions = model.predict(X_test) # These are the predictions from the test data.
# end_predict = time.time()

accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')
MCC = matthews_corrcoef(y_test, y_predictions)
# ROC_AUC = roc_auc_score(y_test, y_predictions, average='weighted')
ROC_AUC = roc_auc_score(y_test, model.predict_proba(X_test)[:,1], average='weighted')

print("Accuracy: "+ "{:.2%}".format(accuracy))
print("Recall: "+ "{:.2%}".format(recall))
print("Precision: "+ "{:.2%}".format(precision))
print("F1-Score: "+ "{:.2%}".format(f1s))
print("MCC: "+ "{:.2%}".format(MCC))
print("ROC AUC score: "+ "{:.2%}".format(ROC_AUC))
print("time to train: "+ "{:.2f}".format(end_train-start)+" s")
print("time to predict: "+"{:.2f}".format(end_predict-end_train)+" s")
print("total: "+"{:.2f}".format(end_predict-start)+" s")
model_performance.loc['Decision Tree'] = [accuracy, recall, precision, f1s,MCC,end_train-start,end_predict-end_train,end_predict-start]

#!pip install scikit-learn --upgrade

plt.rcParams['figure.figsize']=5,5
sns.set_style("white")

# Instead of plot_confusion_matrix, use ConfusionMatrixDisplay.from_estimator: because it was giving me error
from sklearn.metrics import ConfusionMatrixDisplay  # Ensure this is imported
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)

plt.show()

plt.rcParams['figure.figsize']=10,10
sns.set_style("white")
feat_importances = pd.Series(model.feature_importances_, index=feature_names)
feat_importances = feat_importances.groupby(level=0).mean()
feat_importances.nlargest(20).plot(kind='barh').invert_yaxis()
sns.despine()
plt.show()

"""<a id='4_5'></a>
## <p style="padding: 8px;color:white; display:fill;background-color:#aaaaaa; border-radius:5px; font-size:100%"> <b>Random Forest</b>
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from sklearn.ensemble import RandomForestClassifier
# start = time.time()
# model = RandomForestClassifier(n_estimators = 100,n_jobs=-1,random_state=0,bootstrap=True,).fit(X_train,y_train)
# end_train = time.time()
# y_predictions = model.predict(X_test) # These are the predictions from the test data.
# end_predict = time.time()

accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')
MCC = matthews_corrcoef(y_test, y_predictions)
# ROC_AUC = roc_auc_score(y_test, y_predictions, average='weighted')
ROC_AUC = roc_auc_score(y_test, model.predict_proba(X_test)[:,1], average='weighted')

print("Accuracy: "+ "{:.2%}".format(accuracy))
print("Recall: "+ "{:.2%}".format(recall))
print("Precision: "+ "{:.2%}".format(precision))
print("F1-Score: "+ "{:.2%}".format(f1s))
print("MCC: "+ "{:.2%}".format(MCC))
print("ROC AUC score: "+ "{:.2%}".format(ROC_AUC))
print("time to train: "+ "{:.2f}".format(end_train-start)+" s")
print("time to predict: "+"{:.2f}".format(end_predict-end_train)+" s")
print("total: "+"{:.2f}".format(end_predict-start)+" s")
model_performance.loc['Random Forest'] = [accuracy, recall, precision, f1s,MCC,end_train-start,end_predict-end_train,end_predict-start]

plt.rcParams['figure.figsize']=5,5
sns.set_style("white")

# Instead of plot_confusion_matrix, use ConfusionMatrixDisplay.from_estimator: because it was giving me error
from sklearn.metrics import ConfusionMatrixDisplay  # Ensure this is imported
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)

plt.show()

"""<a id='4_6'></a>
## <p style="padding: 8px;color:white; display:fill;background-color:#aaaaaa; border-radius:5px; font-size:100%"> <b>Gradient Boosting Classifier</b>
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from sklearn.ensemble import GradientBoostingClassifier
# start = time.time()
# model = GradientBoostingClassifier().fit(X_train,y_train)
# end_train = time.time()
# y_predictions = model.predict(X_test) # These are the predictions from the test data.
# end_predict = time.time()

accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')
MCC = matthews_corrcoef(y_test, y_predictions)
# ROC_AUC = roc_auc_score(y_test, y_predictions, average='weighted')
ROC_AUC = roc_auc_score(y_test, model.predict_proba(X_test)[:,1], average='weighted')

print("Accuracy: "+ "{:.2%}".format(accuracy))
print("Recall: "+ "{:.2%}".format(recall))
print("Precision: "+ "{:.2%}".format(precision))
print("F1-Score: "+ "{:.2%}".format(f1s))
print("MCC: "+ "{:.2%}".format(MCC))
print("ROC AUC score: "+ "{:.2%}".format(ROC_AUC))
print("time to train: "+ "{:.2f}".format(end_train-start)+" s")
print("time to predict: "+"{:.2f}".format(end_predict-end_train)+" s")
print("total: "+"{:.2f}".format(end_predict-start)+" s")
model_performance.loc['Gradient Boosting Classifier'] = [accuracy, recall, precision, f1s,MCC,end_train-start,end_predict-end_train,end_predict-start]

plt.rcParams['figure.figsize']=5,5
sns.set_style("white")

# Instead of plot_confusion_matrix, use ConfusionMatrixDisplay.from_estimator: because it was giving me error
from sklearn.metrics import ConfusionMatrixDisplay  # Ensure this is imported
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)

plt.show()

plt.rcParams['figure.figsize']=10,10
sns.set_style("white")
feat_importances = pd.Series(model.feature_importances_, index=feature_names)
feat_importances = feat_importances.groupby(level=0).mean()
feat_importances.nlargest(20).plot(kind='barh').invert_yaxis()
sns.despine()
plt.show()


import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             ConfusionMatrixDisplay)

# Assuming X_train, X_test, y_train, y_test are already defined

# Parameters
max_epochs = 1000  # Number of epochs

start = time.time()
model = MLPClassifier(
    hidden_layer_sizes=(1,),  # Single-layer, single perceptron
    activation='relu',  # relu activation function
    solver='sgd',  # Stochastic gradient descent solver
    max_iter=max_epochs,  # Maximum number of iterations (epochs)
    random_state=0,
    verbose=True  # Enables logging of training progress
)

# Training the model
model.fit(X_train, y_train)
end_train = time.time()

# Predictions
y_predictions = model.predict(X_test)
end_predict = time.time()

# Metrics Calculation
accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')
MCC = matthews_corrcoef(y_test, y_predictions)

# Handle potential error for roc_auc_score if predict_proba isn't available.
try:
    ROC_AUC = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], average='weighted')
except AttributeError:
    print("Warning: predict_proba not available for ROC AUC calculation. Using default value.")
    ROC_AUC = 0  # Or handle differently

# Print results
print("Epochs trained:", model.n_iter_)
print("Accuracy:", "{:.2%}".format(accuracy))
print("Recall:", "{:.2%}".format(recall))
print("Precision:", "{:.2%}".format(precision))
print("F1-Score:", "{:.2%}".format(f1s))
print("MCC:", "{:.2%}".format(MCC))
print("ROC AUC score:", "{:.2%}".format(ROC_AUC))
print("Time to train:", "{:.2f}".format(end_train - start), "s")
print("Time to predict:", "{:.2f}".format(end_predict - end_train), "s")
print("Total time:", "{:.2f}".format(end_predict - start), "s")

# Confusion Matrix
plt.rcParams['figure.figsize'] = (5, 5)
sns.set_style("white")
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

"""after 36 epochs, training loss didnt improve, so best number of epochs for this is

solution: adjusting tol, increasing max_iter, or changing the optimizer to improve convergence.
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay

# Assuming X_train, X_test, y_train, y_test are already defined

start = time.time()
model = MLPClassifier(hidden_layer_sizes=(1,),  # Single-layer, single perceptron
                      activation='relu',  # ReLU activation function
                      solver='sgd',  # Stochastic gradient descent solver
                      max_iter=500,  # Increased max_iter to allow more epochs
                      tol=1e-6,  # Lower tolerance for better convergence
                      random_state=0,
                      verbose=True).fit(X_train, y_train)
end_train = time.time()

# Predictions
y_predictions = model.predict(X_test)
end_predict = time.time()

# Metrics Calculation
accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')
MCC = matthews_corrcoef(y_test, y_predictions)

# Handle potential error for roc_auc_score if predict_proba isn't available.
try:
    ROC_AUC = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], average='weighted')
except AttributeError:
    print("Warning: predict_proba not available for ROC AUC calculation. Using default value.")
    ROC_AUC = 0  # Default value

# Display Results
print(f"Epochs trained: {model.n_iter_}")
print("Accuracy:", "{:.2%}".format(accuracy))
print("Recall:", "{:.2%}".format(recall))
print("Precision:", "{:.2%}".format(precision))
print("F1-Score:", "{:.2%}".format(f1s))
print("MCC:", "{:.2%}".format(MCC))
print("ROC AUC score:", "{:.2%}".format(ROC_AUC))
print("Time to train:", "{:.2f}".format(end_train - start), "s")
print("Time to predict:", "{:.2f}".format(end_predict - end_train), "s")
print("Total time:", "{:.2f}".format(end_predict - start), "s")

# Confusion Matrix
plt.rcParams['figure.figsize'] = (5, 5)
sns.set_style("white")
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

"""The 500-epoch model significantly improved the **ROC AUC score (78.21% vs. 30.72%)** but took **8.63s vs. 0.63s** to train, with all other metrics remaining the same. The 35-epoch model trained much faster but failed to generalize well in distinguishing between classes.

**Single layer Multi Perceptron**

Some suggestions:

max_iter (Epochs)

1000 (Default)	Best for complex problems or convergence issues

500	Moderate training time, usually enough for good results

200	If you want faster training (but risk underfitting)

50-100	Quick testing to check model setup
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    matthews_corrcoef, roc_auc_score, ConfusionMatrixDisplay
)

# Assuming X_train, X_test, y_train, y_test are already defined

start = time.time()
model = MLPClassifier(
    hidden_layer_sizes=(10,),  # Single hidden layer with 10 perceptrons
    activation='relu',  # Better for multi-perceptron networks
    solver='adam',  # Adaptive optimizer for better convergence
    max_iter=1000,  # Number of epochs
    verbose=True,  # Logs training progress
    random_state=0
).fit(X_train, y_train)

end_train = time.time()

# Predictions
y_predictions = model.predict(X_test)
end_predict = time.time()

# Metrics Calculation
accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')
MCC = matthews_corrcoef(y_test, y_predictions)

# Handle potential error for ROC AUC
try:
    ROC_AUC = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], average='weighted')
except AttributeError:
    print("Warning: predict_proba not available for ROC AUC calculation. Using default value.")
    ROC_AUC = 0

# Display Metrics
print("Accuracy:", "{:.2%}".format(accuracy))
print("Recall:", "{:.2%}".format(recall))
print("Precision:", "{:.2%}".format(precision))
print("F1-Score:", "{:.2%}".format(f1s))
print("MCC:", "{:.2%}".format(MCC))
print("ROC AUC score:", "{:.2%}".format(ROC_AUC))
print("Time to train:", "{:.2f}".format(end_train - start), "s")
print("Time to predict:", "{:.2f}".format(end_predict - end_train), "s")
print("Total time:", "{:.2f}".format(end_predict - start), "s")
print("Epochs Run:", model.n_iter_)  # Number of epochs actually completed

# Confusion Matrix Plot
plt.rcParams['figure.figsize'] = 5, 5
sns.set_style("white")
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

"""**Multilayer Multi Perceptron**"""

import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming X_train, X_test, y_train, y_test are already defined

start = time.time()
model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),  # Three hidden layers (Multi-Layer)
    activation='relu',  # Best for multi-layer perceptrons
    solver='adam',  # Adaptive learning rate optimization
    max_iter=500,  # Number of epochs
    verbose=True,  # Show training progress
    random_state=0
).fit(X_train, y_train)
end_train = time.time()

y_predictions = model.predict(X_test)
end_predict = time.time()

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')
MCC = matthews_corrcoef(y_test, y_predictions)

# Handle potential error for roc_auc_score if predict_proba isn't available
try:
    ROC_AUC = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], average='weighted')
except AttributeError:
    print("Warning: predict_proba not available for ROC AUC calculation. Using default value.")
    ROC_AUC = 0  # Default value

# Print results
print("Accuracy:", "{:.2%}".format(accuracy))
print("Recall:", "{:.2%}".format(recall))
print("Precision:", "{:.2%}".format(precision))
print("F1-Score:", "{:.2%}".format(f1s))
print("MCC:", "{:.2%}".format(MCC))
print("ROC AUC score:", "{:.2%}".format(ROC_AUC))
print("Time to train:", "{:.2f}".format(end_train - start), "s")
print("Time to predict:", "{:.2f}".format(end_predict - end_train), "s")
print("Total time:", "{:.2f}".format(end_predict - start), "s")

# Plot Confusion Matrix
plt.rcParams['figure.figsize'] = (5, 5)
sns.set_style("white")
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

import plotly.graph_objects as go

# Assuming 'model_performance' DataFrame is already populated

fig = go.Figure()

# Add traces for each model
fig.add_trace(go.Bar(x=model_performance.index, y=model_performance['Accuracy'], name='Accuracy'))
fig.add_trace(go.Bar(x=model_performance.index, y=model_performance['Recall'], name='Recall'))
fig.add_trace(go.Bar(x=model_performance.index, y=model_performance['Precision'], name='Precision'))
fig.add_trace(go.Bar(x=model_performance.index, y=model_performance['F1-Score'], name='F1-Score'))
fig.add_trace(go.Bar(x=model_performance.index, y=model_performance['MCC score'], name='MCC Score'))
fig.add_trace(go.Bar(x=model_performance.index, y=model_performance['time to train'], name='Training Time'))
fig.add_trace(go.Bar(x=model_performance.index, y=model_performance['time to predict'], name='Prediction Time'))
fig.add_trace(go.Bar(x=model_performance.index, y=model_performance['total time'], name='Total Time'))


fig.update_layout(title='Model Performance Comparison',
                  xaxis_title='Model',
                  yaxis_title='Score',
                  barmode='group')  # Grouped bar chart

fig.show()

import pickle

# Assuming all the variables you want to save are in the current environment
#  e.g., model, X_train, y_train, etc.
# Create a dictionary to hold these variables
variables_to_save = {
    'model': model,
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test,
    # Add any other variables here
}


# Specify the filename for the pickle file
filename = 'variables.pkl'

# Open the file in write binary mode ('wb') and save the variables using pickle
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Load the dataset
df = pd.read_csv('ai4i2020.csv')

# Prepare the features and target
X = df.drop(['Machine failure'], axis=1)
y = df['Machine failure']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create MLP model
mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Example architecture
    max_iter=500,
    activation='relu',
    solver='adam',
    random_state=42
)

# Fit the model
mlp_model.fit(X_train_scaled, y_train)

# Save training data
with open('mlp_training_data.pkl', 'wb') as f:
    pickle.dump({
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test
    }, f)

# Save the model
with open('mlp_model.pkl', 'wb') as f:
    pickle.dump(mlp_model, f)

print("Pickle files created successfully!")