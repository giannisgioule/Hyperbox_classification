# Hyperbox classification

This classifier creates hypeboxes, each one representing a single class in the data. The algorithm optimizes the coordinates and dimensions of the hyperboxes in order to minimize the number of misclassified samples.

## Getting Started
This implementation uses third-party software. The optimisation is based on GAMS and the CPLEX solver to love all the mixed integer linear optimisation problems, while the rest of the implementation is based on python.

### Prerequisites

This implementation uses the following libraries-software

|Software|Version
|--------|-------
|GAMS    |v.24.7.1
|python|v.3.6.5
|gdxpds|v.1.1.0
|pandas|v.0.23.4
|numpy|v.1.15.4
|scikit-learn|v.0.21.0


## Example

The implementation follows the scikit-learn estimator syntax. The method requires the used to scale the input variables to the range of [0,1].

```python
from GAMS_Hyperbox import GamsHyperboxClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

X=load_breast_cancer()['data']
y=load_breast_cancer()['target']

#=================================================
# The method requires the user to scale the data
# to the range of (0,1)
#=================================================
scaler=MinMaxScaler(feature_range=(0,1))
scaler.fit(X)
scaled_X=scaler.transform(X)
#=================================================

X_train,X_validation,y_train,y_validation=train_test_split(scaled_X,y,test_size=0.2,random_state=1)

hc=GamsHyperboxClassifier()
hc.fit(X_train,y_train)

y_pred=hc.predict(X_validation)

y_validation=list(map(str,y_validation))

class_accuracy=accuracy_score(y_true=y_validation,y_pred=y_pred,normalize=True)
print("Classification accuracy: {} %".format(round(class_accuracy*100,1)))
```