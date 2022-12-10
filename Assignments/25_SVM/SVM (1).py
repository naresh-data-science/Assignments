import pandas as pd #Here pandas library is import to perform some functions
import numpy as np#Here numpy library is import to perform some functions
#here reading the dataset using the pandas library and assigning to the variable called letter
letters = pd.read_csv("C:\\Datasets_BA\\Python Scripts\\letters.csv")
letters.describe()#here describing the assigning variable dataset to get min,max,mean,meidan,50%,25%,75% values in dataframe format

from sklearn.svm import SVC #here we are importing the svc library and using sklearn for model building
from sklearn.model_selection import train_test_split # here we are importing train_test split to divide our dataset into train data ,test data for model selection purpose.
#here we are assigning the values for train and test 
train,test = train_test_split(letters, test_size = 0.20)
#here we have train_X,Train_y ,test_X,test_y(x:-independent varaible,y:-dependent variables)
train_X = train.iloc[:, 1:]#independent variables of train_X
train_y = train.iloc[:, 0]#dependent variables of train_y
test_X  = test.iloc[:, 1:]#independent variables of test_X
test_y  = test.iloc[:, 0]#dependent variables of test_y


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")#here we make kernel as linear on svc and assigning to the new variable model_linear
model_linear.fit(train_X, train_y)#here we are fitting the train data 
pred_test_linear = model_linear.predict(test_X)#here we are predicting the output variable using the test_X input variables

np.mean(pred_test_linear == test_y)#here we are finding the average of comparing the predicting output values with test_y output variable 

# kernel = rbf
model_rbf = SVC(kernel = "rbf")#here we are using different type kernel function rbf and assigning to the new variable model_rbf
model_rbf.fit(train_X, train_y)#again we are fitting the train data input variables and the output variables
pred_test_rbf = model_rbf.predict(test_X)#here predicting the output variable using the exisitng test input variables

np.mean(pred_test_rbf==test_y)#here we fingind the mean value while comparing the predicting output with exisiting test_y output variable.

