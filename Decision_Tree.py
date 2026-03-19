#importing the needed packages:
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

#loading in the dataset:
student_alcohol_df = pd.read_csv('student-mat.csv')

#selecting the propper/key features we are using:
selected_features = [
    #the parents background:
    'Medu', 'Fedu', 'Mjob', 'Fjob',
    #the living conditions:
    'address', 'Pstatus', 'traveltime', 'internet',
    #the family demographics:
    'famsup', 'famsize', 'famrel'
]

#creating the target variable which is G3:
target_variable = 'G3'

#now we have make sure we are not including everything else (removing the leakage)
x = student_alcohol_df[selected_features]
y= student_alcohol_df[target_variable]

#converting the categorical values into numerical:
#converting address: (U-values=1 and R-values=0)
x["adress"] = x["adress"].map({"U":1, "R":0})
#famsize: (LE3=1 and GT3=0)
x["famsize"] = x["famsize"].map({"LE3":1, "GT3":0})




#train/test splitting into 80/20
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
