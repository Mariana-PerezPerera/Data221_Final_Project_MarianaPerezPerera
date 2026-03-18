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
