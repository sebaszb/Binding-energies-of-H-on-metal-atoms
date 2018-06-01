import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
#################################
train_file = "/Users/zuluags/Documents/Vanderbilt/v-work/project_10/train.csv"
test_file = "/Users/zuluags/Documents/Vanderbilt/v-work/project_10/test.csv"
table = pd.read_csv(train_file, header = 0)
test_table = pd.read_csv(test_file, header = 0)
#print(table)

def relation1(table): #gets the table
    x = list(table["Delectrons"] * table["Electronegativity"])
    y = list(table["Electronegativity"])
    z = list(table["Bindingenergy"])
    return x, y, z
    
def relation2(table): #gets the table
    x = list(table["Delectrons"] * table["Electronegativity"])
    y = list(table["Delectrons"])
    z = list(table["Bindingenergy"])
    return x, y, z

def relation3(table): #gets the table
    x = list(table["Delectrons"])
    y = list(table["Delectrons"])
    z = list(table["Bindingenergy"])
    return x, y, z

def relation4(table): #gets the table
    x = list(table["Electronegativity"])
    y = list(table["Electronegativity"])
    z = list(table["Bindingenergy"])
    return x, y, z

def relation5(table): #gets the table
    x = list(table["Delectrons"])
    y = list(table["Electronegativity"])
    z = list(table["Bindingenergy"])
    return x, y, z
##############################################################################

x, y, z = relation1(table)
xy = list(zip(x, y))
tx, ty, tz = relation1(test_table)
txy = list(zip(tx, ty))

#reg = linear_model.LinearRegression()
#reg = linear_model.Lasso(alpha = .5)
#reg = linear_model.BayesianRidge()
reg = linear_model.Ridge (alpha = .5)
q = reg.fit (xy, z)
print("coeficients for linear: ", q.coef_)
print("y intercept for linear: ", q.intercept_)
#print("R2 score for linear: ", q.score(xy, z))
print("R2 score for linear: ", q.score(txy, tz))


zpredicted = q.predict(txy)

plt.scatter(tz, zpredicted)
plt.xlabel('DFT binding energy (eV)')
plt.ylabel('Predicted binding energy (eV)')
plt.show()
