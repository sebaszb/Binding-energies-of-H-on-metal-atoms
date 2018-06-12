import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
##############################################################################
fullAtributes = ["Delectrons", "x_pos", "y_pos", "Electronegativity", "electron_afinity", "Num_electrons", "Calculated_radius", "Bindingenergy"]
#XAtributes = ["Delectrons", "x_pos", "y_pos", "Electronegativity", "electron_afinity", "Num_electrons", "Calculated_radius"]
XAtributes = ["Delectrons", "y_pos", "Electronegativity", "electron_afinity", "Num_electrons", "Calculated_radius"]
####################FUNCTIONS#################################################

def load_housing_data(housing_path="/Users/zuluags/Documents/Vanderbilt/v-work/project_10/Binding-energies-of-H-on-metal-atoms"):
    csv_path = os.path.join(housing_path, "data.csv")
    return pd.read_csv(csv_path)

def splitTrainTest(df, size, state):
    SPtrain_df, SPtest_df = train_test_split(df, test_size=size, random_state=state)
    SPtrain_X = SPtrain_df[XAtributes]                                            
    SPtrain_y = SPtrain_df[['Bindingenergy']]
    SPtest_X = SPtest_df[XAtributes]
    SPtest_y = SPtest_df[['Bindingenergy']]
    return SPtrain_X, SPtrain_y, SPtest_X, SPtest_y

def scaleData(SDtrain_X, SDtest_X, SDtrain_y, SDtest_y):
    scalerTrain = StandardScaler()
    scalerTest = StandardScaler()
    SDtrain_X =  scalerTrain.fit_transform(SDtrain_X)
    SDtest_X =  scalerTest.fit_transform(SDtest_X)
    SDtrain_y = SDtrain_y.values.ravel()                                              #need to add ravel to avoid warnings
    SDtest_y = SDtest_y.values.ravel()
    return SDtrain_X, SDtest_X, SDtrain_y, SDtest_y

def ramdomForestGo(RFtrain_X, RFtrain_y, RFtest_X, RFXAtributes):
#    Reg = RandomForestRegressor(n_estimators = 90, max_features = 5)
    Reg = RandomForestRegressor()
    Reg.fit(RFtrain_X, RFtrain_y)
    RFtrain_predictions = Reg.predict(RFtrain_X)
    RFtest_predictions = Reg.predict(RFtest_X)
    print(RFXAtributes)
    print(Reg.feature_importances_)
    return RFtrain_predictions, RFtest_predictions    

def linearRegresGo(LRtrain_X, LRtrain_y, LRtest_X, LRXAtributes):
    Reg = linear_model.LinearRegression()
    Reg.fit(LRtrain_X, LRtrain_y) 
    LRtrain_predictions = Reg.predict(LRtrain_X)
    LRtest_predictions = Reg.predict(LRtest_X)
    print(LRXAtributes)
    print(Reg.coef_)
    return LRtrain_predictions, LRtest_predictions



def plotResults(predictions, y):
    plt.scatter(predictions, y)
    plt.scatter(y, y)
    plt.xlabel('Predicted')
    plt.ylabel('Acutual')
    plt.show()
    lin_mse = mean_squared_error(y, predictions)
    lin_rmse = np.sqrt(lin_mse)
    print('Root mean squared error:', lin_rmse)
    lin_mae =  mean_absolute_error(y, predictions)
    print('Mean absolute error:', lin_mae)
    return lin_mae   

##############################################################################


pTable_raw = load_housing_data()                                              #Get the raw nada
pTable_num = pTable_raw[fullAtributes]                                        #Get rid of the Element column
train_X, train_y, test_X, test_y = splitTrainTest(pTable_num, 0.3, 0)            #split the set into train and test
train_X, test_X, train_y, test_y = scaleData(train_X, test_X, train_y, test_y)#scale the data and gets rid of the name of the columns

##################Now the regressors##########################################
#train_predictions, test_predictions = ramdomForestGo(train_X, train_y, test_X, XAtributes)
train_predictions, test_predictions = linearRegresGo(train_X, train_y, test_X, XAtributes)

#plotResults(train_predictions, train_y)
plotResults(test_predictions, test_y)

