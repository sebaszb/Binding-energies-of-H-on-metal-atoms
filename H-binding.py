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

def load_data(path, File):
    csv_path = os.path.join(path, File)
    return pd.read_csv(csv_path)

def splitTrainTest(df, size, state):                                                 #splits the data into training and testing and also into features and targets
    SPtrain_df, SPtest_df = train_test_split(df, test_size=size, random_state=state) #splits the data
    SPtrain_X = SPtrain_df[XAtributes]                                               #gets the features                                     
    SPtrain_y = SPtrain_df[['Bindingenergy']]                                        #gets the targets
    SPtest_X = SPtest_df[XAtributes]                                                 #gets the features
    SPtest_y = SPtest_df[['Bindingenergy']]                                          #gets the targets
    return SPtrain_X, SPtrain_y, SPtest_X, SPtest_y

def scaleData(SDtrain_X, SDtest_X, SDtrain_y, SDtest_y, scaleData=True):
    if scaleData == True:    
        scalerTrain = StandardScaler()
        #scalerTest = StandardScaler()
        SDtrain_X =  scalerTrain.fit_transform(SDtrain_X)                     #fit for the scaler and scales the training set
        SDtest_X =  scalerTrain.transform(SDtest_X)                           #scales the test set
        SDtrain_y = SDtrain_y.values.ravel()                                  #only values
        SDtest_y = SDtest_y.values.ravel()                                    #only values
        return SDtrain_X, SDtest_X, SDtrain_y, SDtest_y, scalerTrain
    if scaleData == False:
        SDtrain_X = SDtrain_X.values#.ravel()                                 #gets only the values
        SDtest_X = SDtest_X.values#.ravel()                                   #gets only the values
        SDtrain_y = SDtrain_y.values.ravel()                                  #gets only the values
        SDtest_y = SDtest_y.values.ravel()                                    #gets only the values
        scalerTrain = 0
        return SDtrain_X, SDtest_X, SDtrain_y, SDtest_y, scalerTrain

def ramdomForestGo(RFtrain_X, RFtrain_y, RFtest_X, RFXAtributes):
    Reg = RandomForestRegressor()                                             #we are using random forest
    Reg.fit(RFtrain_X, RFtrain_y)                                             #train the regressor
    RFtrain_predictions = Reg.predict(RFtrain_X)                              #gets the predictions for training set
    RFtest_predictions = Reg.predict(RFtest_X)                                #gets the predictions for testing set
    print(RFXAtributes)                                                       #prints the features we are considering
    print(Reg.feature_importances_)                                           #how important are the atributes
    return RFtrain_predictions, RFtest_predictions, Reg    

def linearRegresGo(LRtrain_X, LRtrain_y, LRtest_X, LRXAtributes):
    Reg = linear_model.LinearRegression()                                     #use the simple linear regressor
    #Reg = linear_model.Ridge (alpha = .5)
    #Reg = linear_model.Lasso(alpha = 0.1)
    Reg.fit(LRtrain_X, LRtrain_y)                                             #train the regressor
    LRtrain_predictions = Reg.predict(LRtrain_X)                              #gets the prediction for the training set
    LRtest_predictions = Reg.predict(LRtest_X)                                #gets the predictions for the testing set
    print(LRXAtributes)                                                       #prints the features we are considering
    print(Reg.coef_)                                                          #coeficient of the linear equation
    print("intercept: ", Reg.intercept_)                                      #the intercept of the linear equation
    return LRtrain_predictions, LRtest_predictions, Reg



def plotResults(predictions, y):
    plt.scatter(predictions, y)
    plt.scatter(y, y)
    plt.xlabel('Predicted')
    plt.ylabel('Acutual')
    plt.show()                                                                #plots the prediction and the "perfect prediction"
    lin_mse = mean_squared_error(y, predictions)                              #calculate the mean squared error
    lin_rmse = np.sqrt(lin_mse)                                               #the square root
    print('Root mean squared error:', lin_rmse)
    lin_mae =  mean_absolute_error(y, predictions)                            #the mean absolute error
    print('Mean absolute error:', lin_mae)
    return lin_mae   

def predictor(PR_df, PR_Reg, PR_scalerTrain):                                 #gets the prediction on unseen data
    PR_X = PR_df[XAtributes].values                                           #gets only the values of the features
    PR_element = PR_df["Element"].values                                      #gets the name of the atoms
    PR_element = list(PR_element)
    if PR_scalerTrain != 0:                                                   #in case we are scaling the data
        PR_X = PR_scalerTrain.transform(PR_X)
    PR_predictions = PR_Reg.predict(PR_X)                                     #gets the prediction of the new data
    PR_predictions = list(PR_predictions)       
    PR_results = list(zip(PR_element, PR_predictions))                        #joins the list of atoms and the list of predictions
    return PR_results 
     
        
    
##############################################################################



pTable_raw = load_data("./", "data.csv")                                                                         #Gets the raw nada
pTable_num = pTable_raw[fullAtributes]                                                                           #Gets rid of the Element column
train_X, train_y, test_X, test_y = splitTrainTest(pTable_num, 0.2, 19)                                           #splits the set into train and test
train_X, test_X, train_y, test_y, scalerTrain = scaleData(train_X, test_X, train_y, test_y, scaleData=False)     #prepares the data. I am not scaling it
#train_X, test_X, train_y, test_y, scalerTrain = scaleData(train_X, test_X, train_y, test_y, scaleData=True)


##################Now the regressors and the performance######################
#train_predictions, test_predictions, Reg = ramdomForestGo(train_X, train_y, test_X, XAtributes)
train_predictions, test_predictions, Reg = linearRegresGo(train_X, train_y, test_X, XAtributes)

#plotResults(train_predictions, train_y)
plotResults(test_predictions, test_y)

###################Predictions on new data####################################

group1_raw = load_data("./", "group1.csv")
results = predictor(group1_raw, Reg, scalerTrain)
print(results)
