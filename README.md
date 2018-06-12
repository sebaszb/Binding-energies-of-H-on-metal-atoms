# H Binding energies
prediction of H binding energies on metal atoms 

The binding energy of H on metal atoms is an important descriptor of the
catalytic activity (Hydrogen evolution reaction).

The code predicts the binding energy of H on D metals based only on the
properties of the atoms (electronegativity, number of electrons, radius,
etc). The regression techniques implemented in the code are:

linear_model 

RandomForestRegressor


The code read the attributes and binding energies from the file "data.csv".
Then divide the data into training(70%) and testing(30%). Then trains on
the training set, and predict the binding energy on the testing set. It
prints the mean absolute error and plots a graph showing how the
predictions on the binding energy differ from the DFT calculated values
(y=x straight line).

While the data set is small, the mean absolute error is 0.08 eV. Which is
very close to the level of noise of DFT calculations. 


