# H Binding energies
prediction of H binding energies on metal atoms 

The binding energy of H on metal atoms is an important descriptor of the
catalytic activity (Hydrogen evolution reaction).

The code predicts the binding energy of H on atoms siting on top of MoC
surface. The code uses only the
properties of the atoms (electronegativity, number of electrons, radius,
etc).

The code read the attributes and binding energies from the file "data.csv".
Then divide the data into training(80%) and testing(20%). Then trains on
the training set, and predict the binding energy on the testing set. It
prints the mean absolute error and plots a graph showing how the
predictions on the binding energy differ from the DFT calculated values
(y=x straight line).

While the data set is small, the mean absolute error is 0.08 eV. Which is
very close to the level of noise of DFT calculations. 

Once the model is trained, the code reads the file group1.csv, which is
the data for the elements we are interested in, and predicts the binding
energy of H on each one of this atoms. The results are in eV. The goal is
to find atoms where the H binds with a binding energy close to the one
seen in the case of Pt (-0.4 eV).

