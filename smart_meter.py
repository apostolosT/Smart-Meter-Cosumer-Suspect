from minisom import MiniSom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

suspicious = pd.read_csv('suspicious.csv')
suspicious = suspicious.drop('LCLid', axis = 1)
suspicious['new_id'] = suspicious.index
X = suspicious.values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

som = MiniSom(x = 7, y = 5, input_len = 8, sigma = 1.0, learning_rate = 0.5)

som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,  #plotting at the
         w[1] + 0.5)  #centre of the block
         
show()

mappings = som.win_map(X)