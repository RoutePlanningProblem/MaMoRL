# Importing Required Libraries

import time
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Training Dataset

#Start time
ts=time.time()

#Load Data

data=np.load("data/qData.npy")  #data directory
Qdf = pd.DataFrame(data, columns = ['Ship1_ID','State_Ship1','Ship2_ID', 'State_Ship2', 'New_State_ship1', 
                                    'New_State_ship2','feature1', 'feature2', 'feature3', 'feature4', 'feature5',
                                    'feature6', 'qVal'])


#Change the type of data to integer
cols=[i for i in Qdf.columns if i not in ["qVal"]]
for col in cols:
    Qdf[col]=pd.to_numeric(Qdf[col], downcast='integer')

    
#Remove columns that are not important to train model

Qdf= Qdf.drop(['Ship1_ID', 'Ship2_ID' ],axis = 'columns')


# Training and Test dataset

X = Qdf.drop(['qVal'],axis = 'columns')
Y = Qdf["qVal"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)


# Neural Network Model

#Build model
model = Sequential([
    Dense(5, activation='relu', input_shape=(10,)),
    Dense(1, activation='linear')])

model.summary()

model.compile(optimizer='sgd',
              loss='mean_squared_error',
             metrics=['mse', 'mae', 'mape'])

#Split test data and validation data
X_val, X_test_N, Y_val, Y_test_N = train_test_split(X_test, Y_test, test_size=0.25)

#Train model
hist = model.fit(X_train, Y_train,
          batch_size=1000, epochs=10000,
          validation_data=(X_val, Y_val))


#End time
te=time.time()

#Running Time
print(str(te-ts))


#Model evaluation
model.evaluate(X_test_N, Y_test_N)


#Visualize Loss

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#Save model
model.save('NN_Models/Neural_Net_q')

#Load model
reconstructed_model = keras.models.load_model("Neural_Net_q")


#Predict using test data
Prediction = reconstructed_model.predict(X_test_N)
print(Prediction)

