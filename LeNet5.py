import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

class LeNet5:
  def __init__(self,x_train,y_train,x_test,y_test):
    self.x_train=x_train
    self.y_train=y_train
    self.x_test=x_test
    self.y_test=y_test
    pass
  
  def train(self,Lr=0.01, Decay=1e-6, Momentum=0.9,batch_size=128, epochs=12):
    # output type
    # output images are 28*28 pixels in grey scale
    num_classes = 2
    img_rows, img_cols = 28, 28

    x_train = self.x_train
    x_test = self.x_test
    input_shape = (img_rows, img_cols, 4)

    # Convert data type into float32
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    print('Training on ',x_train.shape[0], 'samples with SGD')
    print('learning rate:',Lr,',weight_decay:',Decay,',momentum:',Momentum,',batch_size:',batch_size)
    # turn categories 0-9 into binary, convenient for training purpose
    y_train = self.y_train
    y_test = self.y_test
    # use a sequential model
    self.model = Sequential()

    # add a 2D convolutional layer with 6 filters (i.e. convolutional channels), activation function is 'relu', padding is valid, kernal is 5*5 pixels window
    # add a max pooling layer with size 2*2
    # add a 2D convolutional layer with 16 filters (i.e. convolutional channels), activation function is 'relu', padding is valid, kernal is 5*5 pixels window
    # add a max pooling layer with size 2*2
    self.model.add(Conv2D(filters=6,activation='relu', input_shape=input_shape,padding='valid',kernel_size=(5,5)))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Conv2D(filters=16, activation='relu',padding='valid',kernel_size=(5,5)))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    # add a Flatten layer
    # flatten layer converts the pooled feature map to a single column, passed to the fully connected layer. Flatten won't affect the batch size
    # add a fully connected layer with an output dimension of 120
    self.model.add(Flatten())
    self.model.add(Dense(50, activation='relu'))
    self.model.add(Dense(num_classes, activation='softmax'))

    sgd = optimizers.SGD(lr=Lr, decay=Decay, momentum=Momentum, nesterov=True)
    self.model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
    self.model.fit(x_train, y_train, batch_size, epochs, verbose=1, shuffle=True)  

  def test(self):
    x_train=self.x_train
    y_train=self.y_train
    x_test=self.x_test
    y_test=self.y_test
    print('Test on',self.x_test.shape[0], 'samples')
    score = self.model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    y_train_pred = self.model.predict(x_train)  # predict the whole training dataset
    # calculate the accuracy over the whole dataset and get information about falses
    train_accuracy, (true_labels, pred_labels) = self.calculate_performance(y_train, y_train_pred)

    print(f'Don\'t use as a metric - Original Training Dataset Accuracy: {np.round(train_accuracy*100, 3)}%')

    plt.figure(figsize=(3, 3))

    # Calculate the confusion matrix and visualize it
    train_matrix = confusion_matrix(y_pred=pred_labels, y_true=true_labels)
    sns.heatmap(data=train_matrix, annot=True, cmap='Blues', fmt=f'.0f')

    plt.title('Confusion Matrix - Training Dataset', size=15)
    plt.xlabel('Predictions', size=10);
    plt.ylabel('Labels', size=10);

    y_test_pred = self.model.predict(x_test)  # predict the whole test dataset

    # calculate the accuracy over the whole dataset and get information about falses
    test_accuracy, (true_labels, pred_labels) = self.calculate_performance(y_test, y_test_pred,)

    print(f'Test Dataset Accuracy: {np.round(test_accuracy*100, 3)}%')

    plt.figure(figsize=(3, 3))

    # Calculate the confusion matrix and visualize it
    test_matrix = confusion_matrix(y_pred=pred_labels, y_true=true_labels)
    sns.heatmap(data=test_matrix, annot=True, cmap='Blues', fmt=f'.0f')

    plt.title('Confusion Matrix - Test Dataset', size=15)
    plt.xlabel('Predictions', size=10);
    plt.ylabel('Labels', size=10);

  def calculate_performance(self, labels, pred):
    pred_cat = np.argmax(pred, axis=1)  # categorical predictions 0-9
    labels_cat = np.argmax(labels, axis=1)  # categorical labels 0-9
    
    # a boolean vector of element-wise comparison between prediction and label
    corrects = (pred_cat == labels_cat)
    
    # get the falses data
    falses_labels = labels_cat[~corrects]  # true labels of the falsely classified images - categorical
    falses_preds = pred[~corrects]  # the false predictions of the images - 10-dim prediction
     
    examples_num = labels.shape[0]  # total numbers of examples
    accuracy = np.count_nonzero(corrects) / examples_num

    return accuracy, [labels_cat, pred_cat]