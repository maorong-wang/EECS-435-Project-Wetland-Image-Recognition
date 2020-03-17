import scipy.io
import numpy as np
def LoadData(root_path):
  matNw = scipy.io.loadmat(root_path+'Nonwetland.mat')
  matW = scipy.io.loadmat(root_path+'Wetland.mat')

  nonWetLandData=matNw.get('NNW')
  WetLandData1=matW.get('Xw_train')
  WetLandData2=matW.get('Xw_test')

  x_train = nonWetLandData[:,:,:,:90000]
  x_train=np.concatenate((x_train,WetLandData1),axis=3)
  y_train1=np.concatenate((np.zeros(90000),np.ones(97906)),axis=0)
  y_train2=np.concatenate((np.ones(90000),np.zeros(97906)),axis=0)

  x_test = nonWetLandData[:,:,:,90000:]
  x_test=np.concatenate((x_test,WetLandData2),axis=3)
  y_test1=np.concatenate((np.zeros(30000),np.ones(24477)),axis=0)
  y_test2=np.concatenate((np.ones(30000),np.zeros(24477)),axis=0)

  x_train=x_train.swapaxes(2,3)
  x_train=x_train.swapaxes(1,2)
  x_train=x_train.swapaxes(0,1)
  x_test=x_test.swapaxes(2,3)
  x_test=x_test.swapaxes(1,2)
  x_test=x_test.swapaxes(0,1)

  y_train1=np.reshape(y_train1,(y_train1.shape[0],1))
  y_train2=np.reshape(y_train2,(y_train2.shape[0],1))
  y_test1=np.reshape(y_test1,(y_test1.shape[0],1))
  y_test2=np.reshape(y_test2,(y_test2.shape[0],1))
  y_train=np.concatenate((y_train1,y_train2),axis=1)
  y_test=np.concatenate((y_test1,y_test2),axis=1)
  return (x_train,y_train,x_test,y_test)