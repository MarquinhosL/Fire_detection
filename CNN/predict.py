from keras.models import load_model
import matplotlib.image as processimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import cv2


class Prediction(object):
    def __init__(self,ModelFile,PredictFile,isFire,Width=100,Height=100):
        self.ModelFile =ModelFile
        self.PredictFile =PredictFile
        self.isFire =isFire
        self.Width=Width
        self.Height =Height
        
    
    #预测
    def Predict(self):
        #引入model
        model = load_model(self.ModelFile)
        """
        #处理照片格式和大小
        img_open = Image.open(self.PredictFile)
        cov_RGB = img_open.convert('RGB')
        new_img = cov_RGB.resize((self.Width,self.Height),Image.BILINEAR)
        new_img.save(self.PredictFile)
        """
        #用opencv处理
        #pre_img = cv2.imread(self.PredictFile)
        dim=(100,100)
        resized=cv2.resize(self.PredictFile,dim,interpolation = cv2.INTER_AREA)
        
        #处理照片shape
        
        #resized = processimage.imread(self.PredictFile)
        
        img_to_array =np.array(resized)/255.0
        
        img_to_array = img_to_array.reshape(-1,100,100,3)
        print('image reshaped')

        #预测图片
        prediction = model.predict(img_to_array)
        
        print(prediction)
        return np.argmax(prediction)

        #读取概率
        count = 0
        for i in prediction[0]:
            percent = '%.2f%%'%(i*100)
            print(self.isFire[count],'概率',percent)
            count+=1

check_result=['no fire','find fire']
#Pred = Prediction(PredictFile='E:/FireDetection/image/face/pic00005.jpg',ModelFile='./firefinder0329.h5',isFire=check_result)
#num = Pred.Predict()
#print(num)