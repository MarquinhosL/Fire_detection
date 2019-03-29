import cv2
from PIL import Image
import numpy as np 
from keras.models import Sequential
from keras.layers import Convolution2D,Flatten,MaxPooling2D,Dense,Activation
from keras.optimizers import Adam
from keras.utils import np_utils
import os

class Training(object):
    def __init__(self,batch_size,number_batch,categories,train_folder):
        self.batch_size = batch_size
        self.number_batch = number_batch
        self.categories = categories
        self.train_folder = train_folder

    def read_train_image(self,filename):
        img = Image.open(self.train_folder+'/'+filename)
        return np.array(img)


    def train(self):
        
        train_image_list = []#X_train
        train_label_list = []#Y_train
        
        for file in os.listdir(self.train_folder):
            #files_img_in_array = self.read_train_image(filename=file)
            files_img_in_array = cv2.imread(self.train_folder+'/'+file)
            train_image_list.append(files_img_in_array)
            train_label_list.append(int(file.split('_')[0]))
        
        train_image_list = np.array(train_image_list)
        train_label_list = np.array(train_label_list)

        train_label_list = np_utils.to_categorical(train_label_list,self.categories)

        train_image_list = train_image_list.astype('float32')
        train_image_list /= 255.0

    #创建神经网络模型 CNN
        model =Sequential()

    #CNN layer -1
        model.add(Convolution2D(
            input_shape =(100,100,3),#输入时shape为(100,100,3)
            filters=32,#下一层输出成32层 (100,100,32)
            kernel_size=(5,5),#每一次扫描像素范围
            padding='same',#外边距处理
            
        ))
        model.add(Activation('relu'))#激活函数
        model.add(MaxPooling2D(#池化层
            pool_size=(2,2),#每次扫描多少像素 下一层输出(50,50,32)
            strides=(2,2),
            padding='same'
        ))

    #CNN layer-2
        model.add(Convolution2D(
            
            filters=64,#下一层输出成64层 
            kernel_size=(2,2),#每一次扫描像素范围
            padding='same',#外边距处理
            
        ))
        model.add(Activation('relu'))#激活函数
        model.add(MaxPooling2D(#池化层
            pool_size=(2,2),#每次扫描多少像素 下一层输出(25,25,64)
            strides=(2,2),
            padding='same'
        ))

    #fully connected layer -1
        model.add(Flatten())#展开成1维，降维
        model.add(Dense(1024))
        model.add(Activation('relu'))
    #fully connected layer -2
        
        model.add(Dense(512))
        model.add(Activation('relu'))
    #fully connected layer -3
        
        model.add(Dense(256))
        model.add(Activation('relu'))
    #fully connected layer -4
        model.add(Dense(self.categories))
        model.add(Activation('softmax'))

    #define optimizer
        adam = Adam(lr=0.0001)#学习率
    #compile the model
        model.compile(optimizer=adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                    )

    #fire up the network
        model.fit(
            x=train_image_list,
            y=train_label_list,
            epochs=self.number_batch,
            batch_size=self.batch_size,
            verbose=1
        )

    #保存神经网络
        model.save('firefinder0329.h5')


def MAIN():
    Train = Training(batch_size=64,number_batch=30,categories=2,train_folder="E:/YOLO/train_data")
    Train.train()

if __name__ == "__main__":
    MAIN()