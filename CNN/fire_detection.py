import cv2 as cv
import time
import datetime
import predict

img = []
catpure = cv.VideoCapture("E:/FireDetection/visivideo/barbeq.avi")

firstFrame =None
while(catpure.isOpened()):
    ret,frame = catpure.read()#逐帧捕捉
    if not ret:#如果返回值为空，则到了视频最后，需要返回。
        break

    
    #对于首帧的处理
    #gray = cv.resize(frame,(500,500),interpolation = cv.INTER_AREA)
    #重新划大小
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) 
    #转换成灰度图像
    gray = cv.GaussianBlur(gray,(21,21),0)
    #进行高斯模糊，模糊程度与高斯矩阵和标准差有关

    if firstFrame is None:
        firstFrame = gray
        continue
    
    frameDiff = cv.absdiff(firstFrame,gray)
    #计算每一帧与起始帧的差值

    values,thresh = cv.threshold(frameDiff,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    #图像二值化
    #选取一个合适的二值阈值，最好是两者差度到了25以上。

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,2))
    dst = cv.dilate(thresh,kernel)
    #膨胀
    #cv.imshow('ss',dst)
    (_,cnts, _) = cv.findContours(dst.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv.contourArea(c) < 1800: # 对于较小矩形区域，选择忽略
            continue
        flat=1#设置一个标签，当有运动的时候为1
        # 计算轮廓的边界框，在当前帧中画出该框
        (x, y, w, h) = cv.boundingRect(c)
        predict_image = frame[y:y+h,x:x+w]
        Pred = predict.Prediction(PredictFile=predict_image,ModelFile='./firefinder0329.h5',isFire=predict.check_result)
        index = Pred.Predict()
        if index is 1:
            cv.rectangle(frame, (x, y), (x + w, y+ h), (0, 0, 255), 2)
        
    cv.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    #cv.imshow("Frame Delta", frameDelta)
 
    cv.imshow("fps", frame)
    #cv.imshow("Thresh", thresh)
 
    key = cv.waitKey(1) & 0xFF
 
    # 如果q键被按下，跳出循环
    ch = cv.waitKey(1)
    if key == ord("q"):
        break


