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

    predict_image = frame
    size = frame.shape
    Pred = predict.Prediction(PredictFile=predict_image,ModelFile='./firefinder0329.h5',isFire=predict.check_result)
    index = Pred.Predict()
    if index is 1:
        cv.rectangle(frame, (0, 0), (size[0],size[1]), (0, 0, 255), 2)
        
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


