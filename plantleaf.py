from cvzone.ClassificationModule import Classifier
import cv2
import glob
cs=Classifier('keras_model.h5','labels.txt')


path = r"C:\Users\freed\Downloads\archive\test\test/*.*"
for file in glob.glob(path):
    img = cv2.imread(file)
    img=cv2.resize(img, (1020, 500))
    prediction = cs.getPrediction(img)

    cv2.imshow("IMG",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
