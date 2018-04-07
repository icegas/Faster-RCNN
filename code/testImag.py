import cv2

img = cv2.imread("/root/Desktop/DeepLearning/Faster-RCNN/code/RPNDataset/train/X/0.jpg", 0)
crop = img[172 : 172 + 32, 296 : 296 + 32 ]
#img = cv2.resize(img, (240, 320))
cv2.imshow('img', img)
cv2.imshow('crop', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
