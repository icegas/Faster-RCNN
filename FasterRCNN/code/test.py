from RPNDataset import RPNDataset
import numpy as np
import cv2

image_path_list = []
train_data_dir = '/root/Desktop/DeepLearning/FasterRCNN/code/RPNData/train/X/'
test_data_dir = '/root/Desktop/DeepLearning/FasterRCNN/code/RPNData/test/'
rpnDataset = RPNDataset(100, 100, 1, train_data_dir, test_data_dir, 2)
X_train = rpnDataset.load()

cv2.imshow('image', X_train[1])
cv2.waitKey(0)
cv2.destroyAllWindows()