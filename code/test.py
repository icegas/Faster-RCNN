#from RPNDataset import RPNDataset
#import numpy as np
import cv2

img = cv2.imread('/root/Downloads/iccv09Data/images/0000051.jpg', 0)
cv2.imshow('img', img)
h, w = img.shape
print("Height: {}\nWidth: {}\nChannel:{}".format(h, w, 1))
cv2.waitKey(0)
cv2.destroyAllWindows()
'''image_path_list = []
train_data_dir = '/root/Desktop/DeepLearning/FasterRCNN/code/RPNData/train/X/'
test_data_dir = '/root/Desktop/DeepLearning/FasterRCNN/code/RPNData/test/'
rpnDataset = RPNDataset(100, 100, 1, train_data_dir, test_data_dir, 2)
X_train = rpnDataset.load()

print(np.shape(X_train))
'''