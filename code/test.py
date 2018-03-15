#from RPNDataset import RPNDataset
import numpy as np
f = '/root/Desktop/DeepLearning/Faster-RCNN/code/RPNDataset/train/Y/xywh0.txt'
with open(f, 'r') as fl:
    arr = [int(i) for i in fl.read().split()]
    #arr = np.reshape(arr, (-1, 4))
print(arr)
'''image_path_list = []
train_data_dir = '/root/Desktop/DeepLearning/FasterRCNN/code/RPNData/train/X/'
test_data_dir = '/root/Desktop/DeepLearning/FasterRCNN/code/RPNData/test/'
rpnDataset = RPNDataset(100, 100, 1, train_data_dir, test_data_dir, 2)
X_train = rpnDataset.load()

print(np.shape(X_train))
'''