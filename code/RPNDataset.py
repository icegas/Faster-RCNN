from DatasetGen import Dataset
import cv2
import os
from tqdm import tqdm
import numpy as np

class RPNDataset(Dataset):

    _sizes = []

    def __init__(self, img_width, img_height, img_channels, train_data_dir, validation_data_dir, r, sizes = [], iou = 0.5, batch_size = 32):
        self._img_width = img_width
        self._img_height = img_height
        self._img_channels = img_channels
        self._r = r #r - subsampling ratio
        self._anchors = len(sizes) * 3

        for i in sizes:
            self._sizes.extend( [i, i] )
            self._sizes.extend( [i * 2, i / 2] )
            self._sizes.extend( [i / 2, i * 2] )
        self._sizes = np.reshape(self._sizes, (-1, 2))
        self._iou = iou
        Dataset.__init__(self, train_data_dir, validation_data_dir, batch_size)

    def load(self):
        X_train, Y_train_reg, Y_train_cls = [], np.empty((0, 4)), np.empty((0, int(self._img_height / self._r)
         * int(self._img_width / self._r) * self._anchors))

        image_path_list, label_path_list = [], []
        background = 0
        foreground = 1

        self._train_data_dir = self._train_data_dir + 'X/'
        print("Reading names of images in the folder")
        for file in tqdm(os.listdir(self._train_data_dir)):
            image_path_list.append(os.path.join(self._train_data_dir, file))
        image_path_list.sort()

        print("Reading data from images")
        for imagePath in tqdm(image_path_list):
            image = cv2.imread(imagePath, 0)
            X_train.append(image)

        self._train_data_dir = self._train_data_dir[: -2] + 'Y/'
        print("Reading names of labels")
        for file in tqdm(os.listdir(self._train_data_dir)):
            label_path_list.append(os.path.join(self._train_data_dir, file))
        label_path_list.sort()

        lbl_counter = 0
        file_str = ''
        #32x32 16x64 64x16 64x64 64x128 128x64
        print("Reading data from labels")
        for labelPath in tqdm(label_path_list):
            lbl_counter += 1
            lbl_tmp_reg, lbl_tmp_cls = np.empty((0, 4)), np.array([]) #label of regression and classification outputs

            with open(labelPath, 'r') as f:
                arr = [int(i) for i in f.read().split()]
                arr = np.reshape(arr, (-1, 4) )

            for i in range(len(self._sizes)):
                for x in range( int(len(X_train[0][0]) / self._r) ):
                    for y in range( int(len(X_train[0]) / self._r) ):


                        tmp_x, tmp_y, tmp_width, tmp_height = x * self._r, y * self._r, self._sizes[i][0], self._sizes[i][1]
                        lbl_tmp_reg = np.vstack((lbl_tmp_reg, [tmp_x, tmp_y, tmp_width, tmp_height]))

                        if(self._IOU(tmp_x, tmp_y, tmp_width, tmp_height, arr)):
                            lbl_tmp_cls = np.append(lbl_tmp_cls, foreground)
                        else:
                            lbl_tmp_cls = np.append(lbl_tmp_cls, background)

            Y_train_cls = np.vstack((Y_train_cls, lbl_tmp_cls))
            Y_train_reg = np.vstack((Y_train_reg, lbl_tmp_reg)) #Y_train_reg.append(lbl_tmp_reg)

        X_train = np.reshape(X_train, [ np.shape(X_train)[0], 1, np.shape(X_train)[2], np.shape(X_train)[1] ] ).astype('float32')
        Y_train_cls = np.reshape(Y_train_cls, (-1, 1 * self._anchors, 80, 60))
        #Y_train_reg = Y_train_reg.T
        Y_train_reg = np.reshape(Y_train_reg, (-1, 4 * self._anchors, 80, 60))

        return X_train, Y_train_reg, Y_train_cls

    def _IOU(self, x, y, w, h, arr):

        flag = False
        for nums in range(len(arr)):
            if( (arr[nums][0] - x) - w * 0.5 - arr[nums][2] * 0.5 < 0):

                xA = max(x - w / 2, arr[nums][0] - arr[nums][2] / 2)
                yA = max(y - h / 2, arr[nums][1] - arr[nums][3] / 2)
                xB = min(x + w / 2, arr[nums][0] + arr[nums][2] / 2)
                yB = min(y + h / 2, arr[nums][1] + arr[nums][3] / 2)

                #compute the area if intersection
                interArea = (xB - xA + 1) * (yB - yA + 1)

                #compute area of reactangels
                areaA = h * w
                areaB = arr[nums][2] * arr[nums][3]

                #if iou > predefined number it is object
                if(float(areaA + areaB - interArea) != 0):
                    if( interArea / float(areaA + areaB - interArea) > self._iou):
                        flag = True
                        break
        return flag
