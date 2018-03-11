from keras.preprocessing.image import ImageDataGenerator
from DatasetGen import Dataset
import cv2
import os
from tqdm import tqdm

class RPNDataset(Dataset):

    def __init__(self, img_width, img_height, img_channels, train_data_dir, validation_data_dir, batch_size = 32):
        self._img_width = img_width
        self._img_heigh = img_height
        self._img_channels = img_channels
        Dataset.__init__(self, train_data_dir, validation_data_dir, batch_size)

    def load(self):
        X_train, Y_train, X_test, Y_test = [],[],[],[]
        image_path_list = []

        print("Reading names of images in the folder")
        for file in tqdm(os.listdir(self._train_data_dir)):
            image_path_list.append(os.path.join(self._train_data_dir, file))

        print("Reading data from images")
        for imagePath in tqdm(image_path_list):
            image = cv2.imread(imagePath, 0)
            X_train.append(image)

        return X_train
