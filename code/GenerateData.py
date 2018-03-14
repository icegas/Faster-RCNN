import cv2
import os
from tqdm import tqdm
from keras.datasets import mnist
import numpy as np

def main():
    base_image_width, base_image_height = 320, 240
    background_images_dir = '/root/Downloads/iccv09Data/images/'    
    image_path_list, num_width, num_height, num_x, num_y = [], [], [], [], []
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print("Reading names of images in the folder")
    for file in tqdm(os.listdir(background_images_dir)):
        image_path_list.append(os.path.join(background_images_dir, file))
    
    counter = 0
    print("Generate images")
    for imagePath in tqdm(image_path_list):
        image = cv2.imread(imagePath, 0)
        image = cv2.resize(image, (base_image_width, base_image_height))
        numbers_of_nums = np.random.randint(1, 6)#uniform(1, 6)
        number_of_nums, num_height, num_width, num_x, num_y = checkOnSame(numbers_of_nums, base_image_width, base_image_height)

        num_x = np.asarray(num_x, dtype = int)
        num_y = np.asarray(num_y, dtype = int)
        num_height = np.asarray(num_height, dtype = int)
        num_width = np.asarray(num_width, dtype = int)

        for i in range(number_of_nums):
            image[num_y[i] : num_y[i] + num_height, num_x[i] : num_x[i] + num_width[i] ] = np.asarray(X_train[counter])
            counter += 1
            cv2.imwrite('/root/Desktop/DeepLearning/Faster-RCNN/code/RPNDataset/train/X/'+counter+'.jpg', image)
        
def checkOnSame(numbers_of_nums, base_image_width, base_image_height):

    num_height, num_width, num_x, num_y = [], [], [], []
    for nums in range(numbers_of_nums):
            num_width.append(np.random.randint(20, 50))
            num_height.append(np.random.randint(20, 50))
            num_x.append(np.random.randint(0, base_image_width - num_width[nums]))
            num_y.append(np.random.randint(0, base_image_height - num_height[nums]))
        
    #watch list impliment
    flags, pixel_arr, arr_one, arr_two = [], [], [], []
    for nums in range(numbers_of_nums):
        for j  in range(num_width[nums]):

            for i in range(num_height[nums]):
                arr_one.append(num_y[nums] + i)

            arr_two.append(arr_one)
        pixel_arr.append(arr_two)
    
    
    #if numbers have intersections on the base image remove them from the list
    for nums in range(numbers_of_nums):
        for i in range(nums + 1, numbers_of_nums):
            if(i == nums):
                continue
            for pix_x in range(num_width[nums]):
                for pix_y in range(num_height[nums]):
                    if(pixel_arr[nums][pix_x][pix_y] == pixel_arr[i][pix_x][pix_y]):
                        flags.append(nums)
    
    for i in range(numbers_of_nums):
        num_height = [i for j,i in enumerate(num_height) if j not in flags]
        num_width = [i for j,i in enumerate(num_width) if j not in flags]
        num_x = [i for j,i in enumerate(num_x) if j not in flags]
        num_y = [i for j,i in enumerate(num_y) if j not in flags]
    numbers_of_nums = len(num_x)
    return numbers_of_nums, num_height, num_width, num_x, num_y


if __name__ =="__main__":
    main()