import cv2
import os
from tqdm import tqdm
from keras.datasets import mnist
import numpy as np

def main():
    base_image_width, base_image_height = 320, 240
    background_images_dir = '/root/Desktop/Dataset/iccv09Data/images/'    
    image_path_list, num_width, num_height, num_x, num_y = [], [], [], [], []
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print("Reading names of images in the folder")
    for file in tqdm(os.listdir(background_images_dir)):
        image_path_list.append(os.path.join(background_images_dir, file))
    
    print("Generate images")
    for imagePath in tqdm(image_path_list):
        image = cv2.imread(imagePath, 0)
        numbers_of_nums = np.random.uniform(1, 6)

        for nums in numbers_of_nums:
            num_width.append(np.random.uniform(20, 100))
            num_height.append(np.random.uniform(20, 100))
            num_x.append(np.random.uniform(0, base_image_width - num_width[nums]))
            num_y.append(np.random.uniform(0, base_image_height - num_height[nums]))
        
        flags, pixel_arr, arr_one, arr_two = [], [], [], []
        for nums in numbers_of_nums:
            for j  in range(num_width[nums]):

                for i in range(num_height[nums]):
                    arr_one.append(num_y[nums] + i)

                arr_two.append(arr_one)
            pixel_arr.append(arr_two)

        #if numbers have intersections on the base image remove them from the list
        for nums in numbers_of_nums:
            for i in numbers_of_nums:
                if(i == nums):
                    continue
                for pix_x in range(num_width[nums]):
                    for pix_y in range(num_height[nums]):
                        if(pixel_arr[nums][pix_x][pix_y] == pixel_arr[i][pix_x][pix_y]):
                            flags.append(nums)
       
        for i in numbers_of_nums:
           if i in flags: 
               num_height[i] = -1
               num_width[i] = -1
               num_x = -1
               num_y = -1  
                                                  
        for i in numbers_of_nums:
            
            if num_width[i] == -1 and i < np.shape(num_height):
                num_width.remove(i)
                num_height.remove(i)
                num_x.remove(i)
                num_y.remove(i)
        

    
    
   


if __name__ =="__main__":
    main()