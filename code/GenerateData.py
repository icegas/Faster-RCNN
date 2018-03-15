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
    
    #coordinate x,y then width height
    regression_file_name = '/root/Desktop/DeepLearning/Faster-RCNN/code/RPNDataset/train/Y/xywh'
    classifiaction_file_name = '/root/Desktop/DeepLearning/Faster-RCNN/code/RPNDataset/train/Y/bf.txt'
      
    print("Reading names of images in the folder")
    for file in tqdm(os.listdir(background_images_dir)):
        image_path_list.append(os.path.join(background_images_dir, file))
    
    counter, img_counter = 0, 0
    print("Generate images")
    for imagePath in tqdm(image_path_list):
        image = cv2.imread(imagePath, 0)
        image = cv2.resize(image, (base_image_width, base_image_height))
        numbers_of_nums = np.random.randint(1, 20)
        numbers_of_nums, num_height, num_width, num_x, num_y = checkOnSame(numbers_of_nums, base_image_width, base_image_height)
 
        for i in range(numbers_of_nums): 
            tmp_img = cv2.resize(X_train[counter], (num_width[i], num_height[i])) 
            image[num_y[i] : num_y[i] + num_height[i], num_x[i] : num_x[i] + num_width[i] ] = tmp_img
            counter += 1
            with open(regression_file_name + str(img_counter) + ".txt", 'a+') as fr:
                fr.write(" {} {} {} {} ".format(num_x[i], num_y[i], num_width[i], num_height[i]))
                 
        cv2.imwrite('/root/Desktop/DeepLearning/Faster-RCNN/code/RPNDataset/train/X/'+str(img_counter)+'.jpg', image)
        img_counter += 1
        
def checkOnSame(numbers_of_nums, base_image_width, base_image_height):

    num_height, num_width, num_x, num_y = [], [], [], []
    for nums in range(numbers_of_nums):
            num_width.append(np.random.randint(20, 50))
            num_height.append(np.random.randint(20, 50))
            num_x.append(np.random.randint(0, base_image_width - num_width[nums]))
            num_y.append(np.random.randint(0, base_image_height - num_height[nums]))
    flags = []
    for nums in range(numbers_of_nums):
        for i in range(numbers_of_nums):#(nums + 1, numbers_of_nums):
            if(num_x[nums] + num_width[nums] > num_x[i] + num_width[i]):
                if(num_x[nums] < num_x[i] ):
                    if(num_y[nums] > num_y[i] ):
                        if(num_y[nums] < num_y[i] + num_height[i]):
                            flags.append(nums)
                    else:
                        flags.append(nums)
                else:
                    if(num_x[nums] + num_width[nums] > num_x[i] and num_x[nums] < num_x[i]):
                        flags.append(nums)
            else:
                if(num_x[nums] + num_width[nums] > num_x[i] and num_x[nums] < num_x[i]):
                    if(num_y[nums] < num_y[i] + num_height[i] and (num_y[nums] > num_y[i] or num_y[nums] < num_y[i])):
                        flags.append(nums)

   

    num_height = [i for j,i in enumerate(num_height) if j not in flags]
    num_width = [i for j,i in enumerate(num_width) if j not in flags]
    num_x = [i for j,i in enumerate(num_x) if j not in flags]
    num_y = [i for j,i in enumerate(num_y) if j not in flags]
    numbers_of_nums = len(num_x)
    return numbers_of_nums, num_height, num_width, num_x, num_y


if __name__ =="__main__":
    main()