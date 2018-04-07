#from RPNDataset import RPNDataset
from keras.models import load_model
import numpy as np
import cv2
import theano
from matplotlib import pyplot as plt
from keras import backend as K

def get_activation(model, layer, img):
    get_activation = K.function([model.layers[0].input],
                [model.layers[layer].output] )
    activation = get_activation([img])
    return activation

def layer_to_visualize(layer, img):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img)
    convolutions = np.squeeze(convolutions)


    print ('Shape of conv:', convolutions.shape)

    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12,8))
    for i in range(len(convolutions)):
       # if(convolutions[i] < 255 and convolutions[i] > 0):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(convolutions[i], cmap='gray')
        #for x in range(len(convolutions[i])):
        #    for y in range(len(convolutions[i][i])):
        #        if convolutions[x][y] > 100:
        #            print("convolution: {}, x: {}, y: {} \n".format(i, x, y))


img = cv2.imread('/root/Desktop/DeepLearning/Faster-RCNN/code/RPNDataset/train/X/0.jpg', 0)
img = cv2.resize(img, (320, 240) )
img = np.reshape(img, [1, 1, 320, 240])

model = load_model('/root/Desktop/DeepLearning/Faster-RCNN/code/models/rpn_model.h5')
print("model loaded\n")

print(model.layers[6])
print(model.layers[7])
layer_to_visualize(model.layers[6], img)
layer_to_visualize(model.layers[7], img)
plt.show()



'''image_path_list = []
train_data_dir = '/root/Desktop/DeepLearning/FasterRCNN/code/RPNData/train/X/'
test_data_dir = '/root/Desktop/DeepLearning/FasterRCNN/code/RPNData/test/'
rpnDataset = RPNDataset(100, 100, 1, train_data_dir, test_data_dir, 2)
X_train = rpnDataset.load()

print(np.shape(X_train))
'''
