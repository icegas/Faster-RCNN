import numpy as np
from keras.layers import Conv2D
from keras.models import load_model, Model
from keras import backend as K
from keras.utils import np_utils
from RPNDataset import RPNDataset

def main():
    K.set_image_dim_ordering('th')
    img_width = 320; img_height = 240; img_channels = 1;
    train_data_dir = '/root/Desktop/DeepLearning/Faster-RCNN/code/RPNDataset/train/'; validation_data_dir = None;
    subsampling_ratio = 4; sizes = [32, 64];

    #3 - because for each anchor we use 3 sliding boxes. Ex: for 32px ([32, 32], [16, 64], [64, 16])
    anchors = len(sizes) * 3
    dataset = RPNDataset(img_width, img_height, img_channels, train_data_dir, validation_data_dir, subsampling_ratio, sizes, iou = 0.6)

    X_train, Y_train_reg, Y_train_cls = dataset.load()

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255

    print(np.shape(X_train))
    print(np.shape(Y_train_cls))
    print(np.shape(Y_train_reg))

    #240, 320
    BaseNetwork = load_model('/root/Desktop/DeepLearning/Faster-RCNN/code/models/base_CNN_model.h5')
    intermediate = Conv2D(512, (3, 3), padding = 'same', activation = 'relu')(BaseNetwork.output)

    rpn_cls_output = Conv2D(1 * anchors, (1, 1), padding = 'same', activation = 'relu', name = 'rpn_cls')(intermediate)
    rpn_reg_output = Conv2D(4 * anchors, (1, 1), padding = 'same', name = 'rpn_reg')(intermediate)
    print(intermediate)
    print(rpn_cls_output)
    rpn = Model(inputs = BaseNetwork.input, outputs = [rpn_cls_output, rpn_reg_output])

    rpn.compile(optimizer = 'rmsprop', loss = {'rpn_cls' : 'binary_crossentropy', 'rpn_reg' : 'categorical_crossentropy'})
    rpn.fit(X_train, {'rpn_reg' : Y_train_reg, 'rpn_cls' : Y_train_cls}, epochs = 10)

    rpn.save('models/rpn_model.h5')

#def cls_loss(y_true, y_pred):
#    loss = (10 / len(Y)) * np.sum(Y * (y_true)**2 - (y_pred)**2)
#    return loss


if __name__ == '__main__':
    main()
