from abc import ABCMeta, abstractmethod

class Dataset(object):
    
    __metaclass__ = ABCMeta
    def __init__(self, train_data_dir, validation_data_dir, batch_size = 32):
        self._train_data_dir = train_data_dir
        self._validation_data_dir = validation_data_dir
        self._batch_size = batch_size
       
    @abstractmethod
    def load(self):
        pass