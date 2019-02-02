import numpy as np
from math import sqrt
import tensorflow as tf
from xml.etree.ElementTree import Element, SubElement, Comment, tostring, ElementTree
import io
import classnames_id_label__pb2
import random
from scipy import ndarray
from scipy.misc import imread, imsave, imresize
import skimage as sk
from skimage import transform
from skimage import util
from PIL import Image
from io import BytesIO
from skimage.color import rgb2gray
import traceback
import cv2


class Utilities:

    def __init__(self):
        self.total_predictions = 0
        self.correct_predictions = 0
        
    def calc_accuracy(self, correct_pedictions_batch, predictions_batch):
        self.total_predictions = self.total_predictions + predictions_batch
        self.correct_predictions += np.sum(correct_pedictions_batch)
        accuracy = (self.correct_predictions / float(self.total_predictions)) * 100
        return accuracy , np.sum(correct_pedictions_batch)

    def weights_to_image_grid(self, kernel, pad=1):
        '''Visualize conv. filters as an image (mostly for the 1st layer).
        Arranges filters into a grid, with some paddings between adjacent filters.
        Args:
            kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
            pad:               number of black pixels around each filter (between them)
        Return:
            Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
        '''

        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1: print('Who would enter a prime number of filters')
                    return (i, int(n / i))

        (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
        print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel.get_shape()[0] + 2 * pad
        X = kernel.get_shape()[1] + 2 * pad

        channels = kernel.get_shape()[2]

        # put NumKernels to the 1st dimension
        x = tf.transpose(x, (3, 0, 1, 2))
        # organize grid on Y axis
        x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

        # switch X and Y axes
        x = tf.transpose(x, (0, 2, 1, 3))
        # organize grid on X axis
        x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

        # back to normal order (not combining with the next step for clarity)
        x = tf.transpose(x, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x = tf.transpose(x, (3, 0, 1, 2))

        # scaling to [0, 255] is not necessary for tensorboard
        return x

    def object_boundary_label_xml(self, label, boundary_dict, output_dir):
        box = Element('box')
        label_element = Element('label')
        label_element.text = label

        boundary = Element('boundary')

        box.append(label_element)
        box.append(boundary)

        xmin = Element('xmin')
        xmin.text = boundary_dict[0]

        ymin = Element('ymin')
        ymin.text = boundary_dict[1]

        xmax = Element('xmax')
        xmax.text = boundary_dict[2]

        ymax = Element('ymax')
        ymax.text = boundary_dict[3]

        boundary.append(xmin)
        boundary.append(ymin)
        boundary.append(xmax)
        boundary.append(ymax)

        # ElementTree(box).write(output_dir + '/' + label + '.xml')
        return box

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def int64_list_feature(self,value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def float_list_feature(self,value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
        
    def bytes_list_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def create_tf_example_object_detection(self,image,x1,y1,x2,y2,width, height, class_name, class_id):
        try:

            encoded_image_string = cv2.imencode('.png', image)[1].tostring()

            xmins = [x1 / width]
            xmaxs = [(x2) / width]
            ymins = [y1 / height]
            ymaxs = [(y2) / height]
            
            if ((xmins[0] > 1.1) or (xmaxs[0] > 1.1) or (ymins[0] > 1.1) or (ymaxs[0] > 1.1)):
              print(xmins,ymins,xmaxs,ymaxs)

            features_dict = {
                        'image/height': self.int64_feature(height),
                        'image/width': self.int64_feature(width),
                        'image/encoded': self.bytes_feature(encoded_image_string),
                        'image/format': self.bytes_feature('png'.encode('utf8')),
                        'image/object/bbox/xmin': self.float_list_feature(xmins),
                        'image/object/bbox/ymin': self.float_list_feature(ymins),
                        'image/object/bbox/xmax': self.float_list_feature(xmaxs),
                        'image/object/bbox/ymax': self.float_list_feature(ymaxs),
                        'image/object/class/text': self.bytes_list_feature([class_name.encode('utf8')]),
                        'image/object/class/label': self.int64_list_feature([class_id]),

                        }
            example = tf.train.Example(features=tf.train.Features(feature=features_dict))
        except Exception:
            traceback.print_exc()        
        return example

    def add_class_to_label_map(self, class_name, class_map):
        id_if_present = self.is_class_name_present(class_map,class_name)
        if id_if_present == 0:
            item_single = class_map.item.add()
            item_single.name = str(class_name)
            item_single.id = len(class_map.item)
            return item_single.id
        else:
            return id_if_present
                
        
    def open_file(self, file_name, mode):
        return open(file_name, mode)

    def is_class_name_present(self, class_map, class_name):
        presence = 0
        for item_single in class_map.item:
            try:
                if item_single.name.lower()==class_name.lower():
                    presence=item_single.id
                    break
            except Exception:
                continue
 
        return presence

    def create_augmented_images(self,elements_list,creation_count):
        train_list = []
        test_list = []
        for element in elements_list:
            
            if np.random.choice(np.arange(0, 2), p=[0.2, 0.8]) == 1:
                train_list.append(element)
                print('train')
            else:
                test_list.append(element) 
                print('test')  

            img_original = np.asarray(element[0])
            for i in range(creation_count):
                
                #img =  self.random_rotation(img_original) if random.randint(0, 1) == 1 else img_original
                img =  self.random_noise(img_original) if random.randint(0, 1) == 1 else img_original
                #img =  self.greyscale(img) if random.randint(0, 1) == 1 else img

                #image_path = element[0] + str(i)
                #image_path = image_path.replace(".png","")
                #image_path = image_path + ".png"
                
                if np.random.choice(np.arange(0, 2), p=[0.2, 0.8]) == 1:
                    train_list.append([img,element[1],element[2],element[3],element[4],element[5],element[6],element[7],element[8]])
                    #print('train')
                else:
                    #image_path = image_path.replace("train","test")
                    test_list.append([img,element[1],element[2],element[3],element[4],element[5],element[6],element[7],element[8]])  
                    print('test')  
                #imsave(image_path, img)
        return  train_list,test_list          

   
    def random_rotation(self,image_array: ndarray):
        random_degree = random.uniform(-5, 5)
        return sk.transform.rotate(image_array, random_degree)

    def random_noise(self,image_array: ndarray):
        return sk.util.random_noise(image_array)

    def horizontal_flip(self,image_array: ndarray):
        return image_array[:, ::-1]

    def greyscale(self,image_array: ndarray):
        return rgb2gray(image_array)


