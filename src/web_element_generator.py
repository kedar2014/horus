import os as os
import traceback
from io import BytesIO
from selenium.webdriver.common.action_chains import ActionChains
import tensorflow as tf
from PIL import Image
import numpy as np
import app_facing_code as appcode
import string_int_label_map_pb2 
import util_classifier as ut
from scipy.misc import imread, imsave, imresize
from random  import shuffle
from image_container import ImageContainer
import cv2 as cv


tf.enable_eager_execution()
script_dir = os.path.dirname(__file__)
dir_path = 'data/test_1/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    os.makedirs(dir_path + 'samples/')
samples_dir = dir_path + 'samples/'
abs_image_path = os.path.join(script_dir, dir_path)
util_obj = ut.Utilities()
labelmap_file = dir_path + 'label_map.pbtxt'


class WebElementGenerator:

    def __init__(self, driver):
        self.driver = driver

    def get_suffix_from_element(self,element):
        
        if element.text:
            return element.text.replace(" ","_")
        elif element.get_attribute("name"):
            return element.get_attribute("name").replace(" ","_")  
        elif element.get_attribute("title"):
             return element.get_attribute("title").replace(" ","_")
        elif element.get_attribute("value"):
            return element.get_attribute("value").replace(" ","_")

    def generate_elements(self, url):
        train_writer = tf.python_io.TFRecordWriter(dir_path + 'train.record')
        test_writer = tf.python_io.TFRecordWriter(dir_path + 'test.record')

        try:
            if os.path.isfile(labelmap_file) == True:
                label_file = util_obj.open_file(labelmap_file, "rb")
                class_map = string_int_label_map_pb2.StringIntLabelMap()
                class_map.ParseFromString(label_file.read())
                label_file.close()
            else:
                class_map = string_int_label_map_pb2.StringIntLabelMap()

            self.driver.get(url)

            #all_visible_elements = self.driver.find_elements_by_xpath("//a[not(contains(@style,'display:none')) and not(contains(@class,'story'))] | //h2[not(contains(@style,'display:none')) and not(contains(@class,'story'))] | //button[not(contains(@style,'display:none')) and not(contains(@class,'story'))] | //span[not(contains(@style,'display:none')) and not(contains(@class,'story'))] | //input[not(contains(@style,'display:none')) and not(contains(@class,'story'))]")
            all_visible_elements = self.driver.find_elements_by_xpath("//label | //input") 
            

            for element in all_visible_elements:
                elements_list = []
                location = element.location_once_scrolled_into_view
                location = element.location
                size = element.size
                x1_orig = location['x'] * 2
                y1_orig = location['y'] * 2
                x2_orig = x1_orig + size['width'] * 2
                y2_orig = y1_orig + size['height'] * 2
                # words = element.text.split()
                if x2_orig != x1_orig and y2_orig != y1_orig:
                    
                    # writing class to label_map.pbtxt

                    suffix = self.get_suffix_from_element(element)
                    if not suffix:
                        continue
                    class_name = element.tag_name + '_' + suffix
                    class_id = util_obj.add_class_to_label_map(class_name, class_map)

                    png = self.driver.get_screenshot_as_png()
                    pil_image = Image.open(BytesIO(png))
                    image_width, image_height = pil_image.size

                    img_container = ImageContainer(pil_image)
                    img_container.add_object_bounding_box_details([[x1_orig,y1_orig,x2_orig,y2_orig],class_name,class_id])

                    cropped_image_container = self.crop_image(img_container,1024,600)
                    img = np.asarray(cropped_image_container.image_encoded)
                    
                    self.image_with_bounding_box(samples_dir,cropped_image_container)
                                        
                    print("class id", class_id)
                    
                    elements_list.append([cropped_image_container,image_width,image_height])
                    train_list,test_list = util_obj.create_augmented_images(elements_list,200)
                    self.serialize_image_list(train_writer,train_list)
                    self.serialize_image_list(test_writer,test_list)
        except Exception as e:
                print(e)
                traceback(e)

        
        label_map_file = util_obj.open_file(labelmap_file, "wb")
        label_map_file.write(class_map.SerializeToString())
        label_map_file.close()

        train_writer.close()
        test_writer.close()

    def serialize_image_list(self,tf_writer,images_list):
        
        for image_list in images_list:
            example = util_obj.create_tf_example_object_detection(image_list[0],image_list[1],image_list[2])
            tf_writer.write(example.SerializeToString())

    def image_with_bounding_box(self,dir,cropped_image_container):
        image = np.asarray(cropped_image_container.image_encoded)
        inner_object = cropped_image_container.get_inner_objects()[0]
        bounding_box = inner_object[0]
        x1 = bounding_box[0]
        y1 = bounding_box[1]
        x2 = bounding_box[2]
        y2 = bounding_box[3]
        image = cv.rectangle(image, (x1,y1), (x2,y2),  (0,255,0), 3)
        name = inner_object[1]

        imsave(dir+name+".jpeg",image)

    def crop_image(self,image_container,crop_width,crop_height):
        image = image_container.image_encoded
        width, height = image.size
        inner_object = image_container.get_inner_objects()[0]
        bounding_box = inner_object[0]
        x1 = bounding_box[0]
        y1 = bounding_box[1]
        x2 = bounding_box[2]
        y2 = bounding_box[3]

        class_name = inner_object[1]
        class_id = inner_object[2]

        # width calculation
        mid_x = (x1+x2)/2
        element_width = x2-x1

        if (mid_x)>=crop_width/2:
            crop_x_left = mid_x - (crop_width/2)
            new_x1 = (crop_width/2) - (x2-x1)/2
        else:
            crop_x_left = 0
            new_x1 = x1

        
        if (width - mid_x)>=crop_width/2:
            crop_x_right = mid_x + (crop_width/2)
            
        else:
            crop_x_right = width
        
        new_x2 = new_x1 + element_width

        mid_y = (y1+y2)/2
        element_height = y2-y1
        if (mid_y)>=crop_height/2:
            crop_y_up = mid_y - (crop_height/2)
            new_y1 = (crop_height/2) - (y2-y1)/2
        else:
            crop_y_up = 0
            new_y1 = y1

        
        if (height - mid_y)>=crop_height/2:
            crop_y_down = mid_y + (crop_height/2)
        else:
            crop_y_down = height

        new_y2 = new_y1 + element_height

        left_adjustment = ((crop_width/2) - mid_x) if crop_x_left == 0 else 0;
        right_adjustment = ((crop_width/2) - (width - mid_x)) if crop_x_right == width else 0;    
        up_adjustment = ((crop_height/2) - mid_y) if crop_y_up == 0 else 0;
        down_adjustment = ((crop_height/2) - (height - mid_y)) if crop_y_down == height else 0;

        crop_x_left -= right_adjustment
        crop_x_right += left_adjustment
        crop_y_up -= down_adjustment
        crop_y_down += up_adjustment

        new_x1 += right_adjustment
        new_x2 += right_adjustment
        new_y1 += down_adjustment
        new_y2 += down_adjustment

        if ((crop_x_right - crop_x_left ) > 1024 or (crop_y_down - crop_y_up) > 600):
            print ("incorrect cropping")

        image = image.crop((crop_x_left,crop_y_up,crop_x_right,crop_y_down))
        cropped_image_container = ImageContainer(image)
        cropped_image_container.add_object_bounding_box_details([[int(new_x1),int(new_y1),int(new_x2),int(new_y2)],class_name,class_id])
        return cropped_image_container 

if __name__ == '__main__':
    driver = appcode.AppFacing('pc', True).get_driver()
    web_element_generator = WebElementGenerator(driver)
    web_element_generator.generate_elements("http://awful-valentine.com/purchase-forms/slow-ajax/")
    driver.quit()
