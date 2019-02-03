class ImageContainer:
    
    def __init__(self,image_encoded):
        self.image_encoded = image_encoded
        self.inner_objects_list = []


    def add_object_bounding_box_details(self,list_inner_object):
        self.inner_objects_list.append(list_inner_object)


    def get_inner_objects(self):
        return self.inner_objects_list

