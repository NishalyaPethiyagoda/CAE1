import numpy as np
from keras.utils import Sequence

def create_strided_patches(img, patch_size, stride):
    patches = []
    img_h, img_w = img.shape[:2]
    patch_height = patch_size
    patch_width = patch_size

    for y in range(0, img_h - patch_height + 1, stride):
        for x in range(0, img_w - patch_width + 1, stride):
            patch = img[y:y + patch_height, x:x + patch_width]
            patch = patch.astype('float32') / 255.0
            patches.append(patch)
    
    return patches

class DataGenerator(Sequence):
    """Generates data for Keras"""
    def __init__(self, imgs, patch_size, stride):
        self.imgs = imgs
        self.patch_size = patch_size
        self.stride = stride

        # Generate all patches from all images
        self.all_patches = self._generate_all_patches()

    def _generate_all_patches(self):
        all_patches = []
        for img in self.imgs:
            patches = create_strided_patches(img, self.patch_size, self.stride)
            all_patches.extend(patches)
        return np.array(all_patches)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.all_patches)))

    def __getitem__(self, index):
        """Generate one batch of data"""
        data = self.all_patches
        return data, data

# import numpy as np

# from keras.utils import Sequence

# def create_strided_patches(img, patch_size, stride):
    
#     patches = []
#     img_height, img_width = img.shape[ : 2]
#     patch_height, patch_width = patch_size

#     for y in range(0, img_height - patch_height + 1, stride):
#         for x in range(0, img_width - patch_width + 1, stride):
#             patch = img[y: y+patch_height, x: x+patch_width]
#             patches.append(patch)

#     return patches

# def prepare_patch(img, patch_size, stride):
#     patch = create_strided_patches(img, patch_size, stride)

#     return patch.astype('float32') / 255.0

# class DataGenerator(Sequence):
#     """Generates data for Keras"""
#     # def __init__(self, imgs, patch_size, center_size, batch_size=32, shuffle=True):
#     def __init__(self, imgs, patch_size, stride, batch_size=32, shuffle=True):
#         self.imgs = imgs
#         self.patch_size = patch_size
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.stride = stride
        
#         # self.epoch = 0        
#         # self.on_epoch_end()

#     def __len__(self):
#         """Denotes the number of batches per epoch"""
#         return int(np.floor(len(self.imgs) / self.batch_size))

#     def __getitem__(self, index):
#         """Generate one batch of data"""
#         # Generate indexes of the batch
#         indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
#         # Generate data
#         x = self.__data_generation(indexes, [self.imgs[k] for k in indexes])
#         return x, x
            
#     def __data_generation(self, indexes, batch):      
#         """Augments or/and pretransforms data"""
#         patches = [ prepare_patch(img, self.patch_size, self.stride) for img in batch]

#         # patches = []

#         # for img in batch:
#         #     img_patches = create_strided_patches(img, self.patch_size, self.stride)
#         #     for patch in img_patches:
#         #         patch = prepare_patch(patch, self.patch_size, self.mask, (self.epoch * len(patches)) % (2**32 - 1))
#         #         patches.append(patch)

#         return np.array(patches)
    
# class TestDataGenerator(Sequence):
#     """Generates data for Keras"""
#     def __init__(self, patch_size, center_size, stride):
#         self.patch_size = patch_size
#         self.stride = stride 

#     def generate_patches(self, img):
#         """Generate patches from a single image"""
#         patches = [prepare_patch(img, self.patch_size, self.stride)]
#         return patches


