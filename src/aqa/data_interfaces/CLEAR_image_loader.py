import os
from PIL import Image
import numpy as np
import h5py

# Why doing an image builder/loader?
# Well, there are two reasons:
#  - first you want to abstract the kind of image you are using (raw/conv/feature) when you are loading the dataset/batch.
#  One may just want... to load an image!
#  - One must optimize when to load the image for multiprocessing.
#       You do not want to serialize a 2Go of fc8 features when you create a process
#       You do not want to load 50Go of images at start
#
#   The Builder enables to abstract the kind of image you want to load. It will be used while loading the dataset.
#   The Loader enables to load/process the image when you need it. It will be used when you create the batch
#
#   Enjoy design patterns, it may **this** page of code complex but the it makes the whole project easier! Act Local, Think Global :P
#

def resize_image(img, width, height):
    return img.resize((width, height), resample=Image.BILINEAR)

# TODO : Rename to Spectrogram ?
class CLEARImage:
    def __init__(self, id, filename, image_builder, which_set):
        self.id = id
        self.filename = filename

        self.image_loader = image_builder.build(id, filename=filename, which_set=which_set)

    def get_image(self, **kwargs):
        return self.image_loader.get_image(**kwargs)

class AbstractImgBuilder(object):
    def __init__(self, img_dir, is_raw, require_process=False):
        self.img_dir = img_dir
        self.is_raw = is_raw
        self.require_process = require_process

    def build(self, image_id, filename, which_set, **kwargs):
        return self

    def is_raw_image(self):
        return self.is_raw

    def require_multiprocess(self):
        return self.require_process

class AbstractImgLoader(object):
    def __init__(self, img_path):
        self.img_path = img_path

    def get_image(self, **kwargs):
        pass

# TODO : Verify, is this really useful ?
class ErrorImgLoader(AbstractImgLoader):
    def __init__(self, img_path):
        AbstractImgLoader.__init__(self, img_path)

    def get_image(self, **kwargs):
        assert False, "The image/crop is not available in file: {}".format(self.img_path)


class RawImageBuilder(AbstractImgBuilder):
    def __init__(self, img_dir, width, height, per_channel_mean_to_substract=None):
        AbstractImgBuilder.__init__(self, img_dir, is_raw=True, require_process=True)
        self.width = width
        self.height = height
        self.per_channel_mean_to_substract = per_channel_mean_to_substract

    def build(self, image_id, filename, which_set, **kwargs):
        img_path = os.path.join(self.img_dir, which_set, filename)
        return RawImageLoader(img_path, self.width, self.height, per_channel_mean_to_substract=self.per_channel_mean_to_substract)

class RawImageLoader(AbstractImgLoader):
    def __init__(self, img_path, width, height, per_channel_mean_to_substract):
        AbstractImgLoader.__init__(self, img_path)
        self.width = width
        self.height = height
        self.per_channel_mean_to_substract = per_channel_mean_to_substract


    def get_image(self, **kwargs):
        img = Image.open(self.img_path).convert('RGB')

        img = resize_image(img, self.width , self.height)
        img = np.array(img, dtype=np.float32)

        if self.per_channel_mean_to_substract is not None:
            img -= self.per_channel_mean_to_substract[None, None, :]

        return img


h5_basename="features.h5"
h5_feature_key="features"
h5_idx_key="idx2img"

class h5FeatureBuilder(AbstractImgBuilder):
    def __init__(self, img_dir, bufferize):
        AbstractImgBuilder.__init__(self, img_dir, is_raw=False)
        self.bufferize = bufferize
        self.h5files = dict()
        self.img2idx = dict()

    def build(self, image_id, filename, optional=True, which_set=None,**kwargs):

        # Is the h5 features split into train/val/etc. files or gather into a single file
        if which_set is not None:
            h5filename = which_set + "_" + h5_basename
        else:
            h5filename = h5_basename

        # Build full path
        h5filepath = os.path.join(self.img_dir,h5filename)

        # Retrieve
        if h5filename not in self.h5files:
            # Load file pointer to h5
            h5file = h5py.File(h5filepath, 'r')

            # hd5 requires continuous id while image_id can be very diverse.
            # We then need a mapping between both of them
            if h5_idx_key in h5file:
                # Retrieve id mapping from file
                img2idx = {id_img : id_h5 for id_h5, id_img in enumerate(h5file[h5_idx_key])}
            else:
                # Assume their is a perfect identity between image_id and h5_id
                no_images = h5file[h5_feature_key].shape[0]
                img2idx = {k : k for k in range(no_images) }

            self.h5files[h5filename] = h5file
            self.img2idx[h5filename] = img2idx
        else:
            h5file = self.h5files[h5filename]
            img2idx = self.img2idx[h5filename]

        if self.bufferize:
            if (optional and image_id in img2idx) or (not optional):
                return h5FeatureBufloader(h5filepath, h5file=h5file, id=img2idx[image_id])
            else:
                return ErrorImgLoader(h5filepath)
        else:
            return h5FeatureLoader(h5filepath, h5file=h5file, id=img2idx[image_id])

# Load while creating batch
class h5FeatureLoader(AbstractImgLoader):
    def __init__(self, img_path, h5file, id):
        AbstractImgLoader.__init__(self, img_path)
        self.h5file = h5file
        self.id = id

    def get_image(self, **kwargs):
        return self.h5file[h5_feature_key][self.id]

# Load while loading dataset (requires a lot of memory)
class h5FeatureBufloader(AbstractImgLoader):
    def __init__(self, img_path, h5file, id):
        AbstractImgLoader.__init__(self, img_path)
        self.data = h5file[h5_feature_key][id]

    def get_image(self, **kwargs):
        return self.data



def get_img_builder(config, data_dir, bufferize=None):

    input_type = config["type"]

    # FIXME: Figure out why there is the fc8 and fc7 cases with inversed buffersize logic
    if input_type in ["fc8", "fc7"]:
        bufferize = bufferize if bufferize is not None else True
        loader = h5FeatureBuilder(os.path.join(data_dir, 'preprocessed'), bufferize=bufferize)
    elif input_type in ["conv", "raw_h5"]:
        bufferize = bufferize if bufferize is not None else False
        loader = h5FeatureBuilder(os.path.join(data_dir, 'preprocessed'), bufferize=bufferize)
    elif input_type == "raw":
        loader = RawImageBuilder(os.path.join(data_dir, 'images'),
                                height=config["dim"][0],
                                width=config["dim"][1],
                                per_channel_mean_to_substract=config.get("per_channel_mean_to_substract", None))
    else:
        assert False, "incorrect image input: {}".format(input_type)

    return loader

