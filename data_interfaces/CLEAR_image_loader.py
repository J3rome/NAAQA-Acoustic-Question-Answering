import os
from PIL import Image
import numpy as np
import torch
import torchaudio
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

    def get_padding(self):
        return self.image_loader.get_padding()

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

    def get_padding(self):
        pass

# TODO : Verify, is this really useful ?
class ErrorImgLoader(AbstractImgLoader):
    def __init__(self, img_path):
        AbstractImgLoader.__init__(self, img_path)

    def get_image(self, **kwargs):
        assert False, "The image/crop is not available in file: {}".format(self.img_path)


class RawImageBuilder(AbstractImgBuilder):
    def __init__(self, img_dir):
        AbstractImgBuilder.__init__(self, img_dir, is_raw=True, require_process=True)

    def build(self, image_id, filename, which_set, **kwargs):
        img_path = os.path.join(self.img_dir, which_set, filename)
        return RawImageLoader(img_path)


class RawImageLoader(AbstractImgLoader):
    def __init__(self, img_path):
        AbstractImgLoader.__init__(self, img_path)

    def get_image(self, return_tensor=True):
        # Our images are saved as RGBA. The A dimension is always 1.
        # We could simply get rid of it instead of converting
        #img = io.imread(self.img_path)[:,:,:3]
        img = Image.open(self.img_path).convert('RGB')

        if return_tensor:
            # PIL images are in format H x W x C
            # Images tensor are in format C x H x W thus the permutation
            return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        else:
            return img

    def get_padding(self):
        return None


class RawAudioBuilder(AbstractImgBuilder):
    def __init__(self, img_dir):
        AbstractImgBuilder.__init__(self, img_dir, is_raw=True, require_process=True)

    def build(self, image_id, filename, which_set, **kwargs):
        img_path = os.path.join(self.img_dir, which_set, filename)
        return RawAudioLoader(img_path)


class RawAudioLoader(AbstractImgLoader):
    def __init__(self, img_path):
        AbstractImgLoader.__init__(self, img_path)

    def get_image(self, normalize=True):
        audio, sample_rate = torchaudio.load(self.img_path, normalization=normalize)
        
        return audio

    def get_padding(self):
        return None



h5_basename="features.h5"
h5_feature_key="features"
h5_idx_key="idx2img"
h5_img_padding_key='img_padding'

class h5FeatureBuilder(AbstractImgBuilder):
    def __init__(self, img_dir, bufferize, is_raw=False):
        AbstractImgBuilder.__init__(self, img_dir, is_raw=is_raw)
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
        h5filepath = os.path.join(self.img_dir, h5filename)

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

    def get_image(self, return_tensor=True):
        img = self.h5file[h5_feature_key][self.id]

        if return_tensor:
            return torch.tensor(img, dtype=torch.float32)
        else:
            return img

    def get_padding(self):
        return torch.tensor(self.h5file[h5_img_padding_key][self.id], dtype=torch.int)

# Load while loading dataset (requires a lot of memory)
class h5FeatureBufloader(AbstractImgLoader):
    def __init__(self, img_path, h5file, id):
        AbstractImgLoader.__init__(self, img_path)
        self.data = h5file[h5_feature_key][id]

    def get_image(self, **kwargs):
        return self.data



def get_img_builder(input_image_type, data_dir, preprocessed_folder_name='preprocessed', bufferize=None):

    input_type = input_image_type

    if input_type == 'audio':
        loader = RawAudioBuilder(os.path.join(data_dir, 'audio'))
    elif input_type in ["conv", "raw_h5"]:

        # NOTE : When testing multiple dataset configurations, Images and questions are generated in separate folder and
        #        linked together so we don't have multiple copies of the dataset (And multiple preprocessing runs)
        #
        #        If we can't find the 'preprocessed_folder_name', we follow the default preprocessed folder symlink.
        #        If the requested 'preprocessed_folder_name' is present in this folder, we create a symlink to it so we
        #        can have access to it.
        #
        #        If "preprocessed" is not a symlink, 'output_folder_name' will be created in requested 'data_path'

        preprocessed_folder_path = "%s/%s" % (data_dir, preprocessed_folder_name)
        preprocessed_exist = os.path.exists(preprocessed_folder_path)
        preprocessed_default_folder_path = '%s/preprocessed' % data_dir
        if not preprocessed_exist:
            if os.path.exists(preprocessed_default_folder_path) and os.path.islink(preprocessed_default_folder_path):

                # Retrieve paths from symlink
                default_link_value = os.readlink(preprocessed_default_folder_path)
                new_link_value = default_link_value.replace('preprocessed', preprocessed_folder_name)

                real_preprocessed_folder_path = "%s/%s" % (data_dir, new_link_value)
                if os.path.isdir(real_preprocessed_folder_path):
                    # Create symlink
                    os.symlink(new_link_value, preprocessed_folder_path)
                else:
                    assert False, "Preprocessed folder '%s' doesn't exist" % preprocessed_folder_path
            else:
                assert False, "Preprocessed folder '%s' doesn't exist" % preprocessed_folder_path

        bufferize = bufferize if bufferize is not None else False
        loader = h5FeatureBuilder(os.path.join(data_dir, preprocessed_folder_name), is_raw='raw' in input_type,
                                  bufferize=bufferize)
    elif input_type == "raw":
        # TODO : Make the 'images' path parametrable
        loader = RawImageBuilder(os.path.join(data_dir, 'images'))
    else:
        assert False, "incorrect image input: {}".format(input_type)

    return loader

