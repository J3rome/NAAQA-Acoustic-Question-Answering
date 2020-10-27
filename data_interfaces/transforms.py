import torch
import torchvision.transforms.functional as vis_F
import torch.nn.functional as F

import torchaudio
import torchaudio.functional as audio_F

import matplotlib.pyplot as plt


class ResampleAudio(object):

    def __init__(self, original_sample_rate, resample_to):
        self.resample_transform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=resample_to)
        self.resample_to = resample_to

    def __call__(self, sample):
        sample['image'] = self.resample_transform(sample['image'])

        return sample


class GenerateSpectrogram(object):

    def __init__(self, n_fft, hop_length=None, keep_freq_point=None, db_amplitude=True, per_spectrogram_normalize=False, device='cpu'):
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length,
                                                                       normalized=True, wkwargs={'device': device})  # Device for the hanning window function
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB() if db_amplitude else None

        self.n_fft = n_fft
        self.hop_length = self.spectrogram_transform.hop_length
        self.keep_freq_point = keep_freq_point
        self.per_spectrogram_normalize = per_spectrogram_normalize

    def __call__(self, sample):
        specgram = self.spectrogram_transform(sample['image'])[0, :, :]

        if self.keep_freq_point:
            specgram = specgram[:self.keep_freq_point, :]

        if self.amplitude_to_db:
            specgram = self.amplitude_to_db(specgram)

        if self.per_spectrogram_normalize:
            specgram_min = specgram.min()
            specgram = (specgram - specgram_min) / (specgram.max() - specgram_min)

        sample['image'] = torch.flip(specgram, (0,)).unsqueeze(0)

        return sample


class GenerateMelSpectrogram(object):

    def __init__(self, n_fft, n_mels, sample_rate, hop_length=None, keep_freq_point=None,
                 per_spectrogram_normalize=False, device='cpu'):
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length,
                                                                       normalized=True, wkwargs={'device': device})  # Device for the hanning window function
        self.mel_scale = MelScale(sample_rate=sample_rate, n_mels=n_mels)

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = self.spectrogram_transform.hop_length
        self.keep_freq_point = keep_freq_point
        if keep_freq_point:
            print(f"[WARNING] - Be careful, we are using only the {keep_freq_point} fft point before going on the mel scale")
        self.per_spectrogram_normalize = per_spectrogram_normalize

    def __call__(self, sample):
        specgram = self.spectrogram_transform(sample['image'])[0, :, :]
        if self.keep_freq_point:
            specgram = specgram[:self.keep_freq_point, :]

        specgram = self.mel_scale(specgram)

        if self.per_spectrogram_normalize:
            specgram_min = specgram.min()
            specgram = (specgram - specgram_min) / (specgram.max() - specgram_min)

        sample['image'] = torch.flip(specgram, (0,)).unsqueeze(0)

        return sample


# Taken from torchaudio 1.5.0 to fix a bug when the input is on GPU
class MelScale(torch.nn.Module):
    r"""Turn a normal STFT into a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    User can control which device the filter bank (`fb`) is (e.g. fb.to(spec_f.device)).

    Args:
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. Calculated from first input
            if None is given.  See ``n_fft`` in :class:`Spectrogram`. (Default: ``None``)
    """
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self,
                 n_mels = 128,
                 sample_rate = 16000,
                 f_min = 0.,
                 f_max = None,
                 n_stft = None):
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min

        assert f_min <= self.f_max, 'Require f_min: %f < f_max: %f' % (f_min, self.f_max)

        fb = torch.empty(0) if n_stft is None else audio_F.create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate)
        self.register_buffer('fb', fb)

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        # Send internal variable on device if needed
        if self.fb.device != specgram.device:
            print("To Device")
            self.fb = self.fb.to(specgram.device)

        # pack batch
        shape = specgram.size()
        specgram = specgram.view(-1, shape[-2], shape[-1])

        if self.fb.numel() == 0:
            tmp_fb = audio_F.create_fb_matrix(specgram.size(1), self.f_min, self.f_max, self.n_mels, self.sample_rate)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(specgram.transpose(1, 2), self.fb).transpose(1, 2)

        # unpack batch
        mel_specgram = mel_specgram.view(shape[:-2] + mel_specgram.shape[-2:])

        return mel_specgram


class ApplyColormapToSpectrogram(object):
    """
    Will use matplotlib colormap to convert 1 Channel image (Spectrogram) to 3 Channels RGB images (Colored spectrogram)
    This transform assume the format C x H x W
    """

    def __init__(self, cmap="Blues"):
        self.cmap = plt.get_cmap(cmap)

    def __call__(self, sample):
        img = sample['image'].squeeze(0)

        original_device = img.device

        if 'cuda' in original_device.type:
            img = img.cpu()

        sample['image'] = torch.tensor(self.cmap(img, alpha=None)[:, :, :3], device=original_device)

        return sample


class ImgBetweenZeroOne(object):
    """ Normalize the image between 0 and 1 """

    def __init__(self, max_val=255, min_val=0):
        self.max_val = max_val
        self.min_val = min_val

    def __call__(self, sample):
        sample['image'] = (sample['image'] - self.min_val) / (self.max_val - self.min_val)

        return sample


class ResizeTensor(object):
    """ Resize PIL image to 'output_shape' """
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def __call__(self, sample):
        sample['image'] = F.interpolate(sample['image'].unsqueeze(0), size=self.output_shape, mode='bilinear',
                                        align_corners=False).squeeze(0)
        return sample


class ResizeTensorBasedOnHeight(object):
    """ Resize PIL image to 'output_height' while keeping image ration """
    def __init__(self, output_height):
        self.output_height = output_height

    def get_resized_dim(self, height, width):
        new_height = self.get_output_height()
        new_width = self.get_output_width(height, width)

        return new_height, new_width

    def get_output_height(self):
        return self.output_height

    def get_output_width(self, height, width):
        return int(self.output_height * width / height + 0.5)

    def __call__(self, sample):
        # Images tensor are in format C x H x W
        output_width = self.get_output_width(sample['image'].shape[1], sample['image'].shape[2])

        if output_width + self.output_height != sample['image'].shape[2] + sample['image'].shape[1]:
            sample['image'] = F.interpolate(sample['image'].unsqueeze(0), size=(self.output_height, output_width),
                                            mode='bilinear', align_corners=False).squeeze(0)

        return sample


class ResizeTensorBasedOnWidth(object):
    """
    Resize PIL image to 'output_width' while keeping image ration
    If max_width is provided,
    the resizing will be made according to the max_width ratio instead of the sample width ratio
    """
    def __init__(self, output_width, max_width=None):
        self.output_width = output_width
        self.max_width = max_width

    def get_resized_dim(self, height, width):
        if self.max_width:
            reference_width = self.max_width
        else:
            reference_width = width

        new_height = self.get_output_height(height, reference_width)
        new_width = self.get_output_width(new_height, height, width)

        return int(new_height + 0.5), int(new_width + 0.5)

    def get_output_height(self, height, width):
        return self.output_width * height / width

    def get_output_width(self, target_height, height, width):
        return target_height * width / height

    def __call__(self, sample):
        output_height, output_width = self.get_resized_dim(sample['image'].shape[1], sample['image'].shape[2])


        if output_height + output_width != sample['image'].shape[1] + sample['image'].shape[2]:
            sample['image'] = F.interpolate(sample['image'].unsqueeze(0), size=(output_height, output_width),
                                            mode='bilinear', align_corners=False).squeeze(0)

        return sample


class ResizeTensorBasedOnMaxWidth(object):
    """
    Resize Tensor to 'output_width' while keeping time axis ratio
    """
    def __init__(self, output_width, max_width=None, output_height=None):
        self.output_height = output_height
        self.output_width = output_width
        self.max_width = max_width

        if self.max_width:
            self.width_ratio = self.output_width / self.max_width

    def get_resized_dim(self, height, width):
        output_height = self.output_height if self.output_height else height
        output_width = self.output_width if self.max_width is None else int(width * self.width_ratio + 0.5)

        return output_height, output_width

    def __call__(self, sample):
        output_height, output_width = self.get_resized_dim(sample['image'].shape[1], sample['image'].shape[2])

        if output_height + output_width != sample['image'].shape[1] + sample['image'].shape[2]:
            sample['image'] = F.interpolate(sample['image'].unsqueeze(0), size=(output_height, output_width),
                                            mode='bilinear', align_corners=False).squeeze(0)

        return sample


class ResizeImg(object):
    """ Resize PIL image to 'output_shape' """
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def __call__(self, sample):
        sample['image'] = vis_F.resize(sample['image'], self.output_shape)
        return sample


class ResizeImgBasedOnHeight(object):
    """ Resize PIL image to 'output_height' while keeping image ration """
    def __init__(self, output_height):
        self.output_height = output_height

    def __call__(self, sample):
        output_width = int(self.output_height * sample['image'].width / sample['image'].height)

        if output_width + self.output_height != sample['image'].width + sample['image'].height:
            sample['image'] = vis_F.resize(sample['image'], (self.output_height, output_width))

        return sample


class ResizeImgBasedOnWidth(object):
    """ Resize PIL image to 'output_width' while keeping image ration"""
    def __init__(self, output_width):
        self.output_width = output_width

    def __call__(self, sample):
        output_height = int(self.output_width * sample['image'].height / sample['image'].width)

        if output_height + self.output_width != sample['image'].height + sample['image'].width:
            sample['image'] = vis_F.resize(sample['image'], (output_height, self.output_width))

        return sample


class PadTensorHeight(object):
    """ Pad image Tensor to output_height -- Pad at bottom of image"""
    def __init__(self, output_height):
        self.output_height = output_height

    def __call__(self, sample):
        height_to_pad = self.output_height - sample['image'].shape[1]

        if height_to_pad > 0:
            sample['image'] = F.pad(sample['image'], [0, 0, 0, height_to_pad])

        if 'image_padding' in sample:
            sample['image_padding'][0] += height_to_pad
        else:
            sample['image_padding'] = torch.tensor([height_to_pad, 0], dtype=torch.int)

        return sample


class PadTensor(object):
    """ Pad image Tensor to output_height x output_width. Pad to right & bottom of image"""
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def __call__(self, sample):
        width_to_pad = max(self.output_shape[1] - sample['image'].shape[2], 0)
        height_to_pad = max(self.output_shape[0] - sample['image'].shape[1], 0)

        if width_to_pad + height_to_pad > 0:
            sample['image'] = F.pad(sample['image'], [0, width_to_pad, 0, height_to_pad])

        if 'image_padding' in sample:
            sample['image_padding'][0] += height_to_pad
            sample['image_padding'][1] += width_to_pad
        else:
            sample['image_padding'] = torch.tensor([height_to_pad, width_to_pad], dtype=torch.int)

        return sample


class RemovePadding(object):
    """
    When stored in H5 files, images all have the same size since they are padded to the biggest dimensions.
    If we don't want to use a different/no padding for these images, we first need to remove the padding from the images
    """
    def __call__(self, sample):
        if 'image_padding' in sample and sum(sample['image_padding']) > 0:
            sample['image'] = sample['image'][:, sample['image_padding'][0]:, sample['image_padding'][1]:]
            sample['image_padding'] = torch.tensor([0, 0], dtype=torch.int)

        return sample


class NormalizeSample(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace      # FIXME : Not sure about inplace

    def __call__(self, sample):
        """
        Args:
            sample (Dict): Dict containing a Tensor image of size (C, H, W) to be normalized (Key is 'image').

        Returns:
            Tensor: Normalized Tensor image.
        """

        sample['image'] = vis_F.normalize(sample['image'], self.mean, self.std, self.inplace)

        return sample


class NormalizeInverse(NormalizeSample):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, sample):
        return super().__call__(sample)#.clone())