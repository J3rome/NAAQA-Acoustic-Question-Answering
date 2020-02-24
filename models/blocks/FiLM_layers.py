from collections import OrderedDict

import torch.nn as nn

from models.utils import Conv2d_padded, append_spatial_location


class FiLM_layer_separated(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_drop_prob=0.0, save_gammas_betas=True,
                 film_layer_transformation=None):
        super(FiLM_layer_separated, self).__init__()

        # Used for text summary
        self.summary_level = 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.save_gammas_betas = save_gammas_betas
        self.film_layer_transformation = film_layer_transformation

        # TODO : Prealocate gamma and betas ? In Tensor ? Np Array ? Only if save_gammas_betas == True
        self.gammas = None
        self.betas = None

        self.gamma_proj = nn.Linear(self.in_channels, self.out_channels)
        self.beta_proj = nn.Linear(self.in_channels, self.out_channels)

        #self.params_vector = nn.Linear(self.in_channels, 2 * self.out_channels)     # FIXME : Original film have another multiplier : num_modules (which is 4 --> Number of resblock)
                                                                                    # FIXME : The linear layer is not in here on the original tho. This might be why we don't have *4 for the number of resblock (We are inside the resblock here)
        self.dropout = nn.Dropout(p=dropout_drop_prob)

    def forward(self, input_features_rnn_state):
        input_features, rnn_last_hidden_state = input_features_rnn_state

        gammas = self.gamma_proj(rnn_last_hidden_state)
        betas = self.beta_proj(rnn_last_hidden_state)

        if self.save_gammas_betas:
            self.gammas = gammas.detach()
            self.betas = betas.detach()

        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(input_features)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(input_features)

        if self.film_layer_transformation == 'plus':
            # FIXME : Original implementation have a shift of 1 (Just after film_gen.decoder forward)
            output = (1 + gammas) * input_features + betas        # FIXME : Tensorflow implementation do 1 + gammas
        else:
            output = gammas * input_features + betas

        return output


class FiLM_layer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_drop_prob=0.0, save_gammas_betas=True,
                 film_layer_transformation=None):
        super(FiLM_layer, self).__init__()

        # Used for text summary
        self.summary_level = 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.save_gammas_betas = save_gammas_betas
        self.film_layer_transformation = film_layer_transformation

        # TODO : Prealocate gamma and betas ? In Tensor ? Np Array ? Only if save_gammas_betas == True
        self.gammas = None
        self.betas = None

        self.params_vector = nn.Linear(self.in_channels, 2 * self.out_channels)     # FIXME : Original film have another multiplier : num_modules (which is 4 --> Number of resblock)
                                                                                    # FIXME : The linear layer is not in here on the original tho. This might be why we don't have *4 for the number of resblock (We are inside the resblock here)
        self.dropout = nn.Dropout(p=dropout_drop_prob)

    def forward(self, input_features_rnn_state):
        input_features, rnn_last_hidden_state = input_features_rnn_state

        film_params = self.params_vector(rnn_last_hidden_state)

        # FIXME : Is it a good idea to have dropout here ?
        #film_params = self.dropout(film_params)

        gammas, betas = film_params.split(self.out_channels, dim=-1)

        if self.save_gammas_betas:
            self.gammas = gammas.detach()
            self.betas = betas.detach()

        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(input_features)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(input_features)

        if self.film_layer_transformation == 'plus':
            # FIXME : Original implementation have a shift of 1 (Just after film_gen.decoder forward)
            output = (1 + gammas) * input_features + betas        # FIXME : Tensorflow implementation do 1 + gammas
        else:
            output = gammas * input_features + betas

        return output


class FiLMed_resblock(nn.Module):
    def __init__(self, in_channels, out_channels, context_size, dropout_drop_prob=0.0, kernel1=(1, 1), kernel2=(3, 3),
                 film_layer_transformation=None):

        super(FiLMed_resblock, self).__init__()

        # Used for text summary
        self.summary_level = 1

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel1, stride=1,
                               padding='SAME', dilation=1)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        #self.dropout = nn.Dropout(p=dropout_drop_prob)

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel2, stride=1,
                               padding='SAME', dilation=1)),
            # Center/reduce output (Batch Normalization with no training parameters)
            ('batchnorm', nn.BatchNorm2d(out_channels, affine=False, eps=0.001))
        ]))

        # Film Layer
        self.film_layer = nn.Sequential(OrderedDict([
            ('film', FiLM_layer(context_size, out_channels, film_layer_transformation=film_layer_transformation)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, input_features, rnn_state, spatial_location):
        if spatial_location:
            input_features = append_spatial_location(input_features)

        conv1_out = self.conv1(input_features)
        #conv1_out = self.dropout(conv1_out)
        out = self.conv2(conv1_out)
        out = self.film_layer((out, rnn_state))
        #out = self.dropout(out)

        return out + conv1_out