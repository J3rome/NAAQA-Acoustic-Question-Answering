from collections import OrderedDict

import torch
import torch.nn as nn

from models.feature_extractor import Resnet_feature_extractor
from utils.random import set_random_state, get_random_state
from utils.model import Conv2d_tf, append_spatial_location


class FiLM_layer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_drop_prob=0.0, save_gammas_betas=True,
                 film_layer_transformation=None):
        super(FiLM_layer, self).__init__()

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

        self.is_container = True

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_tf(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel1, stride=1,
                               padding='SAME', dilation=1)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        #self.dropout = nn.Dropout(p=dropout_drop_prob)

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_tf(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel2, stride=1,
                               padding='SAME', dilation=1)),
            # Center/reduce output (Batch Normalization with no training parameters)
            ('batchnorm', nn.BatchNorm2d(out_channels, affine=False))
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


class CLEAR_FiLM_model(nn.Module):
    def __init__(self, config, input_image_channels, nb_words, nb_answers, feature_extraction_config=None,    # FIXME : The index should probably be in the config
                 sequence_padding_idx=0, save_features=True):
        super(CLEAR_FiLM_model, self).__init__()

        self.current_device = 'cpu'
        self.config = config

        self.early_stopping = config.get('early_stopping', None)
        self.early_stopping = self.early_stopping if self.early_stopping and self.early_stopping['enable'] else None

        dropout_drop_prob = float(config['optimizer'].get('dropout_drop_prob', 0.0))

        film_layer_transformation = config['resblock'].get('film_projection_type', None)
        #self.dropout = nn.Dropout(p=dropout_drop_prob)

        # Question Pipeline
        self.question_pipeline = Question_pipeline(config, nb_words, dropout_drop_prob, sequence_padding_idx)

        # Image Pipeline
        self.image_pipeline = Image_pipeline(config, input_image_channels, feature_extraction_config)

        # Question and Image Fusion
        resblock_out_channels = config['stem']['conv_out']
        resblock_in_channels = resblock_out_channels + ( 2 if config['resblock']['spatial_location'] else 0 )
        self.resblocks = nn.ModuleList()
        self.nb_resblock = config['resblock']['no_resblock']
        for i in range(self.nb_resblock):
            self.resblocks.append(FiLMed_resblock(in_channels=resblock_in_channels,
                                                  out_channels=resblock_out_channels,
                                                  context_size=config["question"]["rnn_state_size"],
                                                  kernel1=config['resblock']['kernel1'],
                                                  kernel2=config['resblock']['kernel2'],
                                                  dropout_drop_prob=dropout_drop_prob,
                                                  film_layer_transformation=film_layer_transformation))

        if config['classifier'].get('type', '').lower() == 'conv':
            # Classification (Via 1x1 conv & GlobalPooling)
            classifier_class = Conv_classifier
        else:
            # Fully connected classifier
            classifier_class = Fcn_classifier

        self.classifier = classifier_class(in_channels=resblock_out_channels,
                                           projection_size=config["classifier"]['projection_size'],
                                           output_size=nb_answers,
                                           spatial_location_layer=config['classifier']['spatial_location'],
                                           dropout_drop_prob=dropout_drop_prob)

    def forward(self, question, question_lengths=None, input_image=None, pack_sequence=False):
        if question is not None and question_lengths is None and input_image is None:
            # Parse first parameter as a list of parameter
            assert type(question) == list and len(question) == 3, "Invalid parameters"
            # Allow passing a list of arguments
            question_lengths = question[1]
            input_image = question[2]
            question = question[0]

        # Question Pipeline
        rnn_hidden_state = self.question_pipeline(question, question_lengths, pack_sequence)

        # Image Pipeline
        conv_out = self.image_pipeline(input_image)

        # Question and Image fusion
        for i, resblock in enumerate(self.resblocks):
            conv_out = resblock(conv_out, rnn_hidden_state,
                                spatial_location=self.config['resblock']['spatial_location'])

        # Classification
        logits, logits_softmaxed = self.classifier(conv_out)

        return logits, logits_softmaxed

    def get_gammas_betas(self):
        gammas = []
        betas = []
        for resblock in self.resblocks:
            betas.append(resblock.film_layer.film.betas)
            gammas.append(resblock.film_layer.film.gammas)

        return gammas, betas

    def get_cleaned_state_dict(self):
        state_dict = self.state_dict()

        if self.image_pipeline.feature_extractor is not None:
            state_dict = {k : p for k,p in state_dict.items() if 'feature_extractor' not in k}

        return state_dict

    def train(self, mode=True):
        # Only call train if model is in eval mode
        if self.training != mode:
            super(CLEAR_FiLM_model, self).train(mode)

        # Keep the feature extractor in eval mode
        if self.image_pipeline.feature_extractor and self.image_pipeline.feature_extractor.training:
            self.feature_extractor.eval()

    def to(self, device=None, dtype=None, non_blocking=False):
        # Copy model to device only once
        if self.current_device != device:
            self.current_device = device
            super(CLEAR_FiLM_model, self).to(device, dtype, non_blocking)


class Conv_classifier(nn.Module):
    def __init__(self, in_channels, projection_size, output_size, spatial_location_layer, dropout_drop_prob):
        super(Conv_classifier, self).__init__()

        self.spatial_location_layer = spatial_location_layer

        # Classification (Via 1x1 conv & GlobalPooling)
        self.classif_prof_conv = nn.Sequential(OrderedDict([
            ('conv', Conv2d_tf(in_channels=in_channels + (2 if spatial_location_layer else 0),
                               out_channels=projection_size,
                               kernel_size=[1, 1], stride=1, padding="VALID", dilation=1)),
            ('batchnorm', nn.BatchNorm2d(projection_size)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.logits = nn.Sequential(OrderedDict([
            ('conv', Conv2d_tf(in_channels=projection_size,
                               out_channels=output_size,
                               kernel_size=[1, 1], stride=1, padding="VALID", dilation=1)),
            ('batchnorm', nn.BatchNorm2d(output_size)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=dropout_drop_prob)

    def forward(self, input_features):
        if self.spatial_location_layer:
            input_features = append_spatial_location(input_features)

        classif_proj_out = self.classif_prof_conv(input_features)

        # FIXME : Should we really use dropout here ?
        #classif_proj_out = self.dropout(classif_proj_out)
        logits_maps = self.logits(classif_proj_out)

        # Global Max Pooling
        # FIXME : Max vs mean ? Concat/Avg both together ?
        logits, _ = logits_maps.view(logits_maps.shape[0], logits_maps.shape[1], -1).max(dim=2)
        #logits = logits_maps.mean(dim=[2, 3])

        logits_softmaxed = self.softmax(logits)

        return logits, logits_softmaxed


class Fcn_classifier(nn.Module):
    def __init__(self, in_channels, projection_size, output_size, spatial_location_layer, dropout_drop_prob):
        super(Fcn_classifier, self).__init__()

        self.spatial_location_layer = spatial_location_layer

        first_conv_out = 32

        self.classif_conv = nn.Sequential(OrderedDict([
            ('conv', Conv2d_tf(in_channels=in_channels + (2 if spatial_location_layer else 0),
                               out_channels=first_conv_out,
                               kernel_size=[1, 1], stride=1, padding="VALID", dilation=1)),
            ('batchnorm', nn.BatchNorm2d(first_conv_out)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.hidden_layer = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(first_conv_out, projection_size)),
            ('batchnorm', nn.BatchNorm1d(projection_size)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        # FIXME : Why we don't have batchnorm & relu here ?
        self.logits = nn.Linear(projection_size, output_size)

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=dropout_drop_prob)

    def forward(self, input_features):
        if self.spatial_location_layer:
            input_features = append_spatial_location(input_features)

        conv_out = self.classif_conv(input_features)

        # FIXME : Max vs mean ?
        # Global Max Pooling
        logits, _ = conv_out.view(conv_out.shape[0], conv_out.shape[1], -1).max(dim=2)
        #logits, _ = conv_out.max(dim=2)
        #logits, _ = logits.max(dim=2)
        #logits = conv_out.mean(dim=[2, 3])

        hidden_out = self.hidden_layer(logits)
        #hidden_out = self.dropout(hidden_out)

        logits = self.logits(hidden_out)

        logits_softmaxed = self.softmax(logits)

        return logits, logits_softmaxed


class Question_pipeline(nn.Module):
    def __init__(self, config, nb_words, dropout_drop_prob=0, sequence_padding_idx=0):
        super(Question_pipeline, self).__init__()

        # Question Pipeline
        self.word_emb = nn.Embedding(num_embeddings=nb_words,
                                     embedding_dim=config['question']['word_embedding_dim'],
                                     padding_idx=sequence_padding_idx)

        # FIXME : Dropout always set to zero ?
        # FIXME: Are we using correct rnn_state ?
        # FIXME : Bidirectional
        # FIXME : Are we missing normalization here ?
        # TODO : Make sure we have the correct activation fct (Validate that default is tanh)
        self.rnn_state = nn.GRU(input_size=config['question']['word_embedding_dim'],
                                hidden_size=config["question"]["rnn_state_size"],
                                batch_first=True,
                                dropout=0)

        self.dropout = nn.Dropout(p=dropout_drop_prob)

    def forward(self, question, question_lengths, pack_sequence):
        word_emb = self.word_emb(question)
        word_emb = self.dropout(word_emb)

        if pack_sequence:
            word_emb = torch.nn.utils.rnn.pack_padded_sequence(word_emb, question_lengths, batch_first=True,
                                                               enforce_sorted=False)

        rnn_out, rnn_hidden = self.rnn_state(word_emb)

        rnn_hidden_state = self.dropout(rnn_hidden.squeeze(0))

        return rnn_hidden_state


class Image_pipeline(nn.Module):
    def __init__(self, config, input_image_channels, feature_extraction_config, dropout_drop_prob=0):
        super(Image_pipeline, self).__init__()

        self.config = config

        spatial_location_extra_channels = 2 if config['stem']['spatial_location'] else 0

        if feature_extraction_config is not None:
            # Instantiating the feature extractor affects the random state (Raw Vs Pre-Extracted Features).
            # We restore it to ensure reproducibility between input type
            random_state = get_random_state()
            self.feature_extractor = Resnet_feature_extractor(resnet_version=feature_extraction_config['version'],
                                                              layer_index=feature_extraction_config['layer_index'])
            input_image_channels = self.feature_extractor.get_out_channels()
            set_random_state(random_state)
        else:
            self.feature_extractor = None

        ## Stem
        self.stem_conv = nn.Sequential(OrderedDict([
            ('conv', Conv2d_tf(in_channels=input_image_channels + spatial_location_extra_channels,
                               out_channels=config['stem']['conv_out'], kernel_size=config['stem']['conv_kernel'],
                               stride=1, padding='SAME', dilation=1)),
            ('batchnorm', nn.BatchNorm2d(config['stem']['conv_out'])),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.stem_conv2 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_tf(in_channels=config['stem']['conv_out'] + spatial_location_extra_channels,
                               out_channels=config['stem']['conv_out'], kernel_size=config['stem']['conv_kernel'],
                               stride=1, padding='SAME', dilation=1)),
            ('batchnorm', nn.BatchNorm2d(config['stem']['conv_out'])),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.stem_conv3 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_tf(in_channels=config['stem']['conv_out'] + spatial_location_extra_channels,
                               out_channels=config['stem']['conv_out'], kernel_size=config['stem']['conv_kernel'],
                               stride=1, padding='SAME', dilation=1)),
            ('batchnorm', nn.BatchNorm2d(config['stem']['conv_out'])),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.stem_conv4 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_tf(in_channels=config['stem']['conv_out'] + spatial_location_extra_channels,
                               out_channels=config['stem']['conv_out'], kernel_size=config['stem']['conv_kernel'],
                               stride=1, padding='SAME', dilation=1)),
            ('batchnorm', nn.BatchNorm2d(config['stem']['conv_out'])),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.stem_conv5 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_tf(in_channels=config['stem']['conv_out'] + spatial_location_extra_channels,
                               out_channels=config['stem']['conv_out'], kernel_size=config['stem']['conv_kernel'],
                               stride=1, padding='SAME', dilation=1)),
            ('batchnorm', nn.BatchNorm2d(config['stem']['conv_out'])),
            ('relu', nn.ReLU(inplace=True))
        ]))

        # Pooling
        # TODO : Combine with avg pooling ?
        self.max_pool_in_freq = nn.MaxPool2d((3, 1))
        self.max_pool_in_time = nn.MaxPool2d((1, 3))
        self.max_pool_square = nn.MaxPool2d((2, 2))
        # self.max_pool_square = nn.MaxPool2d((3, 3))

        self.dropout = nn.Dropout(p=dropout_drop_prob)

    def forward(self, input_image):
        conv_out = input_image
        if self.feature_extractor:
            # Extract features using pretrained network
            conv_out = self.feature_extractor(conv_out)

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        # FIXME : Move the stem logic inside our own feature extractor
        # FIXME : Probably need to be resnet style, otherwise we'll get vanishing gradients
        # FIXME : We should keep a stem layer after our feature extractor (I guess ?)

        conv_out = self.stem_conv(conv_out)
        conv_out = self.max_pool_in_freq(conv_out)

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        conv_out = self.stem_conv2(conv_out)
        conv_out = self.max_pool_in_time(conv_out)

        # Additionals
        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        conv_out = self.stem_conv3(conv_out)
        conv_out = self.max_pool_in_freq(conv_out)

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        conv_out = self.stem_conv4(conv_out)
        conv_out = self.max_pool_in_time(conv_out)

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        conv_out = self.stem_conv5(conv_out)
        conv_out = self.max_pool_square(conv_out)

        return conv_out


if __name__ == "__main__":
    print("Main")




