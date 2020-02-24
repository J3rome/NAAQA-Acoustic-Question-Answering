from collections import OrderedDict

import torch
import torch.nn as nn

from models.CLEAR_nlp import Question_pipeline
from models.CLEAR_feature_extractor import Original_Film_Extractor, Freq_Time_Extractor
from models.blocks.FiLM_layers import FiLMed_resblock
from models.blocks.Classifiers import Conv_classifier, Fcn_classifier
from models.utils import get_trainable_childs

from models.utils import append_spatial_location, Conv2d_padded
from models.Resnet_feature_extractor import Resnet_feature_extractor
from utils.random import set_random_state, get_random_state


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
        #self.question_pipeline = Question_pipeline_no_GRU(config, nb_words, dropout_drop_prob, sequence_padding_idx)

        # Image Pipeline
        self.image_pipeline = Original_Film_Extractor(config, input_image_channels)
        # self.image_pipeline = Freq_Time_Extractor(input_image_channels, config['stem']['conv_out'])
        # self.image_pipeline = Separable_conv_image_pipeline(config, input_image_channels)

        stem_conv_in = self.image_pipeline.get_out_channels() + 2 if config['stem']['spatial_location'] else 0
        self.stem_conv = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels= stem_conv_in,
                                   out_channels=config['stem']['conv_out'], kernel_size=[3, 3],
                                   stride=1, dilation=1, bias=False, padding='SAME')),
            ('batchnorm', nn.BatchNorm2d(config['stem']['conv_out'], eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        # Question and Image Fusion
        resblock_out_channels = config['stem']['conv_out']
        resblock_in_channels = resblock_out_channels + 2 if config['resblock']['spatial_location'] else 0
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
                                           pooling_type=config['classifier']['global_pool_type'],
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

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        conv_out = self.stem_conv(conv_out)

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

        if hasattr(self.image_pipeline, 'feature_extractor') and self.image_pipeline.feature_extractor:
            state_dict = {k : p for k,p in state_dict.items() if 'feature_extractor' not in k}

        return state_dict

    def train(self, mode=True):
        # Only call train if model is in eval mode
        if self.training != mode:
            super(CLEAR_FiLM_model, self).train(mode)

        # Keep the feature extractor in eval mode
        if hasattr(self.image_pipeline, 'feature_extractor') and self.image_pipeline.feature_extractor and self.image_pipeline.feature_extractor.training:
            self.image_pipeline.feature_extractor.eval()

    def to(self, device=None, dtype=None, non_blocking=False):
        # Copy model to device only once
        if self.current_device != device:
            self.current_device = device
            super(CLEAR_FiLM_model, self).to(device, dtype, non_blocking)
