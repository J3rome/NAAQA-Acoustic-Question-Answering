from collections import OrderedDict, namedtuple

import torch.nn as nn

from models.CLEAR_nlp import Question_pipeline
from models.CLEAR_feature_extractor import Original_Film_Extractor, Freq_Time_Pooled_Extractor
from models.CLEAR_feature_extractor import Freq_Time_Separated_Extractor, Freq_Time_Interlaced_Extractor
from models.CLEAR_feature_extractor import Freq_Time_Separated_No_Pool_Extractor
from models.blocks.FiLM_layers import FiLMed_resblock
from models.blocks.Classifiers import Conv_classifier, Fcn_classifier

from models.utils import append_spatial_location, Conv2d_padded
from models.Resnet_feature_extractor import Resnet_feature_extractor
from utils.Reproducibility_Handler import Reproductible_Block


class CLEAR_FiLM_Malimo_model(nn.Module):
    def __init__(self, config, input_image_channels, nb_words, nb_answers, sequence_padding_idx=0):
        super(CLEAR_FiLM_Malimo_model, self).__init__()

        self.current_device = 'cpu'
        self.config = config

        self.early_stopping = config.get('early_stopping', None)
        self.early_stopping = self.early_stopping if self.early_stopping and self.early_stopping['enable'] else None

        dropout_drop_prob = float(config['optimizer'].get('dropout_drop_prob', 0.0))

        film_layer_transformation = config['resblock'].get('film_projection_type', None)

        self.malimo_input_pooling_type = config.get('malimo', {}).get('pooling_type', 'max')

        # Reproducibility Handling
        initial_random_state = Reproductible_Block.get_random_state()

        # Question Pipeline
        with Reproductible_Block(initial_random_state, 10):
            self.question_pipeline = Question_pipeline(config, nb_words, dropout_drop_prob, sequence_padding_idx)

        # Image Pipeline
        with Reproductible_Block(initial_random_state, 125):
            extractor_config = config['image_extractor']
            extractor_type = extractor_config['type'].lower()
            #input_image_channels += len(extractor_config['spatial_location'])
            if extractor_type == "film_original":
                self.image_pipeline = Original_Film_Extractor(config['image_extractor'], input_image_channels)
            elif extractor_type == "resnet":
                self.image_pipeline = Resnet_feature_extractor()
            elif extractor_type == 'resnet_h5':
                # Preextracted resnet features mock object
                mock_image_pipeline = namedtuple('resnet_h5', 'get_out_channels')
                self.image_pipeline = mock_image_pipeline(get_out_channels=lambda: extractor_config['out'])
            elif extractor_type == "freq_time_separated":
                self.image_pipeline = Freq_Time_Separated_Extractor(extractor_config, input_image_channels)
            elif extractor_type == "freq_time_separated_no_pooling":
                self.image_pipeline = Freq_Time_Separated_No_Pool_Extractor(extractor_config, input_image_channels)
            elif extractor_type == "freq_time_interlaced":
                self.image_pipeline = Freq_Time_Interlaced_Extractor(extractor_config, input_image_channels)
            elif extractor_type == "freq_time_pool":
                self.image_pipeline = Freq_Time_Pooled_Extractor(input_image_channels, config['stem']['conv_out'])

        # with Reproductible_Block(initial_random_state, 42):
        #     stem_conv_in = self.image_pipeline.get_out_channels() + len(config['stem']['spatial_location'])
        #     self.stem_conv = nn.Sequential(OrderedDict([
        #         ('conv', Conv2d_padded(in_channels=stem_conv_in,
        #                                out_channels=config['stem']['conv_out'], kernel_size=[3, 3],
        #                                stride=1, dilation=1, bias=False, padding='SAME')),
        #         ('batchnorm', nn.BatchNorm2d(config['stem']['conv_out'], eps=0.001)),
        #         ('relu', nn.ReLU(inplace=True))
        #     ]))

        # Malimo
        with Reproductible_Block(initial_random_state, 1256):
            malimo_hidden_size = config.get('malimo', {}).get('hidden_size', 512)

            self.malimo_rnn = nn.GRU(input_size=self.image_pipeline.get_out_channels(),
                                     hidden_size=malimo_hidden_size,
                                     batch_first=True,
                                     dropout=0)

        # Question and Image Fusion
        with Reproductible_Block(initial_random_state, 4242):
            #resblock_in_channels = config['stem']['conv_out'] + len(config['resblock']['spatial_location'])
            resblock_in_channels = self.image_pipeline.get_out_channels() + len(config['resblock']['spatial_location'])
            self.resblocks = nn.ModuleList()
            self.resblocks_malimo = nn.ModuleList()
            for resblock_out_channels in config['resblock']['conv_out']:
                self.resblocks.append(FiLMed_resblock(in_channels=resblock_in_channels,
                                                      out_channels=resblock_out_channels,
                                                      context_size=config["question"]["rnn_state_size"],
                                                      kernel1=config['resblock']['kernel1'],
                                                      kernel2=config['resblock']['kernel2'],
                                                      dropout_drop_prob=dropout_drop_prob,
                                                      film_layer_transformation=film_layer_transformation))

                self.resblocks_malimo.append(FiLMed_resblock(in_channels=resblock_out_channels,
                                                             out_channels=resblock_out_channels,
                                                             context_size=malimo_hidden_size,
                                                             kernel1=config['resblock']['kernel1'],
                                                             kernel2=config['resblock']['kernel2'],
                                                             dropout_drop_prob=dropout_drop_prob,
                                                             film_layer_transformation=film_layer_transformation))

                resblock_in_channels = resblock_out_channels + len(config['resblock']['spatial_location'])


        with Reproductible_Block(initial_random_state, 425):
            if config['classifier'].get('type', '').lower() == 'conv':
                # Classification (Via 1x1 conv & GlobalPooling)
                self.classifier = Conv_classifier(in_channels=resblock_out_channels,
                                                  projection_size=config["classifier"]['projection_size'],
                                                  output_size=nb_answers,
                                                  pooling_type=config['classifier']['global_pool_type'],
                                                  spatial_location_layer=config['classifier']['spatial_location'],
                                                  dropout_drop_prob=dropout_drop_prob)
            else:
                # Fully connected classifier
                self.classifier = Fcn_classifier(in_channels=resblock_out_channels,
                                                 classifier_conv_out=config['classifier']['conv_out'],
                                                 hidden_layer_size=config["classifier"]['projection_size'],
                                                 output_size=nb_answers,
                                                 pooling_type=config['classifier']['global_pool_type'],
                                                 spatial_location_layer=config['classifier']['spatial_location'],
                                                 dropout_drop_prob=dropout_drop_prob)



        # Set back initial seed
        Reproductible_Block.set_random_state(initial_random_state)

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
        if self.config['image_extractor']['type'].lower() != 'resnet_h5':
            conv_out = self.image_pipeline(input_image)
        else:
            conv_out = input_image

        # Global Max Pooling
        if self.malimo_input_pooling_type == 'max':
            image_vector, _ = conv_out.max(dim=3)
            image_vector = image_vector.transpose(2,1)
        else:
            image_vector = conv_out.mean(dim=[2, 3])

        # if self.config['stem']['spatial_location']:
        #     conv_out = append_spatial_location(conv_out, axis=self.config['stem']['spatial_location'])

        #conv_out = self.stem_conv(conv_out)

        # Malimo module
        malimo_rnn_out, malimo_rnn_hidden = self.malimo_rnn(image_vector)

        # TODO : GRU on pooled conv_out and feed to resblock_malimo

        # Question and Audio fusion
        for i, (resblock, resblock_malimo) in enumerate(zip(self.resblocks, self.resblocks_malimo)):
            conv_out = resblock(conv_out, rnn_hidden_state,
                                spatial_location=self.config['resblock']['spatial_location'])

            conv_out = resblock_malimo(conv_out, malimo_rnn_hidden, spatial_location=self.config['resblock']['spatial_location'])

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
            super(CLEAR_FiLM_Malimo_model, self).train(mode)

        # Keep the feature extractor in eval mode
        if hasattr(self.image_pipeline, 'feature_extractor') and self.image_pipeline.feature_extractor and self.image_pipeline.feature_extractor.training:
            self.image_pipeline.feature_extractor.eval()

    def to(self, device=None, dtype=None, non_blocking=False):
        # Copy model to device only once
        if self.current_device != device:
            self.current_device = device
            super(CLEAR_FiLM_Malimo_model, self).to(device, dtype, non_blocking)
