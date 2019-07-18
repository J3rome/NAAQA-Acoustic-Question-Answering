import torch
import torch.nn as nn
import torchvision.models.resnet


# TODO : Get size of tensor dynamically

class FiLM_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FiLM_layer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gamma = None
        self.beta = None

        self.params_vector = nn.Linear(self.in_channels, 2 * self.out_channels)

    def forward(self, input_features, rnn_state):
        film_params = self.params_vector(rnn_state[:,0,:])

        # FIXME : I don't think unsqueezing is needed
        #film_params = film_params.unsqueeze(0).unsqueeze(0)

        tiled_params = film_params.repeat(1, 1, input_features.size(2), input_features.size(3))

        gammas = tiled_params[:, :, :, :self.out_channels]
        betas = tiled_params[:, :, :, self.out_channels:]

        out = (1 + gammas) * input_features + betas

        self.gamma = gammas[0, 0, 0, :]
        self.beta = betas[0, 0, 0, :]

        return out


class FiLMed_resblock(nn.Module):
    def __init__(self, out_channels, first_kernel=(1, 1), second_kernel=(3, 3), spatial_location=True):
        super(FiLMed_resblock, self).__init__()

        # TODO : Add spatial location --> Adding spatial location will change the number of input channel for the first convolution

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=first_kernel, stride=1, padding=0, dilation=1),

            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=second_kernel, stride=1, padding=0, dilation=1)

        self.conv2_bn = nn.BatchNorm2d(out_channels)

        # Film Layer
        #self.conv2_filmed, self.gamma, self.beta = FiLM_layer(out_channels)
        self.conv2_filmed = FiLM_layer(4096, out_channels)

        self.conv2_relu = nn.ReLU(inplace=True)

    def forward(self, input_features, rnn_state):
        conv1_out = self.conv1(input_features)
        out = self.conv2(conv1_out)
        out = self.conv2_bn(out)
        out = self.conv2_filmed(out, rnn_state)
        out = self.conv2_relu(out)

        return out + conv1_out




class CLEAR_FiLM_model(nn.Module):
    def __init__(self, config, no_words, no_answers):
        super(CLEAR_FiLM_model, self).__init__()

        seq_length = 0

        # Dropout

        # Inputs
            # question (And seq_length ?)
            # Image
            # Answer (Ground Truth)

        # Layers
            # Question Pipeline
                # Question -> Word Embedding
                # Word Embedding -> GRU

            # Image Pipeline
                # Image -> Spatial Location layer (Not sure what it is..)
                # Image + Spatial -> Stem convolution -> Batch Norm -> Relu

                # Resblocks (Can have multiple resblock)
                    # Stem Conv -> Spatial Location Layer (Input can also be output of last resblock
                    # Stem Conv + Spatial -> Conv1 -> relu
                    # Conv1 -> Conv2
                    # Conv2 -> BN
                    ### GRU Hidden State * BN -> Conv2_FiLMed -> Relu
                    # Residual (Conv1 + Conv2_FiLMed_Relu)

            # Classifier
                # Last Resblock Output -> Spatial Location
                # Last Resblock Output + Spatial -> Conv -> BN -> Relu
                # Conv -> Max Pooling (Keep 1 data per dimension)       # TODO : Max Vs Average Vs Concat(Max,Average)
                # MaxPool -> Fully Connected -> BN -> Relu
                # Hidden Fully Connected -> Fully Connected -> Output (Softmax layer)

        # Loss
            # Cross entropy

        # Question Pipeline
        # FIXME : seq_length must be fed as the first dimension
        self.word_emb = nn.Embedding(num_embeddings=no_words,
                                     embedding_dim=config['question']['word_embedding_dim'])

        # TODO : Make sure we have the correct activation fct
        self.rnn_state = nn.GRU(input_size=config['question']['word_embedding_dim'],
                                hidden_size=config["question"]["rnn_state_size"],
                                batch_first=True,
                                dropout=0)

        #### Image Pipeline

        # TODO : If RAW img, should pass through ResNet Feature Extractor Before

        ## Stem
        # TODO : Add spatial location
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=config['stem']['conv_out'],
                      kernel_size=config['stem']['conv_kernel'], stride=1, padding=0, dilation=1),

            nn.BatchNorm2d(config['stem']['conv_out']),      # FIXME : Not sure this is correct

            nn.ReLU(inplace=True)
        )

        ## Resblocks
        resblock_out_channels = config['stem']['conv_out']
        self.resblocks = nn.ModuleList()
        for i in range(config['resblock']['no_resblock']):
            self.resblocks.append(FiLMed_resblock(resblock_out_channels,
                                                  first_kernel=config['resblock']['kernel1'],
                                                  second_kernel=config['resblock']['kernel2'],
                                                  spatial_location=config['resblock']['spatial_location']))

        #### Classification
        # TODO : Spatial location
        self.classif_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=config['classifier']['conv_out'],
                      kernel_size=config['classifier']['conv_kernel'], stride=1, padding=0, dilation=1),

            nn.BatchNorm2d(config['classifier']['conv_out']),    # FIXME : Not sure this is correct

            nn.ReLU(inplace=True),
        )

        self.global_max_pooling = nn.AdaptiveMaxPool2d((1,1))

        self.classif_hidden = nn.Sequential(
            nn.Linear(config['classifier']['conv_out'], config['classifier']['no_mlp_units']),

            nn.BatchNorm2d(config['classifier']['no_mlp_units']),  # FIXME : Not sure this is correct

            nn.ReLU(inplace=True),
        )

        # FIXME : Rename to softmax ? OR are we missing the softmax ?
        self.out = nn.Linear(config['classifier']['no_mlp_units'], no_answers)

    def forward(self, question, input_image):
        # Question Pipeline
        rnn_out = self.word_emb(question)
        rnn_out, rnn_hidden = self.rnn_state(rnn_out)

        # Image Pipeline
        # TODO : If RAW img, should pass through ResNet Feature Extractor Before
        conv_out = self.stem_conv(input_image)

        for resblock in self.resblocks:
            conv_out = resblock(conv_out, rnn_out)

        # Classification
        classif_out = self.classif_conv(conv_out)
        classif_out = self.global_max_pooling(classif_out)
        classif_out = self.classif_hidden(classif_out)
        classif_out = self.out(classif_out)

        # FIXME : Are we missing a softmax here ?

        return classif_out


if __name__ == "__main__":
    print("Main")




