import torch
import torch.nn as nn
import torchvision.models.resnet


# TODO : Get size of tensor dynamically

class FiLM_layer(nn.Module):
    def __init__(self):
        super(FiLM_layer, self).__init__()

        # TODO : Implement this


class FiLMed_resblock(nn.Module):
    def __init__(self, first_kernel=(1,1), second_kernel=(3,3)):
        super(FiLMed_resblock, self).__init__()

        # TODO : Add spatial location

        # TODO : Get this dynamically ? Should probably be calculated outside of the constructor. Simply pass the size
        input_size = 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=input_size,
                      kernel_size=first_kernel, stride=1, padding=0, dilation=1),

            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=input_size,
                      kernel_size=second_kernel, stride=1, padding=0, dilation=1)

        # TODO : Retrieve this dynamically
        conv2_nb_feature = 3
        self.conv2_bn = nn.BatchNorm2d(conv2_nb_feature)

        # Film Layer
        self.conv2_filmed = FiLM_layer()

        self.conv2_relu = nn.ReLU(inplace=True)

        self.out = self.conv2_relu + self.conv1


class CLEAR_FiLM_model(nn.Module):
    def __init__(self, config, no_words, no_answers, device):
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
        self.word_emb = nn.Embedding(num_embeddings=no_words,
                                embedding_dim=config['question']['word_embedding_dim'])

        # TODO : Make sure we have the correct activation fct
        self.rnn_state = nn.GRU(input_size=seq_length,
                           hidden_size=config["question"]["rnn_state_size"],
                           batch_first=True,
                           dropout=0)

        #### Image Pipeline

        ## Stem
        # TODO : Add spatial location
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=config['stem']['conv_out'],
                      kernel_size=config['stem']['conv_kernel'], stride=1, padding=0, dilation=1),

            nn.BatchNorm2d(config['stem']['conv_out']),      # FIXME : Not sure this is correct

            nn.ReLU(inplace=True)
        )

        ## Resblocks
        self.resblocks = []
        for i in range(config['resblock']['no_resblock']):

            # TODO : Create FiLMed_Resblock nn.Module (With spatial location)
            self.resblocks.append(i)

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

        self.out = nn.Linear(config['classifier']['no_mlp_units'], no_answers)


    def forward(self, input):
        print("YOLO")

if __name__ == "__main__":
    print("Main")



