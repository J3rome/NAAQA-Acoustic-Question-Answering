import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet


# Replicate tensorflow 'SAME' padding (Taken from https://github.com/mlperf/inference/blob/master/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py#L40)
class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )



# TODO : Get size of tensor dynamically

class FiLM_layer(nn.Module):
    def __init__(self, in_channels, out_channels, save_gammas_betas=True):
        super(FiLM_layer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.save_gammas_betas = save_gammas_betas

        # TODO : Prealocate gamma and betas ? In Tensor ? Np Array ? Only if save_gammas_betas == True
        self.gammas = None
        self.betas = None

        self.params_vector = nn.Linear(self.in_channels, 2 * self.out_channels)     # FIXME : Original film have another multiplier : num_modules (which is 4 --> Number of resblock)
                                                                                    # FIXME : The linear layer is not in here on the original tho. This might be why we don't have *4 for the number of resblock (We are inside the resblock here)

    def forward(self, input_features, rnn_last_hidden_state):
        film_params = self.params_vector(rnn_last_hidden_state)

        gammas, betas = film_params.split(self.out_channels, dim=-1)

        if self.save_gammas_betas:
            self.gammas = gammas.detach() #.tolist()    # FIXME
            self.betas = betas.detach()#.tolist()

        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(input_features)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(input_features)
        return (gammas * input_features) + betas        # FIXME : Tensorflow implementation do 1 + gammas
                                                        # FIXME : Original implementation have a shift of 1 (Just after film_gen.decoder forward)


class FiLMed_resblock(nn.Module):
    def __init__(self, out_channels, context_size, first_kernel=(1, 1), second_kernel=(3, 3), spatial_location=True):
        super(FiLMed_resblock, self).__init__()

        # TODO : Add spatial location --> Adding spatial location will change the number of input channel for the first convolution

        self.conv1 = nn.Sequential(
            Conv2d_tf(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=first_kernel, stride=1, padding='SAME', dilation=1),

            nn.ReLU(inplace=True),
        )

        self.conv2 = Conv2d_tf(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=second_kernel, stride=1, padding='SAME', dilation=1)

        self.conv2_bn = nn.BatchNorm2d(out_channels)

        # Film Layer
        self.conv2_filmed = FiLM_layer(context_size, out_channels)

        self.conv2_relu = nn.ReLU(inplace=True)

    def forward(self, input_features, rnn_state):
        conv1_out = self.conv1(input_features)
        out = self.conv2(conv1_out)
        out = self.conv2_bn(out)
        out = self.conv2_filmed(out, rnn_state)
        out = self.conv2_relu(out)

        return out + conv1_out


class CLEAR_FiLM_model(nn.Module):
    def __init__(self, config, input_image_channels, nb_words, nb_answers, sequence_padding_idx=0):
        super(CLEAR_FiLM_model, self).__init__()

        # Question Pipeline
        self.word_emb = nn.Embedding(num_embeddings=nb_words,
                                     embedding_dim=config['question']['word_embedding_dim'],
                                     padding_idx=sequence_padding_idx)

        # TODO : Make sure we have the correct activation fct (Validate that default is tanh)
        self.rnn_state = nn.GRU(input_size=config['question']['word_embedding_dim'],
                                hidden_size=config["question"]["rnn_state_size"],
                                batch_first=True,
                                dropout=0)

        #### Image Pipeline

        # TODO : If RAW img, should pass through ResNet Feature Extractor Before

        ## Stem
        # TODO : Add spatial location
        self.stem_conv = nn.Sequential(
            Conv2d_tf(in_channels=input_image_channels, out_channels=config['stem']['conv_out'],
                      kernel_size=config['stem']['conv_kernel'], stride=1, padding='SAME', dilation=1),

            nn.BatchNorm2d(config['stem']['conv_out']),

            nn.ReLU(inplace=True)
        )

        ## Resblocks
        resblock_out_channels = config['stem']['conv_out']
        self.resblocks = nn.ModuleList()
        for i in range(config['resblock']['no_resblock']):
            self.resblocks.append(FiLMed_resblock(resblock_out_channels,
                                                  context_size=config["question"]["rnn_state_size"],
                                                  first_kernel=config['resblock']['kernel1'],
                                                  second_kernel=config['resblock']['kernel2'],
                                                  spatial_location=config['resblock']['spatial_location']))

        #### Classification
        # TODO : Spatial location
        self.classif_conv = nn.Sequential(
            Conv2d_tf(in_channels=resblock_out_channels, out_channels=config['classifier']['conv_out'],
                      kernel_size=config['classifier']['conv_kernel'], stride=1, padding="SAME", dilation=1),

            nn.BatchNorm2d(config['classifier']['conv_out']),

            nn.ReLU(inplace=True),
        )

        self.classif_hidden = nn.Sequential(
            nn.Linear(config['classifier']['conv_out'], config['classifier']['no_mlp_units']),

            nn.BatchNorm1d(config['classifier']['no_mlp_units']),

            nn.ReLU(inplace=True),
        )

        self.linear_out = nn.Linear(config['classifier']['no_mlp_units'], nb_answers)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, question, question_lengths, input_image, pack_sequence=True):
        # Question Pipeline
        word_emb = self.word_emb(question)
        if pack_sequence:
            word_emb = torch.nn.utils.rnn.pack_padded_sequence(word_emb, question_lengths, batch_first=True,
                                                               enforce_sorted=False)        # FIXME : Verify implication of enforce_sorted
        rnn_out, rnn_hidden = self.rnn_state(word_emb)

        if pack_sequence:
            rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        # We wan't an index to the last token  (len - 1)
        question_lengths = question_lengths - 1
        last_token_indexes = question_lengths.view(-1, 1, 1).expand(-1, 1, self.rnn_state.hidden_size).long()
        rnn_hidden_state = rnn_out.gather(1, last_token_indexes).view(-1, self.rnn_state.hidden_size)       # FIXME : Is rnn_hidden_state the correct name here ?

        # Image Pipeline
        # TODO : If RAW img, should pass through ResNet Feature Extractor Before
        conv_out = self.stem_conv(input_image)

        for resblock in self.resblocks:
            conv_out = resblock(conv_out, rnn_hidden_state)

        # Classification
        classif_out = self.classif_conv(conv_out)

        # Global Max Pooling (Max pooling over whole dimensions 3,4)
        classif_out, _ = classif_out.max(dim=2)
        classif_out, _ = classif_out.max(dim=2)

        classif_out = self.classif_hidden(classif_out)
        classif_out = self.linear_out(classif_out)
        classif_out = self.softmax(classif_out)

        return classif_out

    def get_gammas_betas(self):
        gammas = []
        betas = []
        for resblock in self.resblocks:
            betas.append(resblock.conv2_filmed.betas)
            gammas.append(resblock.conv2_filmed.gammas)

        return gammas, betas



if __name__ == "__main__":
    print("Main")




