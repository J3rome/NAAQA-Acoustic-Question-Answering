import torch
import torch.nn.functional as F



# TODO : Credits to https://github.com/vickyliin/gradcam_plus_plus-pytorch

# TODO : Add GradCAMpp
# TODO : Why not extend torch.nn.module instead ? (would remove the need for __call__)
class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        gradcam = GradCAM.from_config(model_type='resnet', arch=resnet, layer_name='layer4')

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    """

    def __init__(self, model: torch.nn.Module, target_layers, apply_relu=False):
        self.model = model

        self.apply_relu = apply_relu
        self.target_layers = target_layers
        self.reverse_layer_map = {m: k for k, m in target_layers.items()}

        self.hooks = []
        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients[self.reverse_layer_map[module]] = grad_output[0]
        def forward_hook(module, input, output):
            self.activations[self.reverse_layer_map[module]] = output

        for target_layer in target_layers.values():
            self.hooks.append(target_layer.register_forward_hook(forward_hook))
            self.hooks.append(target_layer.register_backward_hook(backward_hook))

    # TODO : is useful ?
    def saliency_map_size(self, *input_size):
        device = next(self.model.parameters()).device
        self.model(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def forward(self, question, question_lengths, input_image, class_idx=None, pack_sequence=True, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        batch_size, nb_channels, img_height, img_width = input_image.size()

        logit, logit_softmaxed = self.model(question, question_lengths, input_image, pack_sequence=pack_sequence)

        if class_idx is None:
            class_idx = logit_softmaxed.max(1)[-1]

        scores = logit.gather(1, class_idx.unsqueeze(1))

        self.model.zero_grad()
        scores.backward(torch.ones_like(scores), retain_graph=retain_graph)

        saliency_maps = {}
        for layer_name, layer in self.target_layers.items():
            gradients = self.gradients[layer_name]
            activations = self.activations[layer_name]
            batch_size, grad_nb_channels, grad_height, grad_width = gradients.size()

            alpha = gradients.view(batch_size, grad_nb_channels, -1)
            if self.apply_relu:
                alpha = F.relu(alpha)
            alpha = alpha.mean(2)

            weights = alpha.view(batch_size, grad_nb_channels, 1, 1)

            saliency_map = (weights*activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            # TODO : F.upsample is deprecated
            saliency_map = F.upsample(saliency_map, size=(img_height, img_width), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
            saliency_maps[layer_name] = saliency_map

        return saliency_maps, scores, logit_softmaxed

    def __call__(self, question, question_lengths, input_image, class_idx=None, pack_sequence=True, retain_graph=False):
        return self.forward(question, question_lengths, input_image, class_idx, pack_sequence, retain_graph)
