"""Grad-CAM: which part of the image made the network decide what it decided?

It takes the gradient of the predicted class score with respect to the last
conv layer's feature maps, uses those gradients to weight the maps, and
upsamples the result into a heatmap over the input image.
(Selvaraju et al., 2017, "Grad-CAM: Visual Explanations from Deep Networks")
"""
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, layer):
        self.model = model.eval()
        self.activations = self.gradients = None
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        """x: (1, 3, H, W). Returns (heatmap [H, W] in 0..1, predicted class, probabilities)."""
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        probs = logits.softmax(1)[0]
        cls = int(probs.argmax()) if class_idx is None else class_idx
        self.model.zero_grad()
        logits[0, cls].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # importance of each feature map
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.numpy(), cls, probs.detach().numpy()
