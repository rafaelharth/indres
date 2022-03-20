"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

Loads the resnet model from shelf (i.e., with all parameters already fully trained). I have made almost no changes to this code.
"""

import torch
import torchvision
import settings

def load_model(hook_fn, feature_names=settings.FEATURE_NAMES, hook_modules=None, pretrained_override=None):
    model_fn = torchvision.models.__dict__[settings.MODEL]

    if settings.MODEL_FILE is None:
        model = model_fn(pretrained=pretrained_override if pretrained_override is not None else True)
    elif settings.MODEL_FILE == "<UNTRAINED>":
        model = model_fn(pretrained=pretrained_override if pretrained_override is not None else False)
    else:
        # no GPU :(
        # checkpoint = torch.load(settings.MODEL_FILE, map_location=device)
        checkpoint = torch.load(settings.MODEL_FILE, map_location=torch.device('cpu'))
        if type(checkpoint).__name__ == "OrderedDict" or type(checkpoint).__name__ == "dict":
            model = model_fn(num_classes=settings.NUM_CLASSES)
            ## this is reached
            if settings.MODEL_PARALLEL:
                  # the segmentation_data parallel layer will add 'module' before each layer name
                state_dict = {str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()}
                ## this is reached
            else:
                state_dict = checkpoint
                print('state_dict = checkpoint')
            model.load_state_dict(state_dict)
        else:
            if settings.MODEL == 'densenet161':
                # Fix old densenet pytorch names.
                model = model_fn(num_classes=settings.NUM_CLASSES)
                state_dict = checkpoint.state_dict()

                def rep(k):
                    for i in range(6):
                        k = k.replace(f"norm.{i}", f"norm{i}")
                        k = k.replace(f"relu.{i}", f"relu{i}")
                        k = k.replace(f"conv.{i}", f"conv{i}")
                    return k

                state_dict = {
                    rep(k): v for k, v in state_dict.items()
                }
                model.load_state_dict(state_dict)
            else:
                model = checkpoint


    if hook_fn is not None:
        for name in feature_names:
            if isinstance(name, list):
                # Iteratively retrieve the module
                hook_model = model
                for n in name:
                    hook_model = hook_model._modules.get(n)
            else:
                hook_model = model._modules.get(name)
            if hook_model is None:
                raise ValueError(f"Couldn't find feature {name}")
            if hook_modules is not None:
                hook_modules.append(hook_model)
            hook_model.register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
