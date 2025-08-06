# models/model_selector.py

def get_model(name, input_shape, num_classes, **kwargs):
    if name == "resnet1d":
        from .resnet1d import build_model
        return build_model(input_shape, num_classes, **kwargs)

    elif name == "xresnet1d":
        from .xresnet1d import build_model
        return build_model(input_shape, num_classes, **kwargs)

    elif name == "transformer":
        from .transformer import build_model
        return build_model(input_shape, num_classes, **kwargs)

    elif name == "cnn_transformer":
        from .cnn_transformer import build_model
        return build_model(input_shape, num_classes, **kwargs)

    elif name == "xlstm":
        from .xlstm import build_model
        return build_model(input_shape, num_classes, **kwargs)

    else:
        raise ValueError(f"Unknown model name: {name}")
