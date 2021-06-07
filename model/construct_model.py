from tensorflow.keras import layers
from tensorflow.python.keras.engine.keras_tensor import KerasTensor


def upsample(dst: KerasTensor, pre_name: str = '', idx: int = 0) -> layers.Layer:
    """
    A customized up-sampling layer using bi-linear Interpolation
    :param dst: the target tensor, need it's size for up-sample
    :param pre_name: the layer's prefix name
    :param idx: the index of layer
    :return: the up-sample layer
    """
    from tensorflow import image
    import numpy as np
    np.random.rand(30)
    return layers.Lambda(lambda x, w, h: image.resize(x, (w, h)),
                         arguments={'w': dst.shape[1], 'h': dst.shape[2]}, name=f"{pre_name}_upsample{idx}")


def create_bn_conv(input: KerasTensor, filter: int, kernel_size: int, dilation_rate: int = 1, pre_name: str = '',
                   idx: int = 0, padding='same', activation: str = 'relu') -> KerasTensor:
    """
    A basic convolution-batchnormalization struct
    :param input: the input tensor
    :param filter: the filter number used in Conv layer
    :param kernel_size: the filter kernel size in Conv layer
    :param dilation_rate: the dilation rate for convolution
    :param pre_name: the layer's prefix name
    :param idx: the index of layer
    :param padding: same or valid
    :param activation: activation for Conv layer
    """
    conv = layers.Conv2D(filter, (kernel_size, 1), padding=padding,
                         dilation_rate=(dilation_rate, dilation_rate), activation=activation,
                         name=f"{pre_name}_conv{idx}")(input)
    bn = layers.BatchNormalization(name=f"{pre_name}_bn{idx}")(conv)
    return bn


def create_u_encoder(input: KerasTensor, filter: int, kernel_size: int, pooling_size: int,
                     middle_layer_filter: int, depth: int, pre_name: str = '', idx: int = 0,
                     padding: str = 'same', activation: str = 'relu') -> KerasTensor:
    """
    U-unit: a repetitive sub-structure of SalientSleepNet
    :param input: the input tensor
    :param filter: the filter number used in the first and last Conv layers of U-unit
    :param middle_layer_filter: the filter number used in the middle layers of U-unit
    :param pooling_size: the pool size used in MaxPooling layers
    :param kernel_size: the filter kernel size in Conv layers
    :param depth: the depth of U-unit
    :param pre_name: the layer's prefix name
    :param idx: the index of layer
    :param padding: same or valid
    :param activation: activation for Conv layer
    """
    l_name = f"{pre_name}_U{idx}_enc"
    from_encoder = []
    conv_bn0 = create_bn_conv(input, filter, kernel_size,
                             pre_name=l_name, idx=0,
                             padding=padding, activation=activation)
    conv_bn = conv_bn0
    for d in range(depth - 1):
        conv_bn = create_bn_conv(conv_bn, middle_layer_filter, kernel_size,
                                 pre_name=l_name, idx=d + 1,
                                 padding=padding, activation=activation)
        from_encoder.append(conv_bn)
        if d != depth - 2:
            conv_bn = layers.MaxPooling2D((pooling_size, 1), name=f"{l_name}_pool{d + 1}")(conv_bn)

    conv_bn = create_bn_conv(conv_bn, middle_layer_filter, kernel_size,
                             pre_name=l_name, idx=depth,
                             padding=padding, activation=activation)

    l_name = f"{pre_name}_U{idx}_dec"
    for d in range(depth - 1, 0, -1):
        conv_bn = upsample(from_encoder[-1], pre_name=l_name, idx=d)(conv_bn)
        ch = filter if d == 1 else middle_layer_filter
        conv_bn = create_bn_conv(layers.concatenate([conv_bn, from_encoder.pop()]),
                                 ch, kernel_size, pre_name=l_name,
                                 idx=d, padding=padding, activation=activation)

    return layers.add([conv_bn, conv_bn0])


def create_mse(input: KerasTensor, filter: int, kernel_size: int, dilation_rates: list, pre_name: str = "",
               idx: int = 0, padding: str = 'same', activation: str = "relu") -> KerasTensor:
    """
    Multi-scale Extraction Module: a repetitive sub-structure of SalientSleepNet
    :param input: the input tensor
    :param filter: the filter number used in Conv layers
    :param kernel_size: the filter kernel size in Conv layers
    :param dilation_rates: the dilation rates for convolution
    :param pre_name: the layer's prefix name
    :param idx: the index of layer
    :param padding: same or valid
    :param activation: activation for Conv layer
    """
    l_name = f"{pre_name}_mse{idx}"

    convs = []
    for (i, dr) in enumerate(dilation_rates):
        conv_bn = create_bn_conv(input, filter, kernel_size, dilation_rate=dr,
                                 pre_name=l_name, idx=1 + i, padding=padding, activation=activation)
        convs.append(conv_bn)

    from functools import reduce
    con_conv = reduce(lambda l, r: layers.concatenate([l, r]), convs)

    down = layers.Conv2D(filter * 2, (kernel_size, 1), name=f"{l_name}_downconv1",
                         padding=padding, activation=activation)(con_conv)
    down = layers.Conv2D(filter, (kernel_size, 1), name=f"{l_name}_downconv2",
                         padding=padding, activation=activation)(down)
    out = layers.BatchNormalization(name=f"{l_name}_bn{len(dilation_rates) + 1}")(down)

    return out
