
import sys
import os
import re
import mxnet as mx
import collections
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config

########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate',])


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])


# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)



def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def Conv2dSamePadding(data, out_channels=1, kernel_size=(1,1), stride=(1,1), dilation=1, groups=1, bias=True, pad=(0,0), name=""):
    return mx.sym.Convolution(data=data, num_filter=out_channels, kernel=kernel_size, num_group=groups, stride=stride, pad=pad, no_bias=bias, name=name)
    """
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    sh, sw = stride
    oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
    pad_h = max((oh - 1) * stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
    pad_w = max((ow - 1) * stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
    return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    """

def relu_fn(x, name=""):
    return x * mx.symbol.Activation(data=x, act_type='sigmoid', name=name)
    #return x * mx.sym.sigmoid(x, name=name)

def drop_connect(inputs, p, training):
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += mx.nd.random.normal(shape=(batch_size, 1, 1, 1), dtype = inputs.dtype)
    binary_tensor = random_tensor.floor()
    output = inputs / keep_prob * binary_tensor
    return output


def MBConvBlock(data, block_args, global_params, drop_connect_rate=None, name=""):
    _bn_mom = 1 - global_params.batch_norm_momentum
    _bn_eps = global_params.batch_norm_epsilon
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    id_skip = block_args.id_skip # skip connection and drop connect

    # Expansion phase
    inp = block_args.input_filters
    oup = block_args.input_filters * block_args.expand_ratio # number of output channels

    # Depthwise convolution phase
    k = block_args.kernel_size
    s = block_args.stride

    # Squeeze and Excitation layer, if desired
    if has_se:
        num_squeezed_channels = max(1, int(block_args.input_filters * block_args.se_ratio))

    # Output phase
    final_oup = block_args.output_filters


    # Expansion and Depthwise Convolution
    x = data;
    if block_args.expand_ratio != 1:
        x = Conv2dSamePadding(data=x, out_channels=oup, kernel_size=(1,1), bias=False, name="expand_conv-"+name)
        x = mx.sym.BatchNorm(data=x, name='bn0-'+name, momentum=_bn_mom, eps=_bn_eps)
        x = relu_fn(x, name="relu_fn-expand-"+name)
    if isinstance(s, int):
        stride = (s,s)
    else:
        if len(s) == 1:
            stride = (s[0],s[0])
        else:
            stride = (s[0],s[1])
    x = Conv2dSamePadding(data=x, out_channels=oup, groups=oup, #groups makes it depthwise
        kernel_size=(k,k), stride=stride, pad=((k-1)//2, (k-1)//2), bias=False, name="depthwise_conv-"+name)
    x=mx.sym.BatchNorm(data=x, name="bn1-"+name, momentum=_bn_mom, eps=_bn_eps)
    x = relu_fn(x, name="relu_fn-"+name)

    # Squeeze and Excitation
    if has_se:
        x_squeezed = mx.sym.Pooling(data=x, pool_type='avg', kernel=(1,1), global_pool=True, name="avg_pool-"+name)
        x_squeezed = Conv2dSamePadding(data=x_squeezed, out_channels=num_squeezed_channels, kernel_size=(1,1), name="se_reduce-"+name)
        x_squeezed = relu_fn(x_squeezed, name="relu_fn-se-"+name)
        x_squeezed = Conv2dSamePadding(data=x_squeezed, out_channels=oup, kernel_size=(1,1), name="se_expand-"+name)
        x_squeezed = mx.symbol.Activation(data=x_squeezed, act_type='sigmoid', name="se_sigmoid-"+name)
        x= mx.symbol.broadcast_mul(x, x_squeezed)
        #x = mx.sym.sigmoid(data=x_squeezed, name="sogmoid") * x

    x = Conv2dSamePadding(data=x, out_channels=final_oup, kernel_size=(1,1), bias=False, name='project_conv-'+name)
    x=mx.sym.BatchNorm(data=x, name="bn2-"+name, momentum=_bn_mom, eps=_bn_eps)

    # Skip connection and drop connect
    input_filters, output_filters = block_args.input_filters, block_args.output_filters
    if id_skip and block_args.stride == 1 and input_filters == output_filters:
        if drop_connect_rate:
            x = drop_connect(x, p=drop_connect_rate, training=is_train, name="drop_connect-"+name)
        x = x + data

    return x

def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]

class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings



def efficientnet(width_coefficient=None, depth_coefficient=None,
                 dropout_rate=0.2, drop_connect_rate=0.2):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None
    )

    return blocks_args, global_params

def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    #if model_name.startswith('efficientnet'):
    if model_name.startswith('efficientnet'):
        w, d, _, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def get_symbol():
    arch=config.arch
    blocks_args, global_params = get_model_params(arch, None)

    data = mx.symbol.Variable(name="data") # 224
    data = data-127.5
    data = data*0.0078125

    bn_mom = 1 - global_params.batch_norm_momentum
    bn_eps = global_params.batch_norm_epsilon

    # Stem
    in_channels = 3  # rgb
    out_channels = round_filters(32, global_params)  # number of output channels

    #x = Conv2dSamePadding(data=data, out_channels=out_channels, kernel_size=(3,3), stride=(2,2), pad=(1,1), bias=False)
    x = Conv2dSamePadding(data=data, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), pad=(1,1), bias=False)
    x = mx.sym.BatchNorm(data=x, name="bn0", momentum=bn_mom, eps=bn_eps)
    x = relu_fn(x)

    for idx,block_args in enumerate(blocks_args):
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, global_params),
                output_filters=round_filters(block_args.output_filters, global_params),
                num_repeat=round_repeats(block_args.num_repeat, global_params)
        )

        # The first block needs to take care of stride and filter size increase.
        x = MBConvBlock(x, block_args, global_params, name="block"+str(idx))
        if block_args.num_repeat > 1:
            block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
        for rep in range(block_args.num_repeat - 1):
            x = MBConvBlock(x, block_args, global_params, name="block"+str(idx)+"_repeat"+str(rep))

    # save network
    #mx.viz.plot_network(x, shape={"data":(1, 3, 112, 112)}).view()
    #mx.viz.print_summary(x, shape={"data":(1, 3, 112, 112)}).view()


    # Head
    in_channels = block_args.output_filters  # output of final block
    # ori efficientnet
    """
    out_channels = round_filters(1280, global_params)
    x = Conv2dSamePadding(data=x, out_channels=out_channels, kernel_size=(1,1), bias=False)
    x = mx.sym.BatchNorm(data=x, momentum=bn_mom, eps=bn_eps)
    x = relu_fn(x)
    x = mx.sym.Pooling(data=x, pool_type='avg', global_pool=True, kernel=(1,1), name="avg_pool")

    if global_params.dropout_rate:
        x = mx.sym.Dropout(data=x, p=global_params.dropout_rate)
    fc1 = mx.sym.FullyConnected(data=x, num_hidden=config.emb_size, name='fc1')
    """
    out_channels = config.emb_size
    x = Conv2dSamePadding(data=x, out_channels=out_channels, kernel_size=(1,1), bias=False)
    x = mx.sym.BatchNorm(data=x, momentum=bn_mom, eps=bn_eps)
    x = relu_fn(x)
    fc1 = mx.sym.Pooling(data=x, pool_type='avg', global_pool=True, kernel=(1,1), name="fc1")
    fc1 = mx.sym.Flatten(data=fc1)

    return fc1;

