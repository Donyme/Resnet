import torch
from torch.functional import Tensor
import torch.nn as nn

""" This script defines the network.
"""

class ResNet(nn.Module):
    def __init__(self,
            resnet_version,
            resnet_size,
            num_classes,
            first_num_filters,
        ):
        """
        1. Define hyperparameters.
        Args:
            resnet_version: 1 or 2, If 2, use the bottleneck blocks.
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsampling block layer.
        
        2. Classify a batch of input images.

        Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 2n/3n  | 2n/3n  | 2n/3n  | 1           |
        #filters        | 16    | 16(*4) | 32(*4) | 64(*4) | num_classes |

        n = #residual_blocks in each stack layer = self.resnet_size
        The standard_block has 2 layers each.
        The bottleneck_block has 3 layers each.
        
        Example of replacing:
        standard_block      conv3-16 + conv3-16
        bottleneck_block    conv1-16 + conv3-16 + conv1-64

        Args:
            inputs: A Tensor representing a batch of input images.
        
        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """
        super(ResNet, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        ### YOUR CODE HERE
        # define conv1
        self.start_layer = nn.Conv2d(in_channels = 3, out_channels=self.first_num_filters , kernel_size=1)

        standard_block   = 'standard_block'
        bottleneck_block = 'bottleneck_block'
        ### YOUR CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters, 
                eps=1e-5, 
                momentum=0.997,
            )
        if self.resnet_version == 1:
            block_fn = standard_block
        else:
            block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        for i in range(3):
            filters = self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, self.first_num_filters))
        self.output_layer = output_layer(filters*4, self.resnet_version, self.num_classes)

    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        ### YOUR CODE HERE
        self.batch_norm = torch.nn.BatchNorm2d(num_features)
        self.relu       = torch.nn.ReLU()
        ### YOUR CODE HERE
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        batch_norm_out = self.batch_norm(inputs)
        relu_out = self.relu(batch_norm_out)

        return relu_out
        ### YOUR CODE HERE

class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()
        ### YOUR CODE HERE
        self.filters = filters
        self.projection_shortcut = projection_shortcut
        self.strides = strides
        self.first_num_filters = first_num_filters
        if self.projection_shortcut is not None:
            self.projection_shortcut.out_channels = self.filters
            self.projection_shortcut.stride = strides
        ### YOUR CODE HERE
        ### YOUR CODE HERE
        self.conv3_1 = nn.LazyConv2d(out_channels = self.filters, kernel_size = 3, stride = strides, padding = 1)
        self.batch_norm_relu_1 = batch_norm_relu_layer(self.filters)

        self.conv3_2 = nn.LazyConv2d(out_channels = self.filters, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm = torch.nn.BatchNorm2d(self.filters)
        self.relu = torch.nn.ReLU()

        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        conv3_1_out = self.conv3_1(inputs)
        batch_norm_relu_out1 = self.batch_norm_relu_1(conv3_1_out)
        conv3_2_out = self.conv3_2(batch_norm_relu_out1)
        batch_norm_out = self.batch_norm(conv3_2_out)
        skip_con_out = None 
        if self.projection_shortcut is not None:
          skip_con_out = batch_norm_out + self.projection_shortcut(inputs)
        else:
          skip_con_out = batch_norm_out + inputs
        
        relu_out1 = self.relu(skip_con_out)

        return relu_out1
        ### YOUR CODE HERE

class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution. NOTE: filters_out will be 4xfilters.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()
        ### YOUR CODE HERE
        # Hint: Different from standard lib implementation, you need pay attention to 
        # how to define in_channel of the first bn and conv of each block based on
        # Args given above.
        self.filters = filters
        self.projection_shortcut = projection_shortcut
        self.strides = strides
        self.first_num_filters = first_num_filters

        if projection_shortcut is not None:
          self.projection_shortcut.out_channels = self.filters
          self.projection_shortcut.stride = strides

        self.conv1 = nn.LazyConv2d(out_channels = (self.filters // 4), kernel_size = 1)
        self.batch_norm_relu_1 = batch_norm_relu_layer((self.filters // 4))
        self.conv2 = nn.LazyConv2d(out_channels = (self.filters // 4), kernel_size = 3, stride = strides, padding = 1)
        self.batch_norm_relu_2 = batch_norm_relu_layer((self.filters // 4))
        self.conv3 = nn.LazyConv2d(out_channels = self.filters, kernel_size = 1)
        self.batch_norm = torch.nn.BatchNorm2d(self.filters)
        self.relu = torch.nn.ReLU()

        ### YOUR CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.
        conv1_out = self.conv1(inputs)
        batch_norm_relu_1_out = self.batch_norm_relu_1(conv1_out)
        conv2_out = self.conv2(batch_norm_relu_1_out)
        batch_norm_relu_2_out = self.batch_norm_relu_2(conv2_out)
        conv3_out = self.conv3(batch_norm_relu_2_out)
        batch_norm_out = self.batch_norm(conv3_out)
        skip_con_out = None
        
        if self.projection_shortcut is not None:
          skip_con_out = batch_norm_out + self.projection_shortcut(batch_norm_relu_1_out)
        else:
          skip_con_out = batch_norm_out + inputs

        relu_out = self.relu(skip_con_out)

        return relu_out
        ### YOUR CODE HERE

class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        filters_out = filters * 4 if block_fn is 'bottleneck_block' else filters
        ### END CODE HERE
        # projection_shortcut = ?
        # Only the first block per stack_layer uses projection_shortcut and strides
        self.filters = filters_out
        self.strides = strides
        self.first_num_filters = first_num_filters
        self.projection_shortcut = nn.LazyConv2d(out_channels = self.filters, kernel_size = 1, stride=self.strides) #Random initialization, value set in network
        self.cur_stack = nn.ModuleList()
        if block_fn is 'bottleneck_block':
          for i in range(resnet_size):
            if i == 0:
              b_block = bottleneck_block(self.filters, projection_shortcut = self.projection_shortcut, strides = self.strides, first_num_filters = self.first_num_filters)
              self.cur_stack.append(b_block)
            else:
              b_block = bottleneck_block(self.filters, projection_shortcut = None, strides = 1, first_num_filters = self.first_num_filters)
              self.cur_stack.append(b_block)
        else:
          for i in range(resnet_size):
            if i == 0:
              s_block = standard_block(filters, projection_shortcut = self.projection_shortcut, strides = self.strides, first_num_filters = self.first_num_filters)
              self.cur_stack.append(s_block)
            else:
              s_block = standard_block(filters, projection_shortcut = None, strides = 1, first_num_filters = self.first_num_filters)
              self.cur_stack.append(s_block)
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        out = inputs
        for block_layer in self.cur_stack:
          out = block_layer(out)

        return out
        ### END CODE HERE

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
        if (resnet_version == 2):
            self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)
        
        ### END CODE HERE
        self.filters  = filters
        self.num_classes = num_classes
        self.max_pool = nn.AdaptiveAvgPool2d((1, 1)) # size set as (1, 1) cause need to use self.filters in fc_layer
        self.flatten  = nn.Flatten()
        self.fc_layer = nn.LazyLinear(self.num_classes)
        # self.softmax  = nn.functional.softmax()

        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        max_pool_out   = self.max_pool(inputs)
        flatten_out = self.flatten(max_pool_out)
        fc_out      = self.fc_layer(flatten_out)
        softmax_out = nn.functional.softmax(fc_out)

        return softmax_out
        ### END CODE HERE