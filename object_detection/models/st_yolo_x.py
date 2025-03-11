import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras.layers import layers


class YOLOX:
    def __init__(self, depth_mul, width_mul, num_anchors, input_shape, num_classes):
        """
        Initializes the YOLO_X model with given hyperparameters.

        Parameters:
            depth_mul (float): Depth multiplier to adjust model depth.
            width_mul (float): Width multiplier to adjust model width.
            num_anchors (int): Number of anchor boxes per grid cell.
            input_shape (tuple): Input image shape (height, width, channels).
            num_classes (int): Number of classes for object detection.
        """

        self.features_pick = [-3, -2, -1]
        self.depth_mul = depth_mul
        self.width_mul = width_mul
        self.use_depth_wise_conv = True
        self.regression_len = 4
        self.num_anchors = num_anchors
        self.use_object_scores = True
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = 'relu6'
        self.model_name = 'st_yolo_x'

    @staticmethod
    def activation_by_name(inputs, activation='relu', name=None):
        """
        Applies an activation function to the input tensor.

        Parameters:
            inputs (Tensor): The input tensor to apply activation on.
            activation (str, optional): The name of the activation function to use. Defaults to 'relu'.
            name (str, optional): A base name for the activation layer. Defaults to None.

        Returns:
            Tensor: The transformed tensor after applying the activation function.
        """

        # If no activation function is specified, return the input unchanged
        if activation is None:
            return inputs

        # Construct a layer name by combining the base name and activation function (if provided)
        layer_name = name and activation and name + activation

        # Apply the specified activation function to the input tensor
        return layers.Activation(activation=activation, name=layer_name)(inputs)

    def batch_norm_with_activation(self, inputs, activation=None, name=None):
        """
        Applies batch normalization followed by an optional activation function.

        Parameters:
            inputs (Tensor): The input tensor to normalize and activate.
            activation (str, optional): The activation function to apply after batch normalization. Defaults to None.
            name (str, optional): A base name for the batch normalization and activation layers. Defaults to None.

        Returns:
            Tensor: The transformed tensor after batch normalization and optional activation.
        """

        # Apply batch normalization to stabilize training and improve convergence
        nn = layers.BatchNormalization(name=name and name + 'bn')(inputs)

        # Apply activation function if specified
        if activation:
            nn = self.activation_by_name(nn, activation=activation, name=name)

        return nn

    @staticmethod
    def depth_wise_conv2d_no_bias(inputs, kernel_size, strides=1, padding='valid', name=None):
        """
        Applies a depth-wise convolution without bias.

        Parameters:
            inputs (Tensor): The input tensor to apply depth-wise convolution on.
            kernel_size (int or tuple): The size of the convolution kernel. If an integer is given, it is converted to (kernel_size, kernel_size).
            strides (int, optional): The stride size for the convolution. Defaults to 1.
            padding (str, optional): The padding type, either 'valid' (no padding) or 'same' (zero-padding). Defaults to 'valid'.
            name (str, optional): A base name for the convolution layer. Defaults to None.

        Returns:
            Tensor: The transformed tensor after applying depth-wise convolution.
        """

        # Ensure kernel_size is in tuple format (height, width)
        kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)

        # Convert padding to lowercase if it's a string
        if isinstance(padding, str):
            padding = padding.lower()

        # Apply depth-wise convolution without bias
        return layers.DepthwiseConv2D(kernel_size, strides=strides, padding='valid' if padding == 'valid' else 'same',
                                      use_bias=False, name=name and name + 'dw_conv')(inputs)

    @staticmethod
    def conv2d_no_bias(inputs, filters, kernel_size=1, strides=1, padding='valid', groups=1, name=None):
        """
        Applies a 2D convolution without bias.

        Parameters:
            inputs (Tensor): The input tensor to apply convolution on.
            filters (int): The number of output filters (channels).
            kernel_size (int or tuple, optional): The size of the convolution kernel. If an integer is given, it is converted to (kernel_size, kernel_size). Defaults to 1.
            strides (int, optional): The stride size for the convolution. Defaults to 1.
            padding (str, optional): The padding type, either 'valid' (no padding) or 'same' (zero-padding). Defaults to 'valid'.
            groups (int, optional): Number of groups for grouped convolution. Defaults to 1 (standard convolution).
            name (str, optional): A base name for the convolution layer. Defaults to None.

        Returns:
            Tensor: The transformed tensor after applying convolution.
        """

        # Ensure kernel_size is in tuple format (height, width)
        kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)

        # Convert padding to lowercase if it's a string
        if isinstance(padding, str):
            padding = padding.lower()

        # Ensure groups value is at least 1
        groups = max(1, groups)

        # Apply 2D convolution without bias
        return layers.Conv2D(filters, kernel_size, strides=strides, padding='valid' if padding == 'valid' else 'same',
                             use_bias=False, groups=groups, name=name and name + 'conv')(inputs)

    def conv_dw_pw_block(self, inputs, filters, kernel_size=1, strides=1, use_depth_wise_conv=False, activation='swish',
                         name=''):
        """
        Applies a depth-wise and point-wise convolution block with batch normalization and activation.

        Parameters:
            inputs (Tensor): The input tensor.
            filters (int): The number of output filters (channels).
            kernel_size (int, optional): The size of the convolution kernel. Defaults to 1.
            strides (int, optional): The stride size for the convolution. Defaults to 1.
            use_depth_wise_conv (bool, optional): Whether to use a depth-wise convolution before point-wise convolution. Defaults to False.
            activation (str, optional): The activation function to apply. Defaults to 'swish'.
            name (str, optional): A base name for the layers. Defaults to an empty string.

        Returns:
            Tensor: The transformed tensor after applying the depth-wise and point-wise convolution block.
        """

        # Start with the input tensor
        nn = inputs

        # Apply depth-wise convolution if enabled
        if use_depth_wise_conv:
            nn = self.depth_wise_conv2d_no_bias(nn, kernel_size, strides, padding='same', name=name)
            nn = self.batch_norm_with_activation(nn, activation=activation, name=name + 'dw_')

            # Reset kernel size and strides for point-wise convolution
            kernel_size, strides = 1, 1

        # Apply point-wise (1x1) convolution
        nn = self.conv2d_no_bias(nn, filters, kernel_size, strides, padding='same', name=name)

        # Apply batch normalization and activation
        nn = self.batch_norm_with_activation(nn, activation=activation, name=name)

        return nn

    def focus_stem(self, inputs, filters, kernel_size=3, strides=1, padding='valid', activation='swish', name=''):
        """
        Applies a stem block using depth-wise and point-wise convolution layers to extract features.

        Parameters:
            inputs (Tensor): The input tensor.
            filters (int): The number of output filters (channels) in the final convolution.
            kernel_size (int, optional): The size of the convolution kernel. Defaults to 3.
            strides (int, optional): The stride size for the convolution. Defaults to 1.
            padding (str, optional): The padding type, either 'valid' or 'same'. Defaults to 'valid'.
            activation (str, optional): The activation function to apply. Defaults to 'swish'.
            name (str, optional): A base name for the layers. Defaults to an empty string.

        Returns:
            Tensor: The transformed tensor after the focus stem block.
        """

        # First convolution block with fixed 12 filters, 3x3 kernel, and stride 2 for down-sampling
        nn = self.conv_dw_pw_block(inputs, 12, kernel_size=3, strides=2, activation=activation, name='st_')

        # Second convolution block with user-defined filters and kernel size
        nn = self.conv_dw_pw_block(nn, filters, kernel_size=kernel_size, strides=strides, activation=activation,
                                   name=name)

        return nn

    def spatial_pyramid_pooling(self, inputs, pool_size=(5, 9, 13), activation='swish', name=''):
        """
        Applies Spatial Pyramid Pooling (SPP) to extract multiscale features.

        Parameters:
            inputs (Tensor): The input tensor.
            pool_size (tuple, optional): A tuple defining the different pooling kernel sizes. Defaults to (5, 9, 13).
            activation (str, optional): The activation function to apply. Defaults to 'swish'.
            name (str, optional): A base name for the layers. Defaults to an empty string.

        Returns:
            Tensor: The transformed tensor after applying spatial pyramid pooling.
        """

        # Define the channel axis
        channel_axis = 1

        # Get the number of input channels
        input_channels = inputs.shape[channel_axis]

        # Initial 1x1 convolution to reduce the number of channels
        nn = self.conv_dw_pw_block(inputs, input_channels // 2, kernel_size=1, activation=activation, name=name + '1_')

        # Apply multiple max pooling layers with different pool sizes
        pp = [layers.MaxPool2D(pool_size=ii, strides=1, padding='same')(nn) for ii in pool_size]

        # Concatenate the original feature map with the pooled feature maps
        nn = tf.concat([nn, *pp], axis=channel_axis)

        # Final 1x1 convolution to restore the original number of channels
        nn = self.conv_dw_pw_block(nn, input_channels, kernel_size=1, activation=activation, name=name + '1_')

        return nn

    def csp_block(self, inputs, expansion=0.5, use_shortcut=True, use_depth_wise_conv=False, activation='swish', name=''):
        """
        Implements a Cross Stage Partial (CSP) block to enhance gradient flow and reduce computation.

        Parameters:
            inputs (Tensor): The input tensor.
            expansion (float, optional): The expansion factor to control the number of intermediate channels. Defaults to 0.5.
            use_shortcut (bool, optional): Whether to use a residual shortcut connection. Defaults to True.
            use_depth_wise_conv (bool, optional): Whether to use depth-wise convolutions instead of standard convolutions. Defaults to False.
            activation (str, optional): The activation function to apply. Defaults to 'swish'.
            name (str, optional): A base name for the layers. Defaults to an empty string.

        Returns:
            Tensor: The output tensor after applying the CSP block.
        """

        # Get the number of input channels
        input_channels = inputs.shape[-1]

        # First convolution block with expansion factor
        nn = self.conv_dw_pw_block(inputs, int(input_channels * expansion), activation=activation, name=name + '1_')

        # Second convolution block with optional depth-wise convolution
        nn = self.conv_dw_pw_block(nn, input_channels, kernel_size=3, strides=1, use_depth_wise_conv=use_depth_wise_conv,
                                   activation=activation, name=name + '2_')

        # Apply shortcut connection if enabled
        if use_shortcut:
            nn = layers.Add()([inputs, nn])

        return nn

    def csp_stack(self, inputs, depth, out_channels=-1, expansion=0.5, use_shortcut=True, use_depth_wise_conv=False,
                  activation='swish', name=''):
        """
        Implements a Cross Stage Partial (CSP) stack, consisting of multiple CSP blocks.

        Parameters:
            inputs (Tensor): The input tensor.
            depth (int): The number of CSP blocks to stack.
            out_channels (int, optional): The number of output channels. If -1, it keeps the same as the input. Defaults to -1.
            expansion (float, optional): The expansion factor to control the number of intermediate channels. Defaults to 0.5.
            use_shortcut (bool, optional): Whether to use residual connections inside CSP blocks. Defaults to True.
            use_depth_wise_conv (bool, optional): Whether to use depth-wise convolutions. Defaults to False.
            activation (str, optional): The activation function to apply. Defaults to 'swish'.
            name (str, optional): A base name for the layers. Defaults to an empty string.

        Returns:
            Tensor: The output tensor after applying the CSP stack.
        """

        # Define the channel axis for TensorFlow (N-H-W-C format)
        channel_axis = -1

        # Keep input channels if not specified
        out_channels = inputs.shape[channel_axis] if out_channels == -1 else out_channels

        # Compute the intermediate number of channels
        hidden_channels = int(out_channels * expansion)

        # Create two separate paths: "short" (shortcut) and "deep" (main feature extraction)
        short = self.conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation,
                                      name=name + 'short_')
        deep = self.conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + 'deep_')

        # Apply multiple CSP blocks sequentially
        for id in range(depth):
            block_name = name + f'block{id + 1}_'
            deep = self.csp_block(deep, 1, use_shortcut=use_shortcut, use_depth_wise_conv=use_depth_wise_conv,
                                  activation=activation, name=block_name)

        # Concatenate the "short" and "deep" paths
        out = tf.concat([deep, short], axis=channel_axis)

        # Apply a final 1x1 convolution to merge features
        out = self.conv_dw_pw_block(out, out_channels, kernel_size=1, activation=activation, name=name + 'output_')

        return out

    def csp_darknet(self, width_mul=1, depth_mul=1, out_features=[-3, -2, -1], use_depth_wise_conv=False,
                   input_shape=(512, 512, 3), activation='swish', model_name=''):
        """
        Constructs a CSP-Darknet backbone, commonly used in object detection (e.g., YOLOX).

        Parameters:
            width_mul (float): Multiplier for the number of channels.
            depth_mul (float): Multiplier for the number of CSP block repetitions.
            out_features (list): Indices of the feature maps to output.
            use_depth_wise_conv (bool): Whether to use depth-wise convolutions.
            input_shape (tuple): Shape of the input image (default: 512x512x3).
            activation (str): Activation function to use (default: 'swish').
            model_name (str): Name of the model.

        Returns:
            Keras Model: CSP-Darknet backbone model.
        """

        # Define base number of channels and depth based on multipliers
        base_channels, base_depth = int(width_mul * 64), max(round(depth_mul * 3), 1)

        # Define model input layer
        inputs = tf.keras.Input(input_shape)

        # Initial feature extraction using a stem block
        nn = self.focus_stem(inputs, base_channels, activation=activation, name='stem_')
        features = [nn]  # Store feature maps for later use

        # Define CSP block configurations for each stage
        depthes = [base_depth, base_depth * 3, base_depth * 3, base_depth]
        channels = [base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        use_spps = [False, False, False, True]
        use_shortcuts = [True, True, True, False]

        # Iterate through each stage to construct the backbone
        for id, (channel, depth, use_spp, use_shortcut) in enumerate(zip(channels, depthes, use_spps, use_shortcuts)):
            stack_name = f'stack{id + 1}_'

            # Apply a convolution block with downs-ampling (stride=2)
            nn = self.conv_dw_pw_block(nn, channel, kernel_size=3, strides=2, use_depth_wise_conv=use_depth_wise_conv,
                                       activation=activation, name=stack_name)

            # Apply Spatial Pyramid Pooling (SPP) at the last stage
            if use_spp:
                nn = self.spatial_pyramid_pooling(nn, activation=activation, name=stack_name + 'spp_')

            # Apply a CSP stack with multiple CSP blocks
            nn = self.csp_stack(nn, depth, use_shortcut=use_shortcut, use_depth_wise_conv=use_depth_wise_conv,
                                activation=activation, name=stack_name)

        # Select specific feature maps for output (useful for detection heads like FPN)
        nn = [features[ii] for ii in out_features]

        # Create the Keras Model
        model = models.Model(inputs, nn, name=model_name)

        return model

    def up_sample_merge(self, inputs, csp_depth, use_depth_wise_conv=False, activation='swish', name=''):
        """
        Performs feature up-sampling and merging for FPN-like structures, commonly used in object detection.

        Parameters:
            inputs (list): A list of two feature maps [low-resolution, high-resolution].
            csp_depth (int): The depth of the CSP stack to process the merged features.
            use_depth_wise_conv (bool): Whether to use depth-wise separable convolutions.
            activation (str): Activation function to use (default: 'swish').
            name (str): Base name for naming layers.

        Returns:
            fpn_out (tensor): The up-sampled feature map before merging.
            nn (tensor): The final merged and processed feature map.
        """

        # TensorFlow format (channels last)
        channel_axis = -1

        # Get the number of channels from the high-resolution feature map
        target_channel = inputs[-1].shape[channel_axis]

        # Align the number of channels in the low-resolution feature map to match the high-resolution one
        fpn_out = self.conv_dw_pw_block(inputs[0], target_channel, activation=activation, name=name + 'fpn_')

        # Resize the low-resolution feature map to match the spatial dimensions of the high-resolution one
        size = tf.shape(inputs[-1])[1:-1]
        inputs[0] = tf.image.resize(fpn_out, size, method='nearest')

        # Concatenate the up-sampled low-resolution feature map with the high-resolution feature map
        nn = tf.concat(inputs, axis=channel_axis)

        # Apply a CSP stack to process the merged feature maps
        nn = self.csp_stack(nn, csp_depth, target_channel, 0.5, False, use_depth_wise_conv,
                            activation=activation, name=name)
        return fpn_out, nn

    def down_sample_merge(self, inputs, csp_depth, use_depth_wise_conv=False, activation='swish', name=''):
        """
        Performs feature down-sampling and merging, commonly used in object detection networks.

        Parameters:
            inputs (list): A list of two feature maps [high-resolution, low-resolution].
            csp_depth (int): The depth of the CSP stack to process the merged features.
            use_depth_wise_conv (bool): Whether to use depth-wise separable convolutions.
            activation (str): Activation function to use (default: 'swish').
            name (str): Base name for naming layers.

        Returns:
            nn (tensor): The final merged and processed feature map.
        """

        # TensorFlow format (channels last)
        channel_axis = -1

        # Downsample the high-resolution feature map to match the spatial size of the low-resolution one
        inputs[0] = self.conv_dw_pw_block(inputs[0], inputs[-1].shape[channel_axis], 3, 2,
                                          use_depth_wise_conv, activation=activation, name=name + 'down_')

        # Concatenate the down-sampled high-resolution feature map with the low-resolution feature map
        nn = tf.concat(inputs, axis=channel_axis)

        # Apply a CSP stack to process the merged feature maps
        nn = self.csp_stack(nn, csp_depth, nn.shape[channel_axis], 0.5, False, use_depth_wise_conv,
                            activation=activation, name=name)
        return nn

    def path_aggregation_fpn(self, features, depth_mul=1, use_depth_wise_conv=False, activation='swish', name=''):
        """
        Implements a Path Aggregation Feature Pyramid Network (PA-FPN) for better feature fusion across scales.

        Parameters:
            features (list): A list containing three feature maps [P3, P4, P5] from different scales.
            depth_mul (float): A multiplier to control the depth of the CSP blocks.
            use_depth_wise_conv (bool): Whether to use depth-wise separable convolutions.
            activation (str): Activation function to use (default: 'swish').
            name (str): Base name for naming layers.

        Returns:
            list: Three feature maps after path aggregation [pan_out2, pan_out1, pan_out0].
        """

        # Determine the depth of CSP blocks based on depth multiplier
        csp_depth = max(round(depth_mul * 3), 1)

        # Unpack input feature maps from different levels
        p3, p4, p5 = features   # P3: small-scale, P4: medium-scale, P5: large-scale feature maps

        # Step 1: Up-sample & Merge P5 with P4 (FPN top-down pathway)
        fpn_out0, f_out0 = self.up_sample_merge([p5, p4], csp_depth, use_depth_wise_conv=use_depth_wise_conv,
                                                activation=activation, name=name + 'c3p4_')

        # Step 2: Up-sample & Merge result with P3 (FPN continues)
        fpn_out1, pan_out2 = self.up_sample_merge([f_out0, p3], csp_depth, use_depth_wise_conv=use_depth_wise_conv,
                                                  activation=activation, name=name + 'c3p3_')

        # Step 3: Down-sample & Merge pan_out2 with fpn_out1 (PAN bottom-up pathway)
        pan_out1 = self.down_sample_merge([pan_out2, fpn_out1], csp_depth, use_depth_wise_conv=use_depth_wise_conv,
                                          activation=activation, name=name + 'c3n3_')

        # Step 4: Down-sample & Merge pan_out1 with fpn_out0 (final feature fusion)
        pan_out0 = self.down_sample_merge([pan_out1, fpn_out0], csp_depth, use_depth_wise_conv=use_depth_wise_conv,
                                          activation=activation, name=name + 'c3n4_')

        # Return the final feature maps after path aggregation
        return [pan_out2, pan_out1, pan_out0]

    def yolo_x_head_single(self, inputs, out_channels, num_classes=80, regression_len=4, num_anchor=1,
                           use_depth_wise_conv=False, use_object_scores=True, activation='swish', name=''):
        """
        Constructs a single YOLO_X detection head for a feature map.

        Parameters:
            inputs (Tensor): Input feature map from the backbone or FPN.
            out_channels (int): Number of output channels for intermediate layers.
            num_classes (int): Number of classes for object detection (default: 80 for COCO dataset).
            regression_len (int): Number of parameters for bounding box regression (default: 4 for [x, y, w, h]).
            num_anchor (int): Number of anchors per spatial location (default: 1 for anchor-free YOLOX).
            use_depth_wise_conv (bool): Whether to use depth-wise separable convolutions.
            use_object_scores (bool): Whether to include object-ness scores in the output.
            activation (str): Activation function to use (default: 'swish').
            name (str): Base name for naming layers.

        Returns:
            Tensor: Concatenated output tensor with regression, object-ness, and classification predictions.
        """

        # Initial stem convolution block to process the input features
        stem = self.conv_dw_pw_block(inputs, out_channels, activation=activation, name=name + 'stem_')

        # Classification branch (cls_convs, cls_preds)
        cls_nn = self.conv_dw_pw_block(stem, out_channels, kernel_size=3, use_depth_wise_conv=use_depth_wise_conv,
                                       activation=activation, name=name + 'cls_1_')
        cls_nn = self.conv_dw_pw_block(cls_nn, out_channels, kernel_size=3, use_depth_wise_conv=use_depth_wise_conv,
                                       activation=activation, name=name + 'cls_2_')

        # Final classification output: predicts class scores for each anchor
        cls_out = layers.Conv2D(num_classes * num_anchor, kernel_size=1, use_bias=False, name=name + 'class_out')(cls_nn)

        # Regression branch (reg_convs, reg_preds) - predicts bounding box coordinates
        reg_nn = self.conv_dw_pw_block(stem, out_channels, kernel_size=3, use_depth_wise_conv=use_depth_wise_conv,
                                       activation=activation, name=name + 'reg_1_')
        reg_nn = self.conv_dw_pw_block(reg_nn, out_channels, kernel_size=3, use_depth_wise_conv=use_depth_wise_conv,
                                       activation=activation, name=name + 'reg_2_')

        # Bounding box regression output: predicts bbox coordinates [x, y, w, h] for each anchor
        reg_out = (layers.Conv2D(regression_len * num_anchor, kernel_size=1, use_bias=False,
                                 name=name + 'regression_out')(cls_nn))

        # Object-ness branch (obj_preds) - predicts confidence score for object presence
        obj_out = layers.Conv2D(1 * num_anchor, kernel_size=1, use_bias=False, name=name + 'object_out')(reg_nn)

        # Concatenate outputs along the last axis: [regression, object-ness, classification]
        return tf.concat([reg_out, obj_out, cls_out], axis=-1)

    def yolo_x_head(self, inputs, width_mul=1.0, num_classes=80, regression_len=4, num_anchor=1,
                    use_depth_wise_conv=False, use_object_scores=True, activation='swish', name=''):
        """
        YOLO_X Head for Multi-Scale Object Detection.

        Parameters:
            inputs (list of Tensors): Feature maps from FPN/PAN at different scales.
            width_mul (float): Scaling factor for the number of channels.
            num_classes (int): Number of object classes (default: 80 for COCO dataset).
            regression_len (int): Number of bounding box regression parameters (default: 4 for [x, y, w, h]).
            num_anchor (int): Number of anchors per spatial location (default: 1 for anchor-free YOLOX).
            use_depth_wise_conv (bool): Whether to use depth-wise separable convolutions.
            use_object_scores (bool): Whether to include object-ness scores in the output.
            activation (str): Activation function to use (default: 'swish').
            name (str): Base name for naming layers.

        Returns:
            list of Tensors: Detection outputs for each feature map.
        """

        # Compute output channels based on width multiplier
        out_channel = int(256 * width_mul)

        # List to store output predictions from each feature level
        outputs = []

        # Loop through each input feature map and apply YOLO_X detection head
        for id, input in enumerate(inputs):
            cur_name = name + f'{id + 1}_'  # Generate unique layer name
            out = self.yolo_x_head_single(input, out_channel, num_classes, regression_len, num_anchor,
                                          use_depth_wise_conv, use_object_scores, activation=activation, name=cur_name)
            outputs.append(out)  # Store the output

        # Return list of output tensors (one per input feature map)
        return outputs

    def st_yolo_x(self):
        """
        Builds the full YOLO_X model with CSPDarkNet as the backbone,
        Path Aggregation FPN for multiscale features, and YOLOX detection head.

        Returns:
            model (tf.keras.Model): A YOLO_X model ready for training/inference.
        """

        # Ensure width multiplier is valid
        width_mul = self.width_mul if self.width_mul > 0 else 1

        # Step 1: Build the CSPDarkNet backbone (feature extractor)
        backbone = self.csp_darknet(width_mul, self.depth_mul, self.features_pick, self.use_depth_wise_conv,
                                    self.input_shape, activation=self.activation, model_name='darknet')

        # Step 2: Extract feature maps from the backbone
        features = backbone.outputs  # Feature maps from different layers
        inputs = backbone.inputs[0]  # Input tensor to the model

        # Step 3: Apply Path Aggregation Feature Pyramid Network (FPN)
        fpn_features = self.path_aggregation_fpn(features, depth_mul=self.depth_mul,
                                                 use_depth_wise_conv=self.use_depth_wise_conv,
                                                 activation=self.activation, name='head_')

        # Step 4: Apply YOLO_X detection head to process multiscale features
        outputs = self.yolo_x_head(fpn_features, width_mul, self.num_classes, self.regression_len, self.num_anchors,
                                   self.use_depth_wise_conv, self.use_object_scores, activation=self.activation,
                                   name='head_')

        # Step 5: Create a Keras Model with the input and output tensors
        model = models.Model(inputs, outputs, name=self.model_name)

        # Return the final YOLOX model
        return model
