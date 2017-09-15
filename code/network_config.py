# TODO edit this file to generate a network configuration
# when satisfied with the network configuration run build_model,
# and a json config file as well as network plots will be generated

def make_config():
    
    # model name is used to save plots of your model, and a configuration json file to restore it
    # using the same model name twice will overwrite previous config files, and plots
    model_name = 'example_model'

    # method used to decrease the spatial resolution of conv layers
    downsample_method = 'strided_convolution'
    #downsample_method = 'max_pooling'
    #downsample_method = 'average_pooling'

    
    # method used to increase the spatial resolution of conv layers
    upsample_method = 'nearest_neighbor'
    #upsample_method = 'transposed_convolution'

    # whether to use, or where to use batch normalization, only settable at the block level
    batch_norm = 1
    # batch_norm = [1, 1, 0, 0, 0, 0]

    # type of convolution layer to use, options are separable, or regular
    convolution_type = 'separable'
    # convolution_type = 'regular'

    # for n_blocks_per_side = 3 there will be 3 downsampling and 3 upsampling blocks for a total of six blocks
    n_blocks_per_side = 3
    
    # the number of layers in each block or alternatively a list of layer sizes
    block_size = 2
    #block_size = [1,2,3,3,2,1]
    
    # number of base filters, or the number of filters for each block
    n_filters = 16
    #n_filters = [16,64,64,128,64,32]

    # allowed image resolutions to try: 256 224, 192, 160, 128, 96, 64, 32 
    image_resolution = 256

    # binary_crossentropy will be used for 'sigmoid', and categorical_crossentropy will be used for 'softmax'
    last_layer_activation = 'sigmoid'
    # activation = 'softmax'

    # learning rate used for the adam optimizer(for example network) 
    learning_rate = 0.001

    ## NOTE do not edit below this line
    config_dict = dict()
    config_dict['downsample_method'] = downsample_method
    config_dict['upsample_method'] = upsample_method
    config_dict['batch_norm'] = batch_norm

    config_dict['convolution_type'] = convolution_type
    config_dict['n_blocks_per_side'] = n_blocks_per_side
    config_dict['n_filters'] = n_filters

    config_dict['block_size'] = block_size
    config_dict['image_resolution'] = image_resolution
    config_dict['last_layer_activation'] = last_layer_activation
    config_dict['n_classes'] = 3
    config_dict['model_name'] = model_name
    config_dict['learning_rate'] = learning_rate
    return config_dict

