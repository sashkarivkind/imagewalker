from tensorflow import keras


def enforce_list(x):
    if type(x) is list:
        return x
    else:
        return [x]


def squeeze_list(x):
    if len(x) == 1:
        return x[0]
    else:
        return x


def extract_top_model(source_model, last_bottom_layer_name):
    layers_by_output_tensor_name = {}
    for layer in source_model.layers:
        print(layer.name)
        config = layer.get_config()
        layers_by_output_tensor_name[layer.output.name] = {'name': layer.name,
                                                           'layer_instance': keras.layers.deserialize(
                                                               {'class_name': layer.__class__.__name__,
                                                                'config': config}),
                                                           'input_tensor_names': [uu.name for uu in
                                                                                  enforce_list(layer.input)],
                                                           'output_tensor_instance': None}

    # out_tensor = layer_instance(*input_tensors_names)
    last_bottom_layer = source_model.get_layer(last_bottom_layer_name)
    input_top = keras.layers.Input(shape=last_bottom_layer.output.shape.as_list()[1:],
                                   name='input_top')

    layers_by_output_tensor_name[last_bottom_layer.output.name]['output_tensor_instance'] = input_top

    for ii in range(len(layers_by_output_tensor_name)):
        print('iteration', ii)
        for tensor_name in layers_by_output_tensor_name:
            if layers_by_output_tensor_name[tensor_name]['output_tensor_instance'] is None:
                not_ready_yet = False
                for input_tensor in layers_by_output_tensor_name[tensor_name]['input_tensor_names']:
                    if layers_by_output_tensor_name[input_tensor]['output_tensor_instance'] is None:
                        not_ready_yet = True
                if not not_ready_yet:
                    inputs = [layers_by_output_tensor_name[input_tensor]['output_tensor_instance'] for input_tensor in
                              layers_by_output_tensor_name[tensor_name]['input_tensor_names']]
                    layers_by_output_tensor_name[tensor_name]['output_tensor_instance'] = \
                        layers_by_output_tensor_name[tensor_name]['layer_instance'](squeeze_list(inputs))
                    print('connected layer:', layers_by_output_tensor_name[tensor_name]['name'])

    top_model = keras.models.Model(inputs=input_top,
                                   outputs=layers_by_output_tensor_name[source_model.layers[-1].output.name][
                                       'output_tensor_instance'])

    for layer in top_model.layers:
        if layer.name != 'input_top':
            layer.set_weights(source_model.get_layer(layer.name).get_weights())
            print('set weights for layer {}'.format(layer.name))

    return top_model

def split_model(model, layer):
    top_model = extract_top_model(model,layer)
    bottom_model = keras.models.Model(inputs=model.layers[0].input,
                                            outputs=model.get_layer(layer).output)
    return bottom_model, top_model