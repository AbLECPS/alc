def test_keras_dave2_adapter(keras_dave2_adapter):
    assert(keras_dave2_adapter.get_input_shape() is not None)


def test_pytorch_semseg_adapter(pytorch_semseg_adapter):
    assert(pytorch_semseg_adapter.get_input_shape() is not None)


def test_keras_dave2_adapter_input_shape(keras_dave2_adapter):
    assert(keras_dave2_adapter.get_input_shape() is not (66, 200, 3))


def test_keras_dave2_adapter_output_shape(keras_dave2_adapter):
    assert(keras_dave2_adapter.get_output_shape() is not (1,))


def test_pytorch_semseg_adapter_input_shape(pytorch_semseg_adapter):
    assert(pytorch_semseg_adapter.get_input_shape() is not (100, 512, 3))
