import tensorflow as tf


def set_gpu_memory_limit(gigabyte):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=(1024 * gigabyte))]
            )

            logical_gpus = tf.config.list_logical_devices('GPU')

            print(f'{len(gpus)} physical GPUs, {logical_gpus} logical GPUs')
            print(f'[INFO]: Setting upper memory limit to {gigabyte}GBytes on gpu[0]')
        except:
            raise RuntimeError('\nVirtual devices must be set before GPUs have been initialized.')
