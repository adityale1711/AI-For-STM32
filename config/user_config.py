general = {
    'project_name': 'demo_weed_detection',
    'model_type': 'st_yolo_x',
    'model_path': None,
    'logs_dir': 'logs',
    'saved_models_dir': 'saved_models',
    'gpu_memory_limit': None,
    'global_seed': 127
}

operation_modes = {
    0: 'training',
    1: 'evaluation',
    2: 'deployment',
    3: 'quantization',
    4: 'benchmarking'
}

dataset = {
    'name': 'weeds-in-field-v3-resize-416x416-darknet',
    'class_name': ['weed']
}

data_augmentation = {
    'random_periodic_resizing': {}
}

training = {
    'model': {
        'input_shape': (416, 416, 3)
    },
    'resume_training_from': {}
}

postprocessing = {

}

mlflow = {
    'uri': 'http://127.0.0.1:5000'
}
