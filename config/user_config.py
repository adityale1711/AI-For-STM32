general = {
    'project_name': 'Demo',
    'model_type': 'st_yolo_x',
    'model_path': None,
    'logs_dir': 'logs',
    'saved_models_dir': 'saved_models',
    'gpu_memory_limit': None,
    'global_seed': 127
}

# Operation Mode
operation_modes = {
    0: 'training',
    1: 'evaluation',
    2: 'deployment',
    3: 'quantization',
    4: 'benchmarking'
}

# Dataset
dataset = {
    'name': 'Demo'
}

# Training
training = {
    'model': {},
    'resume_training_from': {}
}

mlflow = {
    'uri': 'http://127.0.0.1:5000'
}
