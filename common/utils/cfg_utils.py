def get_random_seed(global_seed):
    if global_seed is not None:
        seed = global_seed
        if seed == 'None':
            seed = None
        else:
            seed = int(seed)
    else:
        seed = 0

    return seed
