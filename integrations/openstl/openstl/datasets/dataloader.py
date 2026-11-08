# Copyright (c) CAIRI AI Lab. All rights reserved

def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get('pre_seq_length', 10),
        aft_seq_length=kwargs.get('aft_seq_length', 10),
        in_shape=kwargs.get('in_shape', None),
        distributed=dist,
        use_augment=kwargs.get('use_augment', False),
        use_prefetcher=kwargs.get('use_prefetcher', False),
        drop_last=kwargs.get('drop_last', False),
    )

    if dataname == 'bair':
        from .dataloader_bair import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'human':
        from .dataloader_human import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'kitticaltech':
        from .dataloader_kitticaltech import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'kth' in dataname:  # 'kth', 'kth20', 'kth40'
        from .dataloader_kth import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname in ['mmnist', 'mfmnist', 'mmnist_cifar']:  # 'mmnist', 'mfmnist', 'mmnist_cifar'
        from .dataloader_moving_mnist import load_data
        cfg_dataloader['data_name'] = kwargs.get('data_name', 'mnist')
        cfg_dataloader['validation_file'] = kwargs.get('validation_file', None)
        cfg_dataloader['strict_validation'] = kwargs.get('strict_validation', False)
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'noisymmnist' in dataname:  # 'mmnist - perceptual', 'mmnist - missing', 'mmnist - dynamic'
        from .dataloader_noisy_moving_mnist import load_data
        cfg_dataloader['noise_type'] = kwargs.get('noise_type', 'perceptual')
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'kinetics' in dataname:  # 'kinetics400', 'kinetics600'
        from .dataloader_kinetics import load_data
        cfg_dataloader['data_name'] = kwargs.get('data_name', 'kinetics400')
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'taxibj':
        from .dataloader_taxibj import load_data
        cfg_dataloader['validation_fraction'] = kwargs.get('validation_fraction', 0.1)
        cfg_dataloader['validation_gap'] = kwargs.get('validation_gap', 0)
        cfg_dataloader['memmap_cache'] = kwargs.get('memmap_cache', 'taxibj/memmap_v1')
        cfg_dataloader['strict_memmap'] = kwargs.get('strict_memmap', False)
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'weather' in dataname:  # 'weather', 'weather_t2m', etc.
        from .dataloader_weather import load_data
        data_split_pool = ['5_625', '2_8125', '1_40625']
        data_split = '5_625'
        for k in data_split_pool:
            if dataname.find(k) != -1:
                data_split = k
        return load_data(batch_size, val_batch_size, data_root, num_workers,
                         distributed=dist, data_split=data_split, **kwargs)
    elif 'sevir' in dataname:  #'sevir_vis', 'sevir_ir069', 'sevir_ir107', 'sevir_vil'
        from .dataloader_sevir import load_data
        cfg_dataloader['data_name'] = kwargs.get('data_name', 'sevir')
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
