from __future__ import print_function


from . import model_dict


def create_model(name, n_cls, dataset='miniImageNet', dropout=0.1, n_trans=16, embd_sz=64):
    """create model by name"""
    print("***********", name)
    if dataset == 'miniImageNet' or dataset == 'tieredImageNet':
        if name.startswith('resnet50'):
            print('use imagenet-style resnet50')
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('resnet') or name.startswith('seresnet'):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls, no_trans=n_trans, embd_size=embd_sz)
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    elif dataset == 'CIFAR-FS' or dataset == 'FC100' or dataset=="toy":
        if name.startswith('resnet') or name.startswith('seresnet'):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=n_cls, no_trans=n_trans, embd_size=embd_sz)
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    elif dataset == 'CUB' :
        if name.startswith('resnet') or name.startswith('seresnet'):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls, no_trans=n_trans, embd_size=embd_sz)
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))        
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    return model


def get_teacher_name(model_path):
    """parse to get teacher model name"""
    segments = model_path.split('/')[-2].split('_')
    if ':' in segments[0]:
        return segments[0].split(':')[-1]
    else:
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]
