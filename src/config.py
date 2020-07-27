import yaml
import os

def parse_config(args):
    """
    prepare configs
    """
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    messytable_dir = os.path.realpath(os.path.join(file_dir, '..'))
    config_pathname = os.path.join(messytable_dir,'models',args.config_dir,'train.yaml')

    config = yaml.load(open(config_pathname, 'r'))

    config['messytable_dir'] = messytable_dir
    config['config_dir'] = os.path.join(messytable_dir,'models',args.config_dir)
    config['data_dir'] = os.path.join(messytable_dir, 'data') if 'data_dir' not in config else config['data_dir'] # NOTE: either indicate data_dir or put the data in messytable/data
    config['img_dir'] = os.path.join(config['data_dir'],'images') # TODO refine the name
    config['train_label_pathname'] = os.path.join(config['data_dir'],'labels',config['train_json'])

    config['num_workers'] = config['num_workers'] if 'num_workers' in config else 16
    config['milestones'] = config['milestones'] if 'milestones' in config else [60, 80]
    config['split_samples_in_func'] = config['split_samples_in_func'] if 'split_samples_in_func' in config else True
    config['loss_func'] = config['loss_func'] if 'loss_func' in config else 'ERROR_LOSS_FUNC'
    config['triplet_margin']  = config['triplet_margin'] if 'triplet_margin' in config else 0.3
    config['data_augmentation'] = config['data_augmentation'] if 'data_augmentation' in config else False
    config['cropped_img_size'] = (config['cropped_height'],config['cropped_width'])
    config['original_img_size'] = (config['img_height'],config['img_width'])
    config['scene_ratio'] = config['scene_ratio'] if 'scene_ratio' in config else 1.0
    config['cam_selected_num'] = config['cam_selected_num'] if 'cam_selected_num' in config else 8
    config['triplet_sampling_ratio'] = config['triplet_sampling_ratio'] if 'triplet_sampling_ratio' in config else [0.5,0.3,0.2]
    config['image_pairs_per_batch'] = config['image_pairs_per_batch'] if 'image_pairs_per_batch' in config else 24
    config['triplet_batch_size'] = config['triplet_batch_size'] if 'triplet_batch_size' in config else config['batch_size']
    config['learning_rate'] = float(config['learning_rate'])
    config['zoomout_crop_num'] = 'single_crop' if len(config['zoomout_ratio']) == 1 else 'multi_crops'
    
    # make cam_pairs
    test_cam_pairs = []
    for i in range(1,9):
        for j in range(i+1,10):
            test_cam_pairs.append((str(i),str(j)))
    reversed_cam_pairs = []
    for cam_pair in test_cam_pairs:
        reversed_cam_pairs.append((cam_pair[1],cam_pair[0]))
    config['test_cam_pairs'] = test_cam_pairs
    config['train_cam_pairs'] = test_cam_pairs + reversed_cam_pairs
    config['cam_list'] = [str(i) for i in range(1,10)]

    return config
