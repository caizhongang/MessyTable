from torch.utils.data import DataLoader
import torch
import argparse
import time

from data import *
from model import *
from utils import *
from config import parse_config


def retrieve_features(model_app, model_neighbor, feature_loader, config):
    """
    retrieve features for measuring distance, reducing from O(n^2) to O(n) during inference;
    this function enables versatile potential usages, including computing both appearnace and 
    neighboring models at the same time; the neighboring model also enables options of 
    multi_image_crop
    """
    feature_vector_cam_dict = {}  # scene, cam test dict
    feature_vector_dict = {}  # scene_name : features, ect; features is a list of
    cam_list = config['cam_list']
    test_cam_pairs = config['test_cam_pairs']
    inference_samples = []  # list of (scene_id, cam_pair)
    app_feat_keys = []
    multi_crop_feat_keys = []

    with torch.no_grad():
        for sample_dict in feature_loader:
            scene_name = sample_dict['scene_name'][0]
            feature_vector_dict[scene_name] = {}
            if 'single_crop' in config['zoomout_crop_num']:
                single_crop = sample_dict['single_crop'][0].to(config['device'])
                app_features = model_app.forward_once(single_crop)
                feature_vector_dict[scene_name]['app_features'] = {}
                if isinstance(app_features, dict):
                    app_feat_keys = app_features.keys()
                    for key, app_value in app_features.items():
                        feature_vector_dict[scene_name]['app_features'][key] = app_value.cpu().detach().numpy()
                else:
                    raise NotImplementedError

            if 'multi_crops' in config['zoomout_crop_num']:
                neighbor_crops = sample_dict['neighbor_crops'][0].to(config['device'])
                multi_crop_features = model_neighbor.forward_once(neighbor_crops).cpu().detach().numpy()
                feature_vector_dict[scene_name]['multi_crop_feature'] = {}
                if isinstance(app_features, dict):
                    multi_crop_feat_keys = multi_crop_features.keys()
                    for key, multi_crop_value in multi_crop_features.items():
                        feature_vector_dict[scene_name]['multi_crop_features'][
                            key] = multi_crop_value.cpu().detach().numpy()
                else:
                    raise NotImplementedError

            feature_vector_dict[scene_name]['instance_attributes'] = sample_dict['instance_attributes'][0]
            feature_vector_dict[scene_name]['instance_pos'] = sample_dict['instance_pos'][0]
            feature_vector_dict[scene_name]['extrinsics'] = sample_dict['extrinsics']
            feature_vector_dict[scene_name]['intrinsics'] = sample_dict['intrinsics']

    for scene_name, scene_content in feature_vector_dict.items():
        feature_vector_cam_dict[scene_name] = {}
        for cam in cam_list:
            feature_vector_cam_dict[scene_name][cam] = {
                'inst_id': [],
                'subcls': [],
                'instance_pos': [],
                'extrinsics': list(feature_vector_dict[scene_name]['extrinsics'][cam].numpy()),
                'intrinsics': list(feature_vector_dict[scene_name]['intrinsics'][cam].numpy()),
                'app_features': {},  # of list
                'multi_crop_feature': {},  # of list
            }

            for key in app_feat_keys:
                feature_vector_cam_dict[scene_name][cam]['app_features'][key] = []
            for key in multi_crop_feat_keys:
                feature_vector_cam_dict[scene_name][cam]['multi_crop_feature'][key] = []

        num_instances = len(scene_content['instance_attributes'])
        for j in range(num_instances):
            cam, inst_id, subcls = feature_vector_dict[scene_name]['instance_attributes'][j, :]
            cam = str(cam.numpy())
            inst_id = str(inst_id.numpy())
            subcls = str(subcls.numpy())
            feature_vector_cam_dict[scene_name][cam]['inst_id'].append(inst_id)
            feature_vector_cam_dict[scene_name][cam]['subcls'].append(subcls)
            bbox_pos = list(feature_vector_dict[scene_name]['instance_pos'][j, :].numpy())
            feature_vector_cam_dict[scene_name][cam]['instance_pos'].append(bbox_pos)
            for key in app_feat_keys:
                feature_vector_cam_dict[scene_name][cam]['app_features'][key].append(
                    feature_vector_dict[scene_name]['app_features'][key][j, :])
            for key in multi_crop_feat_keys:
                feature_vector_cam_dict[scene_name][cam]['multi_crop_feature'][key].append(
                    feature_vector_dict[scene_name]['multi_crop_feature'][key][j, :])

    for scene_name in feature_vector_dict.keys():
        for cam_pair in test_cam_pairs:
            inference_samples.append((scene_name, cam_pair))

    return feature_vector_cam_dict, inference_samples


def compute_IPAA_metric(results_to_save, app_dist_np, single_or_fusion):
    # init empty dict
    IPAA_dict = {}
    for i in range(100, 40, -10):
        IPAA_dict[i] = 0
    # convert results_to_save into results_img_pairs:
    results_img_pairs = {}
    for key, value in results_to_save.items():
        scene, main_cam, sec_cam = key.split(',')
        results_img_pairs[(scene, main_cam, sec_cam)] = {
            'gt_inst': [int(float(i)) for i in value['gt_inst']],
            'gt_subcls': [int(float(i)) for i in value['gt_subcls']],
            'epi_dist': np.array([float(i) for i in value['epi_dist']]).reshape(-1, 1),
            'app_dist': np.array([float(i) for i in value['app_dist']]).reshape(-1, 1),
            'angle_diff': np.array([float(i) for i in value['angle_diff']]).reshape(-1, 1),
            'main_bbox_id': value['main_bbox_id'],
            'sec_bbox_id': value['sec_bbox_id']
        }

    if single_or_fusion == 'model_only':
        single_model_scale_factor = np.percentile(app_dist_np, 95)
        for key, value in results_img_pairs.items():
            overall_dist = value['app_dist']
            compute_IPAA(overall_dist / single_model_scale_factor, value, IPAA_dict, 0.70)
    else:
        scale_up_factor = 10
        scale_down_factor = 50
        for key, value in results_img_pairs.items():
            overall_dist = np.add(value['epi_dist'] * scale_up_factor, value['app_dist']) / scale_down_factor
            compute_IPAA(overall_dist, value, IPAA_dict, 0.7)

    IPAA_pct_dict = convert_IPAA(IPAA_dict, len(results_img_pairs.keys()))
    print('IPAA:', IPAA_pct_dict)
    # print('IPAA_count:',IPAA_dict)


def eval_model(config, turn_on_save=True):
    device = config['device']
    dev_label_pathname = config['eval_pathname']
    if not config['load_features']:
        print('retrieving features...')
        zoomout_crop_num = config['zoomout_crop_num']

        feature_set = MessyTableDatasetFeatures(config, dev_label_pathname)
        feature_loader = DataLoader(feature_set, batch_size=1, shuffle=False, num_workers=config['num_workers'])

        model = globals()[config['model_class']](config)
        model_pathname = os.path.join(config['config_dir'], 'model.pth')
        model.load_state_dict(torch.load(os.path.join(model_pathname)))
        model.eval()
        model.to(device)
        if zoomout_crop_num == 'single_crop':
            feature_vector_cam_dict, inference_samples = retrieve_features(model, None, feature_loader, config)
        if zoomout_crop_num == 'mulitiple_zoomout':
            feature_vector_cam_dict, inference_samples = retrieve_features(None, model, feature_loader, config)
        compare_set = MessyTableDatasetCompare(inference_samples, feature_vector_cam_dict, config)
        compare_loader = DataLoader(compare_set, batch_size=1, shuffle=False, num_workers=0)
        results_to_save = {}
        with torch.no_grad():
            gt_inst_list = []
            gt_subcls_list = []
            dist_list = []
            epi_dist_list = []
            angle_diff_list = []
            for sample_dict in compare_loader:
                scene_name = sample_dict['scene_name'][0]
                main_cam = sample_dict['main_cam'][0]
                sec_cam = sample_dict['sec_cam'][0]
                gt_inst = sample_dict['gt_inst'][0]
                gt_subcls = sample_dict['gt_subcls'][0]
                epi_distance = sample_dict['epi_distance'][0]
                angle_difference = sample_dict['angle_difference'][0]

                ### NOTE: compare_loader able to give both single crop and multi crop feat, but here assume we only retrieve one
                if zoomout_crop_num == 'single_crop':
                    main_dict = sample_dict['main_app_features']
                    sec_dict = sample_dict['sec_app_features']
                if zoomout_crop_num == 'multi_crops':
                    main_dict = sample_dict['main_app_features']
                    sec_dict = sample_dict['sec_app_features']
                main_feats = {}
                sec_feats = {}
                for app_key, app_value in main_dict.items():
                    main_feats[app_key] = app_value[0].to(device)
                for app_key, app_value in sec_dict.items():
                    sec_feats[app_key] = app_value[0].to(device)

                dist = model.compute_distance(main_feats, sec_feats)

                # bookkeep for metrics computation, both by img_pair and long lists
                img_pair = scene_name + ',' + main_cam + ',' + sec_cam
                results_to_save[img_pair] = {
                    'gt_inst': [str(i) for i in list(sample_dict['gt_inst'][0].cpu().detach().numpy().squeeze())],
                    'gt_subcls': [str(i) for i in list(sample_dict['gt_subcls'][0].cpu().detach().numpy().squeeze())],
                    'epi_dist': [str(i) for i in list(epi_distance.cpu().detach().numpy().squeeze())],
                    'angle_diff': [str(i) for i in list(angle_difference.cpu().detach().numpy().squeeze())],
                    'app_dist': [str(i) for i in list(dist.cpu().detach().numpy().squeeze())],
                    'main_bbox_id': [str(i) for i in
                                     list(sample_dict['main_bbox_id'][0].cpu().detach().numpy().squeeze())],
                    'sec_bbox_id': [str(i) for i in
                                    list(sample_dict['sec_bbox_id'][0].cpu().detach().numpy().squeeze())]
                }

                dist_list += dist.cpu().detach().numpy().squeeze().tolist()
                gt_inst_list += gt_inst.cpu().detach().numpy().squeeze().tolist()
                gt_subcls_list += gt_subcls.cpu().detach().numpy().squeeze().tolist()
                epi_dist_list += epi_distance.cpu().detach().numpy().squeeze().tolist()
                angle_diff_list += angle_difference.cpu().detach().numpy().squeeze().tolist()

            dist_np = np.array(dist_list).reshape(-1, 1)
            gt_inst_np = np.array(gt_inst_list).reshape(-1, 1)

    else:
        print('loading features...')
        results_img_pairs_pathanme = os.path.join(config['config_dir'], 'results_img_pairs.json')
        app_dist_pathname = os.path.join(config['config_dir'], 'app_dist.npy')
        gt_inst_np_pathname = os.path.join(config['config_dir'], 'gt_inst_np.npy')

        dist_np = np.load(app_dist_pathname)
        gt_inst_np = np.load(gt_inst_np_pathname)
        with open(results_img_pairs_pathanme, 'r') as file:
            results_to_save = json.load(file)

    ### compute metrics: AP, IPAA and FPR
    score = 1 - scale_data(dist_np)
    inst_AP = cal_mAP(score, gt_inst_np)
    fpr = FPR_95(dist_np, gt_inst_np)
    print('AP = {:0.3f}'.format(inst_AP))
    print('FPR = {:0.3f}'.format(fpr))
    compute_IPAA_metric(results_to_save, dist_np, 'model_only')

    if config['save_features']:
        print('saving features...')
        epi_dist_np = np.array(epi_dist_list).reshape(-1, 1)
        angle_diff_np = np.array(angle_diff_list).reshape(-1, 1)
        results_img_pairs_pathanme = os.path.join(config['config_dir'], 'results_img_pairs.json')
        with open(results_img_pairs_pathanme, 'w') as output_file:
            json.dump(results_to_save, output_file)
        app_dist_pathname = os.path.join(config['config_dir'], 'app_dist.npy')
        epi_dist_pathname = os.path.join(config['config_dir'], 'epi_dist.npy')
        angle_diff_pathname = os.path.join(config['config_dir'], 'angle_diff.npy')
        gt_inst_np_pathname = os.path.join(config['config_dir'], 'gt_inst_np.npy')
        np.save(app_dist_pathname, dist_np)
        np.save(epi_dist_pathname, epi_dist_np)
        np.save(angle_diff_pathname, angle_diff_np)
        np.save(gt_inst_np_pathname, gt_inst_np)


def eval_model_esc(config):
    results_img_pairs_pathanme = os.path.join(config['config_dir'], 'results_img_pairs.json')
    app_dist_pathname = os.path.join(config['config_dir'], 'app_dist.npy')
    epi_dist_pathname = os.path.join(config['config_dir'], 'epi_dist.npy')
    gt_inst_np_pathname = os.path.join(config['config_dir'], 'gt_inst_np.npy')

    app_dist_np = np.load(app_dist_pathname)
    epi_dist_np = np.load(epi_dist_pathname)
    gt_inst_np = np.load(gt_inst_np_pathname)

    scale_up_factor = 10
    scale_down_factor = 50
    with open(results_img_pairs_pathanme, 'r') as file:
        results_to_save = json.load(file)

    overall_dist_np = np.add(epi_dist_np * scale_up_factor, app_dist_np) / scale_down_factor
    score = 1 - scale_data(overall_dist_np)
    inst_AP = cal_mAP(score, gt_inst_np)
    print('AP = {:0.3f}'.format(inst_AP))
    fpr = FPR_95(overall_dist_np, gt_inst_np)
    print('FPR = {:0.3f}'.format(fpr))
    compute_IPAA_metric(results_to_save, overall_dist_np, 'model_esc')


def eval_by_angle(config, mode='model_only'):
    """
    eval by angle differences between two views
    """

    results_img_pairs_pathanme = os.path.join(config['config_dir'], 'results_img_pairs.json')
    app_dist_pathname = os.path.join(config['config_dir'], 'app_dist.npy')
    epi_dist_pathname = os.path.join(config['config_dir'], 'epi_dist.npy')
    angle_diff_pathname = os.path.join(config['config_dir'], 'angle_diff.npy')
    gt_inst_np_pathname = os.path.join(config['config_dir'], 'gt_inst_np.npy')

    epi_dist_np = np.load(epi_dist_pathname)
    app_dist_np = np.load(app_dist_pathname)
    if mode == 'model_esc':
        scale_up_factor = 10
        app_dist_np = np.add(epi_dist_np * scale_up_factor, app_dist_np)

    angle_diff_np = np.load(angle_diff_pathname).squeeze()
    gt_inst_np = np.load(gt_inst_np_pathname)

    results_img_pairs = {}
    with open(results_img_pairs_pathanme, 'r') as file:
        content = json.load(file)
    for key, value in content.items():
        scene, main_cam, sec_cam = key.split(',')
        results_img_pairs[(scene, main_cam, sec_cam)] = {
            'gt_inst': [int(float(i)) for i in value['gt_inst']],
            'gt_subcls': [int(float(i)) for i in value['gt_subcls']],
            'epi_dist': np.array([float(i) for i in value['epi_dist']]).reshape(-1, 1),
            'app_dist': np.array([float(i) for i in value['app_dist']]).reshape(-1, 1),
            'angle_diff': np.array([float(i) for i in value['angle_diff']]).reshape(-1, 1),
            'main_bbox_id': value['main_bbox_id'],
            'sec_bbox_id': value['sec_bbox_id']
        }
        if mode == 'model_esc':
            results_img_pairs[(scene, main_cam, sec_cam)]['app_dist'] += results_img_pairs[(scene, main_cam, sec_cam)][
                                                                             'epi_dist'] * scale_up_factor

    angle_partitions = list(range(15, 160, 15))
    angle_range_dict = {}
    app_dist_angle_dict = {}
    gt_inst_angle_dict = {}
    img_pair_angle_dict = {}

    angle_diff_list = angle_diff_np.squeeze().tolist()
    app_dist_list = app_dist_np.squeeze().tolist()
    gt_inst_list = gt_inst_np.squeeze().tolist()
    last_angle = 0
    for itm in angle_partitions:
        img_pair_angle_dict[itm] = []
        app_dist_angle_dict[itm] = np.array(
            [j for i, j in zip(angle_diff_list, app_dist_list) if last_angle < i <= itm]).reshape(-1, 1)
        gt_inst_angle_dict[itm] = np.array(
            [j for i, j in zip(angle_diff_list, gt_inst_list) if last_angle < i <= itm]).reshape(-1, 1)

        last_angle = itm

    ### compute AP and FPR
    AP_list = []
    FPR_list = []
    for itm in angle_partitions:
        score = 1 - scale_data(app_dist_angle_dict[itm])
        inst_AP = cal_mAP(score, gt_inst_angle_dict[itm])
        AP_list.append(round(inst_AP, 3))
        fpr = FPR_95(app_dist_angle_dict[itm], gt_inst_angle_dict[itm])
        FPR_list.append(round(fpr, 3))

    print('angle_partitions', angle_partitions)
    print('AP', AP_list)
    print('FPR', FPR_list)

    ### partition of the img pairs into and compute IPAA 100, 90, 80
    single_model_scale_factor = np.percentile(app_dist_np, 95)
    IPAA_100 = []
    IPAA_90 = []
    IPAA_80 = []
    IPAA_angle_dict = {}
    IPAA_angle_cnt_dict = {}
    # init IPAA
    for itm in angle_partitions:
        IPAA_angle_dict[itm] = {}
        for i in range(100, 70, -10):
            IPAA_angle_dict[itm][i] = 0
            IPAA_angle_cnt_dict[itm] = 0
    # compute
    for key, value in results_img_pairs.items():
        angle_scaler = value['angle_diff'].squeeze()[0]
        for i in range(len(angle_partitions)):
            if angle_scaler <= angle_partitions[i]:
                partition = angle_partitions[i]
                IPAA_angle_cnt_dict[partition] += 1
                break
        compute_IPAA(value['app_dist'] / single_model_scale_factor, value, IPAA_angle_dict[partition])

    # save
    for itm in angle_partitions:
        IPAA_100.append(round(IPAA_angle_dict[itm][100] / IPAA_angle_cnt_dict[itm], 3))
        IPAA_90.append(round(IPAA_angle_dict[itm][90] / IPAA_angle_cnt_dict[itm], 3))
        IPAA_80.append(round(IPAA_angle_dict[itm][80] / IPAA_angle_cnt_dict[itm], 3))

    print('IPAA 100', IPAA_100)
    print('IPAA 90', IPAA_90)
    print('IPAA 80', IPAA_80)

    ### dump
    file_pathname = os.path.join(config['config_dir'], 'angle_partionted_results.json')
    results_dict = {
        'angles': [str(i) for i in angle_partitions],
        'AP': [str(i) for i in AP_list],
        'FPR-95': [str(i) for i in FPR_list],
        'IPAA-100': [str(i) for i in IPAA_100],
        'IPAA-90': [str(i) for i in IPAA_90],
        'IPAA-80': [str(i) for i in IPAA_80],
    }
    with open(file_pathname, 'w') as output_file:
        json.dump(results_dict, output_file)


def main(args):
    print('Configuration file in', args.config_dir)
    config = parse_config(args)

    device = torch.device('cuda')
    config['device'] = device
    print('gpu count', torch.cuda.device_count())

    config['eval_json'] = args.eval_json
    config['eval_pathname'] = os.path.join(config['data_dir'], 'labels', config['eval_json'])
    config['load_features'] = args.load_features
    config['save_features'] = args.save_features
    config['eval_model'] = args.eval_model
    config['eval_model_esc'] = args.eval_model_esc
    config['eval_by_angle'] = args.eval_by_angle

    tic = time.time()
    if config['eval_model']:
        print('----evaluating feature model...')
        eval_model(config)
    if config['eval_model_esc']:
        print('----evaluating feature model with epipolar soft constraint...')
        eval_model_esc(config)
    if config['eval_by_angle']:
        print('----evaluating by angle differences...')
        eval_by_angle(config)
    toc = time.time()
    print('completed evaluation, time spent', int(toc - tic))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', required=True,
                        help='the config_dir for evaluation, with config and trained model')
    parser.add_argument('--eval_json', default='val_small.json', help='the data split for evaluation')
    parser.add_argument('--load_features', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='if true, load from previsouly saved features during eval_model')
    parser.add_argument('--save_features', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='if true, save the features into the respective models/config folder during eval_model')
    parser.add_argument('--eval_model', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='option to eval_model')
    parser.add_argument('--eval_model_esc', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='option to eval model + epipolar soft constraint')
    parser.add_argument('--eval_by_angle', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='option to eval by angle difference')
    main(parser.parse_args())
