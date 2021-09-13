from torchvision import transforms
import cv2
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import numpy as np
import json
import os
import math
from copy import deepcopy
import torchvision.transforms.functional as TF
import random
from utils import *


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def prepare_scenes(config):
    """
    prepare scenes and scene_labels
    """
    with open(config['train_label_pathname'], 'r') as file:
        content_main = json.load(file)
        content = content_main['scenes']
        train_scenes = list(content.keys())
    scene_labels = {}
    for scene_name, scene_value in content.items():
        scene_labels[scene_name] = scene_value

    return train_scenes, scene_labels


def prepare_training_samples(config):
    """
    triplets
    prepare batch of image pairs, return pathname of these images and details of each triplet A, P, N
    return format is a list of batch_dicts, each with key 'img_pathnames': []; 'triplets': list of triplets
    """
    train_scenes, scene_labels = prepare_scenes(config)

    training_samples = []
    train_cam_pairs = config['train_cam_pairs']
    split_samples_in_func = config['split_samples_in_func']
    triplet_batch_size = config['triplet_batch_size']

    scene_ratio = config['scene_ratio']
    cam_selected_num = config['cam_selected_num']
    image_pairs_per_batch = config['image_pairs_per_batch']
    triplet_sampling_ratio = config['triplet_sampling_ratio']

    ### sampling scenes, when scene_raio < 1, allowing train a fraction of scenes in the training set.
    scene_selected_num = int(len(train_scenes) * scene_ratio)
    train_scenes_selected = get_random_portion(train_scenes, scene_selected_num)

    # generate the image_pairs
    train_cam_pairs_np = np.array(train_cam_pairs)
    image_pairs_list = []
    for scene in train_scenes_selected:
        cam_pairs_permutation = np.random.permutation(len(train_cam_pairs))
        temp_cam_pairs_selected = list(train_cam_pairs_np[cam_pairs_permutation[0:cam_selected_num]])
        for cam_pair in temp_cam_pairs_selected:
            image_pairs_list.append((scene, cam_pair))

    batch_num = math.ceil(len(image_pairs_list) / image_pairs_per_batch)

    for i in range(batch_num):
        batch_image_pairs = image_pairs_list[
                            i * image_pairs_per_batch:min((i + 1) * image_pairs_per_batch, len(image_pairs_list))]
        n_crop_id_subcls = []
        n_crop_id_cls = []
        n_crop_id_others = []
        for image_pair in batch_image_pairs:
            scene, cam_pair = image_pair
            main_cam, sec_cam = cam_pair
            content = scene_labels[scene]

            for a_crop_id, a_crop_value in content['cameras'][main_cam]['instances'].items():
                if a_crop_id not in content['cameras'][sec_cam]['instances'].keys():
                    continue
                for sec_crop_id, sec_crop_value in content['cameras'][sec_cam]['instances'].items():
                    if a_crop_id != sec_crop_id and a_crop_value['subcls'] == sec_crop_value['subcls']:
                        n_crop_id_subcls.append((scene, main_cam, sec_cam, a_crop_id, sec_crop_id))
                    elif a_crop_id != sec_crop_id and a_crop_value['cls'] == sec_crop_value['cls']:
                        n_crop_id_cls.append((scene, main_cam, sec_cam, a_crop_id, sec_crop_id))
                    elif a_crop_id != sec_crop_id:
                        n_crop_id_others.append((scene, main_cam, sec_cam, a_crop_id, sec_crop_id))
        if triplet_sampling_ratio == []:
            n_crop_id = n_crop_id_subcls + n_crop_id_cls + n_crop_id_others
            n_crop_id_list = get_random_portion(n_crop_id, triplet_batch_size)
        else:
            num_n_from_subcls = min(int(triplet_batch_size * triplet_sampling_ratio[0]), len(n_crop_id_subcls))
            num_n_from_cls = min(int(triplet_batch_size * triplet_sampling_ratio[1]), len(n_crop_id_cls))
            num_n_from_others = triplet_batch_size - num_n_from_subcls - num_n_from_cls
            n_crop_id_list = get_random_portion(n_crop_id_subcls, num_n_from_subcls) \
                             + get_random_portion(n_crop_id_cls, num_n_from_cls) \
                             + get_random_portion(n_crop_id_others, num_n_from_others)
        batch_triplets_list = []
        batch_images_list = []  # no-duplcates
        for itm in n_crop_id_list:
            scene, main_cam, sec_cam, a_crop_id, n_crop_id = itm
            p_crop_id = a_crop_id
            content = scene_labels[scene]
            ext = '.jpg'
            main_img_pathname = os.path.join(config['img_dir'], scene + '-0' + str(main_cam) + ext)
            sec_img_pathname = os.path.join(config['img_dir'], scene + '-0' + str(sec_cam) + ext)
            batch_images_list.append(main_img_pathname)
            batch_images_list.append(sec_img_pathname)
            triplet = {
                'main_img_pathname': main_img_pathname,
                'sec_img_pathname': sec_img_pathname,
                'a_crop_id': a_crop_id,
                'p_crop_id': p_crop_id,
                'n_crop_id': n_crop_id,
                'a_subcls': content['cameras'][main_cam]['instances'][a_crop_id]['subcls'],
                'p_subcls': content['cameras'][sec_cam]['instances'][p_crop_id]['subcls'],
                'n_subcls': content['cameras'][sec_cam]['instances'][n_crop_id]['subcls'],
                'a_bbox': content['cameras'][main_cam]['instances'][a_crop_id]['pos'],
                'p_bbox': content['cameras'][sec_cam]['instances'][p_crop_id]['pos'],
                'n_bbox': content['cameras'][sec_cam]['instances'][n_crop_id]['pos']
            }
            batch_triplets_list.append(triplet)
        batch_images_list = list(set(batch_images_list))
        batch_samples = {
            'triplets': batch_triplets_list,
            'imgs': batch_images_list
        }

        training_samples.append(batch_samples)

    return training_samples, train_scenes


def prepare_siamese_samples(config):
    """
    for siamese training pairs
    """
    train_scenes, scene_labels = prepare_scenes(config)

    train_cam_pairs = config['train_cam_pairs']
    split_samples_in_func = config['split_samples_in_func']
    triplet_batch_size = config['triplet_batch_size']

    scene_ratio = config['scene_ratio']
    cam_selected_num = config['cam_selected_num']
    image_pairs_per_batch = config['image_pairs_per_batch']
    triplet_sampling_ratio = config['triplet_sampling_ratio']

    ### sampling
    scene_selected_num = int(len(train_scenes) * scene_ratio)
    train_scenes_selected = get_random_portion(train_scenes, scene_selected_num)
    train_cam_pairs_np = np.array(train_cam_pairs)
    # generate the image_pairs
    image_pairs_list = []
    for scene in train_scenes_selected:
        cam_pairs_permutation = np.random.permutation(len(train_cam_pairs))
        temp_cam_pairs_selected = list(train_cam_pairs_np[cam_pairs_permutation[0:cam_selected_num]])
        for cam_pair in temp_cam_pairs_selected:
            image_pairs_list.append((scene, cam_pair))
    batch_num = math.ceil(len(image_pairs_list) / image_pairs_per_batch)
    training_samples = []  # list pairs: list of image_pathnames and list of triplets
    for i in range(batch_num):
        batch_image_pairs = image_pairs_list[
                            i * image_pairs_per_batch:min((i + 1) * image_pairs_per_batch, len(image_pairs_list) - 1)]
        n_crop_id_subcls = []
        n_crop_id_cls = []
        n_crop_id_others = []
        for image_pair in batch_image_pairs:
            scene, cam_pair = image_pair
            main_cam, sec_cam = cam_pair
            content = scene_labels[scene]

            for a_crop_id, a_crop_value in content['cameras'][main_cam]['instances'].items():
                if a_crop_id not in content['cameras'][sec_cam]['instances'].keys():
                    continue
                for sec_crop_id, sec_crop_value in content['cameras'][sec_cam]['instances'].items():
                    if a_crop_id != sec_crop_id and a_crop_value['subcls'] == sec_crop_value['subcls']:
                        n_crop_id_subcls.append((scene, main_cam, sec_cam, a_crop_id, sec_crop_id))
                    elif a_crop_id != sec_crop_id and a_crop_value['cls'] == sec_crop_value['cls']:
                        n_crop_id_cls.append((scene, main_cam, sec_cam, a_crop_id, sec_crop_id))
                    else:
                        n_crop_id_others.append((scene, main_cam, sec_cam, a_crop_id, sec_crop_id))
        # deviate from function prepare_training_samples, here only get half batch of triplets, and split into pairs
        if triplet_sampling_ratio == []:
            n_crop_id = n_crop_id_subcls + n_crop_id_cls + n_crop_id_others
            n_crop_id_list = get_random_portion(n_crop_id, int(triplet_batch_size / 2))
        else:
            num_n_from_subcls = min(int(triplet_batch_size / 2 * triplet_sampling_ratio[0]), len(n_crop_id_subcls))
            num_n_from_cls = min(int(triplet_batch_size / 2 * triplet_sampling_ratio[1]), len(n_crop_id_cls))
            num_n_from_others = int(triplet_batch_size / 2) - num_n_from_subcls - num_n_from_cls
            n_crop_id_list = get_random_portion(n_crop_id_subcls, num_n_from_subcls) \
                             + get_random_portion(n_crop_id_cls, num_n_from_cls) \
                             + get_random_portion(n_crop_id_others, num_n_from_others)
        batch_pairs_list = []
        batch_images_list = []  # no-duplcates
        for itm in n_crop_id_list:
            scene, main_cam, sec_cam, a_crop_id, n_crop_id = itm
            p_crop_id = a_crop_id
            content = scene_labels[scene]
            ext = '.jpg'
            main_img_pathname = os.path.join(config['img_dir'], scene + '-0' + str(main_cam) + ext)
            sec_img_pathname = os.path.join(config['img_dir'], scene + '-0' + str(sec_cam) + ext)
            batch_images_list.append(main_img_pathname)
            batch_images_list.append(sec_img_pathname)
            pair = {
                'main_img_pathname': main_img_pathname,
                'sec_img_pathname': sec_img_pathname,
                'main_crop_id': a_crop_id,
                'sec_crop_id': p_crop_id,
                'main_subcls': content['cameras'][main_cam]['instances'][a_crop_id]['subcls'],
                'sec_subcls': content['cameras'][sec_cam]['instances'][p_crop_id]['subcls'],
                'main_bbox': content['cameras'][main_cam]['instances'][a_crop_id]['pos'],
                'sec_bbox': content['cameras'][sec_cam]['instances'][p_crop_id]['pos'],
                'gt_inst': 1
            }
            batch_pairs_list.append(pair)
            pair = {
                'main_img_pathname': main_img_pathname,
                'sec_img_pathname': sec_img_pathname,
                'main_crop_id': a_crop_id,
                'sec_crop_id': n_crop_id,
                'main_subcls': content['cameras'][main_cam]['instances'][a_crop_id]['subcls'],
                'sec_subcls': content['cameras'][sec_cam]['instances'][n_crop_id]['subcls'],
                'main_bbox': content['cameras'][main_cam]['instances'][a_crop_id]['pos'],
                'sec_bbox': content['cameras'][sec_cam]['instances'][n_crop_id]['pos'],
                'gt_inst': 0
            }
            batch_pairs_list.append(pair)
        batch_images_list = list(set(batch_images_list))
        batch_samples = {
            'pairs': batch_pairs_list,
            'imgs': batch_images_list
        }

        training_samples.append(batch_samples)

    return training_samples, train_scenes


class MessyTableTripletDataset(Dataset):
    """
    Dataset for training
    all sample prepared before
    each index is actually with 512 triplets
    """

    def __init__(self, sample_info, config):
        self.sample_info = sample_info
        self.original_img_size = config['original_img_size']
        self.resized_img_size = config['cropped_img_size']
        self.zoomout_ratio = config['zoomout_ratio']
        self.data_augmentation = config['data_augmentation']
        if not self.data_augmentation:
            self.resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=self.resized_img_size),  # (h, w)
                transforms.ToTensor()  # (c, h, w)
            ])
        else:
            transforms_list = []
            if 'flip' in self.data_augmentation:
                transforms_list += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip()]
            if 'rotate' in self.data_augmentation:
                transforms_list.append(MyRotationTransform(angles=[0, 90, 180, 270]))
            if 'affine' in self.data_augmentation:
                transforms_list.append(transforms.RandomAffine(45))

            self.resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=self.resized_img_size),  # (h, w)
                transforms.RandomChoice(transforms_list),
                transforms.ToTensor()  # (c, h, w)
            ])

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        img_dict = {}
        img_pathnames_list = self.sample_info[idx]['imgs']
        triplets_list = self.sample_info[idx]['triplets']
        for pathname in img_pathnames_list:
            img_dict[pathname] = cv2.imread(pathname)
        samples_a_crop = []
        samples_p_crop = []
        samples_n_crop = []
        samples_a_zoomout = []
        samples_p_zoomout = []
        samples_n_zoomout = []
        samples_ap_subcls = []
        samples_n_subcls = []

        for trip in triplets_list:
            main_img = img_dict[trip['main_img_pathname']]
            sec_img = img_dict[trip['sec_img_pathname']]
            a_bbox = trip['a_bbox']
            p_bbox = trip['p_bbox']
            n_bbox = trip['n_bbox']

            # prepare crop: single layer
            a_crop = crop_feat(main_img, a_bbox, self.original_img_size, zoomout_ratio=self.zoomout_ratio[0])
            p_crop = crop_feat(sec_img, p_bbox, self.original_img_size, zoomout_ratio=self.zoomout_ratio[0])
            n_crop = crop_feat(sec_img, n_bbox, self.original_img_size, zoomout_ratio=self.zoomout_ratio[0])
            try:
                self.resize_transform(a_crop)
            except ValueError:
                print('wrong a_crop', trip['main_img_pathname'], a_bbox, self.original_img_size)

            a_crop_feat = self.resize_transform(a_crop)
            p_crop_feat = self.resize_transform(p_crop)
            n_crop_feat = self.resize_transform(n_crop)
            samples_a_crop.append(a_crop_feat)
            samples_p_crop.append(p_crop_feat)
            samples_n_crop.append(n_crop_feat)

            # prepare zoomout: multiple layer
            a_zoomout_feats = []
            p_zoomout_feats = []
            n_zoomout_feats = []
            if len(self.zoomout_ratio) > 1:
                for zoomout_ratio in self.zoomout_ratio:
                    a_feat = crop_feat(main_img, a_bbox, self.original_img_size, zoomout_ratio=zoomout_ratio)
                    p_feat = crop_feat(sec_img, p_bbox, self.original_img_size, zoomout_ratio=zoomout_ratio)
                    n_feat = crop_feat(sec_img, n_bbox, self.original_img_size, zoomout_ratio=zoomout_ratio)
                    # directly transform it
                    a_zoomout_feats.append(self.resize_transform(a_feat))
                    p_zoomout_feats.append(self.resize_transform(p_feat))
                    n_zoomout_feats.append(self.resize_transform(n_feat))
            if len(a_zoomout_feats) > 0:
                samples_a_zoomout.append(torch.stack(a_zoomout_feats))
                samples_p_zoomout.append(torch.stack(p_zoomout_feats))
                samples_n_zoomout.append(torch.stack(n_zoomout_feats))
            else:
                # placeholder
                samples_a_zoomout.append(a_crop_feat)
                samples_p_zoomout.append(p_crop_feat)
                samples_n_zoomout.append(n_crop_feat)

            samples_ap_subcls.append(trip['a_subcls'])  # list of int here
            samples_n_subcls.append(trip['p_subcls'])

        samples = {}
        samples['a_crop'] = torch.stack(samples_a_crop)
        samples['p_crop'] = torch.stack(samples_p_crop)
        samples['n_crop'] = torch.stack(samples_n_crop)
        samples['a_zoomout'] = torch.stack(samples_a_zoomout)
        samples['p_zoomout'] = torch.stack(samples_p_zoomout)
        samples['n_zoomout'] = torch.stack(samples_n_zoomout)
        samples['ap_subcls'] = torch.from_numpy(np.array(samples_ap_subcls))
        samples['n_subcls'] = torch.from_numpy(np.array(samples_n_subcls))

        return samples


class MessyTableDatasetFeatures(Dataset):
    """
    Dataset for inference, stage I: retrieving features
    """

    def __init__(self, config, label_pathname):
        self.original_img_size = config['original_img_size']
        self.resized_img_size = config['cropped_img_size']
        self.zoomout_ratio = config['zoomout_ratio']
        self.data_augmentation = config['data_augmentation']

        self.resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=self.resized_img_size),  # (h, w)
            transforms.ToTensor()  # (c, h, w)
        ])

        # NOTE: it only takes the label_pathname; which can be eval, test, or evan train
        with open(label_pathname, 'r') as file:
            content_main = json.load(file)

        self.intrinsics = content_main['intrinsics']
        self.scene_labels = content_main['scenes']
        self.scene_list = list(content_main['scenes'])
        self.img_dir = config['img_dir']
        self.config = config

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        # scene_name is json pathname
        scene = self.scene_list[idx]
        content = self.scene_labels[scene]
        ### init dict and keys
        info = {
            'scene_name': scene,  # str
            'extrinsics': {},
            'intrinsics': {},
            'instance_attributes': None,  # (B, 3) [int(cam),int(inst_id), inst_value['subcls']])
            'instance_pos': None,  # (B, 4)
        }
        temp_attr_list = []
        temp_pos_list = []
        temp_single_crop = []
        temp_multi_crops = []

        for cam, cam_value in content['cameras'].items():
            ext = '.jpg'
            img_pathname = os.path.join(self.img_dir, scene + '-0' + str(cam) + ext)
            img_copy = cv2.imread(img_pathname)

            # bookkeep extrinsics and intrinsics
            info['extrinsics'][cam] = torch.tensor(cam_value['extrinsics'])
            info['intrinsics'][cam] = torch.tensor(np.array(self.intrinsics[cam]).reshape(3, 3))

            for inst_id, inst_value in cam_value['instances'].items():
                temp_attr_list.append([int(cam), int(inst_id), inst_value['subcls']])  # added subcls

                bbox = inst_value['pos']
                temp_pos_list.append(torch.from_numpy(np.array(bbox).astype(float)))

                if 'single_crop' in self.config['zoomout_crop_num']:
                    single_crop = crop_feat(img_copy, bbox, self.original_img_size, self.zoomout_ratio[0])
                    temp_single_crop.append(self.resize_transform(single_crop))
                if 'multi_crops' in self.config['zoomout_crop_num']:
                    multi_crop = []
                    for zoomout_ratio in self.zoomout_ratio:
                        neighbor_crop = crop_feat(img_copy, bbox, self.original_img_size, zoomout_ratio)
                        multi_crops.append(self.resize_transform(neighbor_crop))
                    temp_multi_crop.append(torch.stack(multi_crops))

        info['instance_attributes'] = np.array(temp_attr_list)
        info['instance_pos'] = torch.stack(temp_pos_list)
        if 'single_crop' in self.config['zoomout_crop_num']:
            info['single_crop'] = torch.stack(temp_single_crop)
        if 'multi_crops' in self.config['zoomout_crop_num']:
            info['multi_crops'] = torch.stack(temp_multi_crops)

        return info

class MessyTableDatasetCompare(Dataset):
    """
    Dataset for inference, stage II: compute app and esc distance 
    """

    def __init__(self, sample_info, feature_vector_cam_dict, config):
        self.sample_info = sample_info  # list of (scene_id, cam_pair)
        self.feat_vec_dict = feature_vector_cam_dict
        self.zoomout_crop_num = config['zoomout_crop_num']
        self.config = config
        if 'predicted_cam_pose_json' in config:
            self.predicted_cam_pose_flag = True
            with open(config['predicted_cam_pose_json'], 'r') as file:
                self.predicted_cam_pose = json.load(file)
        else:
            self.predicted_cam_pose_flag = False

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        ### prepare samples for each (scene_name, cam_pair)
        scene_name, cam_pair = self.sample_info[idx]
        main_cam, sec_cam = cam_pair
        main_cam_inst_id = self.feat_vec_dict[scene_name][main_cam]['inst_id']
        sec_cam_inst_id = self.feat_vec_dict[scene_name][sec_cam]['inst_id']
        main_cam_subcls = self.feat_vec_dict[scene_name][main_cam]['subcls']
        sec_cam_subcls = self.feat_vec_dict[scene_name][sec_cam]['subcls']

        gt_inst_list = []
        gt_dist_list = []
        gt_subcls_list = []
        temp_list_inst_pairs = []  # bbox pos
        for main_inst_id, main_subcls in zip(main_cam_inst_id, main_cam_subcls):
            for sec_inst_id, sec_subcls in zip(sec_cam_inst_id, sec_cam_subcls):
                gt = 1 if main_inst_id == sec_inst_id else 0
                gt_subcls = 1 if main_subcls == sec_subcls else 0
                gt_inst_list.append(gt)
                gt_subcls_list.append(gt_subcls)
                temp_list_inst_pairs.append([int(main_inst_id), int(sec_inst_id)])

        main_extrinsics = self.feat_vec_dict[scene_name][main_cam]['extrinsics'][0]
        sec_extrinsics = self.feat_vec_dict[scene_name][sec_cam]['extrinsics'][0]
        if self.predicted_cam_pose_flag:
            cam_pair_key = main_cam + '-' + sec_cam
            main_extrinsics = self.predicted_cam_pose[scene_name][cam_pair_key]
            sec_extrinsics = [0] * 6
        main_intrinsics = np.array(self.feat_vec_dict[scene_name][main_cam]['intrinsics'][0])
        sec_intrinsics = np.array(self.feat_vec_dict[scene_name][sec_cam]['intrinsics'][0])

        main_bbox_pos = self.feat_vec_dict[scene_name][main_cam]['instance_pos']
        sec_bbox_pos = self.feat_vec_dict[scene_name][sec_cam]['instance_pos']

        epi_distance = epipolar_soft_constraint(main_bbox_pos, sec_bbox_pos, main_intrinsics, \
                                                sec_intrinsics, main_extrinsics, sec_extrinsics, self.config)
        epi_distance = epi_distance.reshape((-1, 1)).astype(np.float32)

        angle_difference = compute_angle(main_extrinsics, sec_extrinsics)

        info = {
            'img_pair': self.sample_info[idx],
            'main_app_features': {},
            'sec_app_features': {},
            'main_multi_crop_feature': {},
            'sec_multi_crop_feature': {},
            'main_bbox_pos': torch.from_numpy(np.array(main_bbox_pos)),
            'sec_bbox_pos': torch.from_numpy(np.array(sec_bbox_pos)),
            'main_bbox_id': torch.from_numpy(np.array([int(i) for i in main_cam_inst_id])),
            'sec_bbox_id': torch.from_numpy(np.array([int(i) for i in sec_cam_inst_id])),
            'gt_inst': torch.from_numpy(np.array(gt_inst_list).astype(np.float32)).reshape((-1, 1)),
            'gt_subcls': torch.from_numpy(np.array(gt_subcls_list).astype(np.float32)).reshape((-1, 1)),
            'epi_distance': torch.from_numpy(epi_distance),
            'angle_difference': torch.from_numpy(np.ones(epi_distance.shape) * angle_difference),
            'shape': np.array([len(main_cam_inst_id), len(sec_cam_inst_id)]),
            'inst_pair': np.array(temp_list_inst_pairs),
            'scene_name': scene_name,
            'main_cam': str(cam_pair[0]),
            'sec_cam': str(cam_pair[1])
        }

        if 'single_crop' in self.zoomout_crop_num:
            for key, main_value in self.feat_vec_dict[scene_name][main_cam]['app_features'].items():
                temp_main_list = []
                temp_sec_list = []
                sec_value = self.feat_vec_dict[scene_name][sec_cam]['app_features'][key]
                for main_feat in main_value:
                    for sec_feat in sec_value:
                        temp_main_list.append(main_feat)
                        temp_sec_list.append(sec_feat)
                info['main_app_features'][key] = torch.from_numpy(np.array(temp_main_list))
                info['sec_app_features'][key] = torch.from_numpy(np.array(temp_sec_list))

        if 'multi_crops' in self.zoomout_crop_num:
            for key, main_value in self.feat_vec_dict[scene_name][main_cam]['multi_crop_features'].items():
                temp_main_list = []
                temp_sec_list = []
                sec_value = self.feat_vec_dict[scene_name][sec_cam]['multi_crop_features'][key]
                for main_feat in main_value:
                    for sec_feat in sec_value:
                        temp_main_list.append(main_feat)
                        temp_sec_list.append(sec_feat)
                info['multi_crop_features'][key] = torch.from_numpy(np.array(temp_main_list))
                info['multi_crop_features'][key] = torch.from_numpy(np.array(temp_sec_list))

        return info
