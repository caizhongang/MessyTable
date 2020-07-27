import numpy as np
from sklearn.metrics import average_precision_score

from sklearn.preprocessing import MinMaxScaler
import json
import KMSolver
from scipy.spatial.transform import Rotation as R


def crop_feat(img_copy, bbox, img_size, zoomout_ratio=1.0):
    """
    input: img and reuqirement on zoomout ratio
    where img_size = (max_x, max_y)
    return: a single img crop
    """
    x1, y1, x2, y2 = bbox
    max_y, max_x = img_size

    # below clip is for MPII dataset, where bbox < 0
    x1 = scalar_clip(x1, 0, max_x)
    y1 = scalar_clip(y1, 0, max_y)
    x2 = scalar_clip(x2, 0, max_x)
    y2 = scalar_clip(y2, 0, max_y)

    img_feat = None

    if zoomout_ratio == 1.0:
        img_feat = img_copy[int(y1):int(y2+1), int(x1):int(x2+1), :]
    elif zoomout_ratio > 1:
        h = y2 - y1
        w = x2 - x1
        img_feat = img_copy[int(max(0,y1-h*(zoomout_ratio-1)/2)):int(min(max_y-1,y2+1+h*(zoomout_ratio-1)/2)),
            int(max(0,x1-w*(zoomout_ratio-1)/2)):int(min(max_x-1,x2+1+w*(zoomout_ratio-1)/2)), :]
    return img_feat


def compute_angle(ext_a, ext_b, degree=True):
    """
    for evaluating the performance under different angle differences
    """
    T_a2r = np.eye(4)
    T_a2r[0:3, 0:3] = R.from_euler('xyz', ext_a[3:]).as_dcm()
    T_a2r[0:3, 3] = np.array(ext_a[:3])

    T_b2r = np.eye(4)
    T_b2r[0:3, 0:3] = R.from_euler('xyz', ext_b[3:]).as_dcm()
    T_b2r[0:3, 3] = np.array(ext_b[:3])

    # T_a2b = T_r2b * T_a2r = T_b2r.inv * T_a2r
    v_a = [0, 0, -1, 1]  # an vector pointing at negative z-axis, and a constant
    v_b = [0, 0, -1, 1]  # an vector pointing at negative z-axis, and a constant

    v_ainr = np.matmul(T_a2r, np.array(v_a))
    v_ainr = v_ainr[:3] / v_ainr[3]
    v_binr = np.matmul(T_b2r, np.array(v_b))
    v_binr = v_binr[:3] / v_binr[3]

    norm_a = np.sum(v_ainr ** 2) ** 0.5
    norm_b = np.sum(v_binr ** 2) ** 0.5

    alpha = np.arccos(np.dot(v_ainr / norm_a, v_binr / norm_b))

    if degree:
        return alpha / np.pi * 180.0
    else:
        return alpha


def epipolar_soft_constraint(
        bbox_list1, bbox_list2, intrin1, intrin2, extrin1, extrin2, config):
    """
    inputs:
    bbox list []
    instrin : (3,3) numpy
    extr: list of len 6
    """
    
    def ext_a2b(ext_a, ext_b):
        T_a2r = np.eye(4)
        T_a2r[0:3, 0:3] = R.from_euler('xyz', ext_a[3:]).as_dcm()
        T_a2r[0:3, 3] = np.array(ext_a[:3])

        T_b2r = np.eye(4)
        T_b2r[0:3, 0:3] = R.from_euler('xyz', ext_b[3:]).as_dcm()
        T_b2r[0:3, 3] = np.array(ext_b[:3])

        # T_a2b = T_r2b * T_a2r = T_b2r.inv * T_a2r
        T_a2b = np.matmul(np.linalg.inv(T_b2r), T_a2r)

        return T_a2b

    def find_line(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        d = (y2 - y1) / (x2 - x1)
        e = y1 - x1 * d
        return [-d, 1, -e]

    def find_foot(a, b, c, pt):
        x1, y1 = pt
        temp = (-1 * (a * x1 + b * y1 + c) / (a * a + b * b))
        x = temp * a + x1
        y = temp * b + y1
        return [x, y]

    def find_dist(pt1, pt2):
        return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

    T_a2b = ext_a2b(extrin1, extrin2)

    dist_matrix = np.zeros((len(bbox_list1), len(bbox_list2)))

    for i in range(len(bbox_list1)):
        for j in range(len(bbox_list2)):
            b1x1, b1y1, b1x2, b1y2 = bbox_list1[i]
            b2x1, b2y1, b2x2, b2y2 = bbox_list2[j]
            bbox1_2dpt = ((b1x1 + b1x2) / 2, (b1y1 + b1y2) / 2)
            bbox2_2dpt = ((b2x1 + b2x2) / 2, (b2y1 + b2y2) / 2)

            # bbox 1 in camera 2
            bbox1_3dpt = np.matmul(np.linalg.inv(intrin1), np.array([*bbox1_2dpt, 1]))
            bbox1_3dpt = np.array([*bbox1_3dpt.tolist(), 1])

            bbox1_in2_3dpt = np.matmul(T_a2b, bbox1_3dpt)[:3]
            bbox1_in2_2dpt = np.matmul(intrin2, bbox1_in2_3dpt)
            bbox1_in2_2dpt = bbox1_in2_2dpt[:2] / bbox1_in2_2dpt[2]

            # camera 1 epipole in camera 2
            epipole1_3dpt = np.array([0, 0, 0, 1])
            epipole1_in2_3dpt = np.matmul(T_a2b, epipole1_3dpt)[:3]
            epipole1_in2_2dpt = np.matmul(intrin2, epipole1_in2_3dpt)
            epipole1_in2_2dpt = epipole1_in2_2dpt[:2] / epipole1_in2_2dpt[2]

            # find epipolar line
            a, b, c = find_line(bbox1_in2_2dpt, epipole1_in2_2dpt)

            foot = find_foot(a, b, c, bbox2_2dpt)
            dist = find_dist(bbox2_2dpt, foot)

            # measure distance
            dist_matrix[i, j] = dist

    # normalize by diagonal line
    diag =  np.sqrt(config['original_img_size'][0]**2+config['original_img_size'][1]**2)
    dist_matrix = dist_matrix / diag
    return dist_matrix


def distance_apply_threshold(matrix, threshold=0.3):
    return (matrix < threshold).astype(np.int)


def apply_km(matrix, threshold=0.5):
    '''
    km: Hungarian Algo
    if the distance > threshold, even it is smallest, it is also false.
    '''
    cost_matrix = (matrix * 1000).astype(np.int)
    prediction_matrix_km = KMSolver.solve(cost_matrix, threshold=int(threshold*1000))
    return prediction_matrix_km


def scalar_clip(x, min, max):
    """
    input: scalar
    """
    if x < min:
        return min
    if x > max:
        return max
    return x


def list_clip(x_list, min, max):
    """
    input 
    """
    return [scalar_clip(x) for x in x_list]


def scale_data(x):
    # 1D data , scale (0, 1)
    scaler = MinMaxScaler()
    scaler.fit(x)
    return scaler.transform(x)


def get_random_portion(target_list, N):
    """
    get N samples randomly from target
    target can be list or np
    """
    if type(target_list) == type(np.array(target_list)):
        np.random.seed(0)
        perm_list = np.random.permutation(len(target_list))
        perm_seg = perm_list[0:N]
        return target_list[perm_seg]
    else:    
        np.random.seed(0)
        perm_list = np.random.permutation(len(target_list))
        target_np = np.array(target_list)
        perm_seg = perm_list[0:N]
        return list(target_np[perm_seg])


def get_stats(data_input, label, data_type='tensor', title='', rescale=False, print_precision=3):
    """
    print stats for pos and neg separately
    both data label are (N,1) tensor or np
    return list of pos and neg data
    Note: dont change input
    """
    data = np.copy(data_input)
    np.set_printoptions(precision=2)
    if data_type == 'list':
        data = np.array(data).reshape((-1,1))
        label = np.array(label).reshape((-1,1))
    if data_type == 'tensor':
        data = data.cpu().detach().numpy()
        label = label.cpu().detach().numpy()  
    if rescale == True:
        data =  scale_data(data)
    data = data.squeeze()
    label = label.squeeze()

    idx_pos = [i for i, j in enumerate(label) if j==1.0]
    idx_neg = [i for i, j in enumerate(label) if j==0.0]

    print ("[[",title,'stats -- pos ]]: mean: {:0.3f};   std: {:0.3f}'.format(np.mean(data[idx_pos]),np.std(data[idx_pos])))
    print ("[[",title,'stats -- neg ]]: mean: {:0.3f};   std: {:0.3f}'.format(np.mean(data[idx_neg]),np.std(data[idx_neg])))


def compute_bbox_stats(scene_pathnames):
    h_list  = []
    w_list = []
    for pathname in scene_pathnames:
        with open(pathname, 'r') as file:
            content = json.load(file)
        for cam, cam_value in content['cameras'].items():
            for inst, inst_value in cam_value['instances'].items():
                x1, y1, x2, y2 = inst_value['pos']
                w_list.append(abs(x1-x2))
                h_list.append(abs(y1-y2))
    h_np = np.array(h_list)
    w_np = np.array(w_list)
    for hw_np in [h_np, w_np]:
        print (int(np.mean(hw_np)),int(np.min(hw_np)),int(np.percentile(hw_np, 10)),int(np.percentile(hw_np, 25)),
        int(np.percentile(hw_np, 50)),int(np.percentile(hw_np, 75)),int(np.percentile(hw_np, 90)),int(np.max(hw_np)))


def eval_matrix(con_mat):
    TN, FP, FN, TP = list(con_mat)
    precision = TP / (FP + TP)
    recall = TP / (FN + TP)
    F1 = 2*precision*recall/(precision+recall)
    print('P = {:0.3f}'.format(precision))
    print('R = {:0.3f}'.format(recall))
    print('F1 = {:0.3f}'.format(F1))
    return precision, recall, F1


def compute_IPAA(distances, sample_dict,IPAA_dict, km_threshold=0.8):
    main_bbox_id_list = sample_dict['main_bbox_id']
    sec_bbox_id_list = sample_dict['sec_bbox_id']
    gt_subcls_list = sample_dict['gt_subcls']
    gt_inst_list = sample_dict['gt_inst']
    distances_mat = np.array(distances).reshape(len(main_bbox_id_list),len(sec_bbox_id_list))
    gt_inst_mat = np.array(gt_inst_list).reshape(len(main_bbox_id_list),len(sec_bbox_id_list))
    pred_mat_instance = apply_km(distances_mat, km_threshold) 
    wrong_count = 0
    id_list = list(set(main_bbox_id_list+sec_bbox_id_list))
    for id in id_list:
        if id in main_bbox_id_list and id in sec_bbox_id_list:
            idx_main = main_bbox_id_list.index(id)
            if not (pred_mat_instance[idx_main,:] == gt_inst_mat[idx_main,:]).all():
                wrong_count+=1
        elif id in main_bbox_id_list:
            idx_main = main_bbox_id_list.index(id)
            if sum(pred_mat_instance[idx_main,:]) != 0:
                wrong_count+=1
        elif id in sec_bbox_id_list:
            idx_sec = sec_bbox_id_list.index(id)
            if sum(pred_mat_instance[:,idx_sec]) != 0:
                wrong_count+=1
        else:
            raise 
    p = float(wrong_count)/float(len(id_list))

    for key in IPAA_dict.keys():
        if (1-p)*100 >= key:
            IPAA_dict[key] +=1


def convert_IPAA(IPAA_dict,num_sample, precision=3):
    pct = {}
    for key, value in IPAA_dict.items():
        pct[key] = round(value/num_sample,precision)
    
    return pct


def FPR_95(overall_distance_np, gt_inst_np):
    """
    compute FPR@95
    """
    recall_point = 0.95
    labels = gt_inst_np.squeeze()
    scores = (1- scale_data(overall_distance_np)).squeeze()
    # Sort label-score tuples by the score in descending order.
    indices = np.argsort(scores)[::-1]   
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    n_match = sum(sorted_labels)
    n_thresh = recall_point * n_match
    thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
    FP = np.sum(sorted_labels[:thresh_index] == 0)
    TN = np.sum(sorted_labels[thresh_index:] == 0)
    return float(FP) / float(FP + TN)


def cal_mAP(threshold, label):
    return average_precision_score(label, threshold)      

