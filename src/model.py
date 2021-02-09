import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


class FeaturesRes18(nn.Module):
    """
    backbone: resnet18
    """

    def __init__(self):
        super(FeaturesRes18, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        # resnet18.load_state_dict(torch.load('resnet18.pth'))
        modules = list(resnet18.children())[:-1]
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        return output


class FeaturesRes50(nn.Module):
    """
    backbone: resnet50
    """

    def __init__(self):
        super(FeaturesRes50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        # resnet50.load_state_dict(torch.load('resnet50.pth'))
        modules = list(resnet50.children())[:-1]
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        return output


class FeaturesRes101(nn.Module):
    """
    backbone: resnet101
    """

    def __init__(self):
        super(FeaturesRes101, self).__init__()
        resnet101 = models.resnet101(pretrained=True)
        # resnet101.load_state_dict(torch.load('resnet101.pth'))
        modules = list(resnet101.children())[:-1]
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        return output


class TripletNet(nn.Module):
    """
    Triplet net that can put different backbones
    """

    def __init__(self, config):
        super(TripletNet, self).__init__()
        self.config = config
        self.device = config['device']
        self.features_net = globals()[config['features_net']]()

    def forward_once(self, x):
        """
        input is simply one tensor
        """
        output = self.features_net(x)
        output = output.view(output.size()[0], -1)
        return {'feat': output}

    def compute_distance(self, a_dict, b_dict):
        a_feat = a_dict['feat']
        b_feat = b_dict['feat']
        dist = F.pairwise_distance(a_feat, b_feat)
        return dist

    def forward(self, sample_dict):
        input_anchor = sample_dict['a']
        input_positive = sample_dict['p']
        input_negative = sample_dict['n']

        a_dict = self.forward_once(input_anchor)
        p_dict = self.forward_once(input_positive)
        n_dict = self.forward_once(input_negative)

        output_dict = {
            'ap': self.compute_distance(a_dict, p_dict),
            'an': self.compute_distance(a_dict, n_dict)
        }
        return output_dict


class TripletLoss(nn.Module):
    """
    Triplet loss, argument: margin
    """

    def __init__(self, config):
        super(TripletLoss, self).__init__()
        self.margin = config['triplet_margin']

    def forward(self, output_dict, sample_dict):
        ap = output_dict['ap']
        an = output_dict['an']
        loss_triplet = torch.mean(F.relu(ap - an + self.margin))
        return loss_triplet


class ASNet(nn.Module):
    """
    ASNet: Appearance Surrounding
    """

    def __init__(self, config):
        super(ASNet, self).__init__()
        self.config = config
        if config['features_net'] == 'FeaturesRes18':
            app = models.resnet18(pretrained=True)
            # app.load_state_dict(torch.load('resnet18.pth'))
            self.app = nn.Sequential(*list(app.children())[:-2])

            sur = models.resnet18(pretrained=True)
            # sur.load_state_dict(torch.load('resnet18.pth'))
            self.sur = nn.Sequential(*list(sur.children())[:-2])

            self.dim = 512
        elif config['features_net'] == 'FeaturesRes50':
            print('model type is FeaturesRes50')
            app = models.resnet50(pretrained=True)
            # app.load_state_dict(torch.load('resnet50.pth'))
            self.app = nn.Sequential(*list(app.children())[:-2])

            sur = models.resnet50(pretrained=True)
            # sur.load_state_dict(torch.load('resnet50.pth'))
            self.sur = nn.Sequential(*list(sur.children())[:-2])

            self.dim = 2048
        elif config['features_net'] == 'FeaturesRes101':
            print('model type is FeaturesRes101')
            app = models.resnet101(pretrained=True)
            # app.load_state_dict(torch.load('resnet101.pth'))
            self.app = nn.Sequential(*list(app.children())[:-2])

            sur = models.resnet101(pretrained=True)
            # sur.load_state_dict(torch.load('resnet101.pth'))
            self.sur = nn.Sequential(*list(sur.children())[:-2])

            self.dim = 2048
        else:
            raise NotImplementedError('Currently only implemented for res18')

        assert len(config['zoomout_ratio']) == 1
        self.zoomout_ratio = config['zoomout_ratio'][0]
        self.keep_center = config.get('keep_center', False)

        H = config['cropped_height']
        W = config['cropped_width']

        self.H_size = int(H / self.zoomout_ratio)  # size of center
        self.H_min = int((H - self.H_size) / 2)
        self.H_max = self.H_min + self.H_size

        self.W_size = int(W / self.zoomout_ratio)  # size of center
        self.W_min = int((W - self.W_size) / 2)
        self.W_max = self.W_min + self.W_size

        # print('Crop sizes. H', H, 'H_min', self.H_min, 'H_max', self.H_max, 'W', W, 'W_min', self.W_min, 'W_max', self.W_max)

    def rotate(self, x, angle):
        if angle == 0:
            return x
        elif angle == 90:
            return x.transpose(-2, -1).flip(-2)
        elif angle == 180:
            return x.flip(-2).flip(-1)
        elif angle == 270:
            return x.transpose(-2, -1).flip(-1)
        else:
            raise ValueError('Unable to handle angle ==', angle)

    def forward_once(self, x):
        B, C, H, W = x.size()

        # crop out center
        center = x[:, :, self.H_min:self.H_max, self.W_min:self.W_max].clone()
        surrnd = x

        if not self.keep_center:
            surrnd[:, :, self.H_min:self.H_max, self.W_min:self.W_max] = \
                torch.zeros((B, C, self.H_size, self.W_size)).to(x.device)

        # extract features
        center_featmap = self.app(center)
        surrnd_featmap = self.sur(surrnd)

        # average pooling
        center_featvec = F.adaptive_avg_pool2d(center_featmap, output_size=1).view(B, -1)
        surrnd_featvec = F.adaptive_avg_pool2d(surrnd_featmap, output_size=1).view(B, -1)

        return {'center': center_featvec, 'surrnd': surrnd_featvec}

    def compute_distance(self, a_dict, b_dict):
        a_center = a_dict['center']
        a_surrnd = a_dict['surrnd']
        b_center = b_dict['center']
        b_surrnd = b_dict['surrnd']
        dist_a_center = F.pairwise_distance(a_center, b_center)  # shape (B, 1)
        dist_b_surrnd = F.pairwise_distance(a_surrnd, b_surrnd)
        surrnd_weight = F.cosine_similarity(a_center, b_center)
        dist = (1 - surrnd_weight) * dist_a_center + surrnd_weight * dist_b_surrnd
        return dist

    def forward(self, sample_dict):
        a = sample_dict['a']
        p = sample_dict['p']
        n = sample_dict['n']

        if self.config.get('rotate', False):
            a = self.rotate(a, 90 * np.random.randint(0, 4))

        output_dict = {
            'a': self.forward_once(a),
            'p': self.forward_once(p),
            'n': self.forward_once(n)
        }
        return output_dict


class CooperativeTripletLoss(nn.Module):
    """
    Loss for ASNet
    Cosine similarity as weight
    """

    def __init__(self, config):
        super(CooperativeTripletLoss, self).__init__()
        self.config = config
        self.margin = config.get('triplet_margin', 0.3)
        print('Cooperative Triplet Loss with margin =', self.margin)

    def forward(self, output_dict, sample_dict):
        anc_center = output_dict['a']['center']
        anc_surrnd = output_dict['a']['surrnd']
        pos_center = output_dict['p']['center']
        pos_surrnd = output_dict['p']['surrnd']
        neg_center = output_dict['n']['center']
        neg_surrnd = output_dict['n']['surrnd']

        # compute pos distance
        dist_pos_center = F.pairwise_distance(anc_center, pos_center)  # shape (B, 1)
        dist_pos_surrnd = F.pairwise_distance(anc_surrnd, pos_surrnd)
        pos_surrnd_weight = F.cosine_similarity(anc_center, pos_center)
        dist_pos = (1 - pos_surrnd_weight) * dist_pos_center + pos_surrnd_weight * dist_pos_surrnd

        # compute neg distance
        dist_neg_center = F.pairwise_distance(anc_center, neg_center)
        dist_neg_surrnd = F.pairwise_distance(anc_surrnd, neg_surrnd)
        neg_surrnd_weight = F.cosine_similarity(anc_center, neg_center)
        dist_neg = (1 - neg_surrnd_weight) * dist_neg_center + neg_surrnd_weight * dist_neg_surrnd

        loss_triplet = torch.mean(F.relu(dist_pos - dist_neg + self.margin))
        return loss_triplet
