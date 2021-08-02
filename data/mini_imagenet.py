import os
import sys
import pdb
import torch
import kornia
import numpy as np

import pickle as pkl
from copy import deepcopy
from torchvision import datasets, transforms

""" Datasets """
class MiniImagenet(datasets.VisionDataset):

    default_size = 84
    default_n_tasks = 20

    def __init__(self, root, train=True, transform=None, download=False):

        if download:
            dump_path = os.path.join(root, 'mini_imagenet')
            if os.path.exists(dump_path):
                print('MiniIm directory exists, skipping download')
            else:
                download_file_from_google_drive(
                    '1gJBUejzwUxBZWaQ_kYYvbXvxcQ0h1PbP',
                    dump_path,
                    'archive.zip'
                )
                os.system(f'cd {dump_path} && unzip archive.zip')

        all_data = []
        for split in ['train', 'val', 'test']:
            afile = open(os.path.join(root, f'mini_imagenet/mini-imagenet-cache-{split}.pkl'), 'rb')
            data  = pkl.load(afile)['image_data'].reshape(-1, 600, 84, 84, 3)
            all_data += [data]

        all_data = np.concatenate(all_data)

        split = int(1 / 6 * all_data.shape[1])
        all_data = all_data[:, split:] if train else all_data[:, :split]
        all_data = (torch.from_numpy(all_data).float() / 255. - .5) * 2

        self.data = all_data.reshape(-1, *all_data.shape[2:]).permute(0, 3, 2, 1)
        self.targets = np.arange(100).reshape(-1, 1).repeat(all_data.size(1), axis=1).reshape(-1)
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y



    def base_transforms():
        """ base transformations applied to *train* images """

        return None


    def train_transforms(use_augs=False):
        """ extra augs applied over *training* images """

        H = MiniImagenet.default_size

        if use_augs:
            tfs = torch.nn.Sequential(
                kornia.augmentation.RandomCrop(size=(H, H), padding=4, fill=-1),
                kornia.augmentation.RandomHorizontalFlip(),
            )
        else:
            tfs = torch.nn.Identity()

        return tfs


    def eval_transforms():
        """ base transformations applied during evaluation """

        return None


# --- Utilities to download the dataset from google drive --- #

# https://github.com/tristandeleu/pytorch-meta/
from torchvision.datasets.utils import _get_confirm_token, _save_response_content

def _quota_exceeded(response: "requests.models.Response"):
    return False
    # See https://github.com/pytorch/vision/issues/2992 for details
    # return "Google Drive - Quota exceeded" in response.text


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        if _quota_exceeded(response):
            msg = (
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )
            raise RuntimeError(msg)

        _save_response_content(response, fpath)

