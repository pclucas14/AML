import os
import sys
import pdb
import torch
import kornia
import numpy as np

from copy import deepcopy
from torchvision import datasets, transforms


""" Datasets """
class MiniImagenet(datasets.ImageFolder):

    default_size = 84
    default_n_tasks = 20

    def __init__(self, root, train=True, transform=None, download=False):

        if download:
            dump_path = os.path.join(root, 'miniimagenet')
            if os.path.exists(dump_path):
                print('MiniIm directory exists, skipping download')
            else:
                download_file_from_google_drive(
                    '1f-AR7gWPOvo5Noxi25hDE8LD878oa5vc',
                    root,
                    'miniim.tar.gz'
                )
                os.system('cd miniimagenet & tar -xvf miniim.tar.gz')

        path = os.path.join(root, 'miniimagenet', 'train' if train else 'test')
        print(path)
        super(MiniImagenet, self).__init__(
            root=path,
            transform=transform
        )

        self.data = np.array([x[0] for x in self.samples])


    def base_transforms(H=None):
        """ base transformations applied to *train* images """

        if H is None:
            H = MiniImagenet.default_size

        tfs = transforms.Compose([
               transforms.Resize(int(H * 1.25)),
               transforms.CenterCrop(int(H * 1.15)), # used to be 1.15
               transforms.ToTensor(),
               lambda x : (x - .5) * .5
        ])

        return tfs


    def train_transforms(H=None, use_augs=False):
        """ extra augs applied over *training* images """

        if H is None:
            H = MiniImagenet.default_size

        if use_augs:
            tfs = torch.nn.Sequential(
                kornia.augmentation.RandomCrop(size=(H, H)),
                kornia.augmentation.RandomHorizontalFlip(),
            )
        else:
            tfs = kornia.augmentation.CenterCrop(size=(H, H))

        return tfs


    def eval_transforms(H):
        """ base transformations applied during evaluation """

        if H is None:
            H = MiniImagenet.default_size

        tfs = transforms.Compose([
               transforms.Resize(int(H * 1.15)), # should this be 1.25 ?
               transforms.CenterCrop(H),
               transforms.ToTensor(),
               lambda x : (x - .5) * .5
        ])

        return tfs


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

