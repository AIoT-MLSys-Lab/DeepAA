import numpy as np
import tensorflow as tf
from torchvision.datasets.imagenet import *
from torch import randperm, default_generator
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset


_DATA_TYPE = tf.float32

CMYK_IMAGES = [
    'n01739381_1309.JPEG',
    'n02077923_14822.JPEG',
    'n02447366_23489.JPEG',
    'n02492035_15739.JPEG',
    'n02747177_10752.JPEG',
    'n03018349_4028.JPEG',
    'n03062245_4620.JPEG',
    'n03347037_9675.JPEG',
    'n03467068_12171.JPEG',
    'n03529860_11437.JPEG',
    'n03544143_17228.JPEG',
    'n03633091_5218.JPEG',
    'n03710637_5125.JPEG',
    'n03961711_5286.JPEG',
    'n04033995_2932.JPEG',
    'n04258138_17003.JPEG',
    'n04264628_27969.JPEG',
    'n04336792_7448.JPEG',
    'n04371774_5854.JPEG',
    'n04596742_4225.JPEG',
    'n07583066_647.JPEG',
    'n13037406_4650.JPEG',
]

PNG_IMAGES = ['n02105855_2933.JPEG']

class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.
    Copied from torchvision, besides warning below.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

        WARN::
        This is the same ImageNet class as in torchvision.datasets.imagenet, but it has the `ignore_archive` argument.
        This allows us to only copy the unzipped files before training.
    """

    def __init__(self, root, split='train', download=None, ignore_archive=False, **kwargs):
        if download is True:
            msg = ("The dataset is no longer publicly accessible. You need to "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)
        elif download is False:
            msg = ("The use of the download flag is deprecated, since the dataset "
                   "is no longer publicly accessible.")
            warnings.warn(msg, RuntimeWarning)

        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        if not ignore_archive:
            self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    def parse_archives(self):
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

class ImageNet_DeepAA(ImageNet):
    def __init__(self, root, split='train', download=None, **kwargs):
        super(ImageNet_DeepAA, self).__init__(root, split=split, download=download, ignore_archive=True, **kwargs)
        _, self.labels_ = zip(*self.samples)

    def on_epoch_end(self):
        print('Dummy one_epoch_end for ImageNet dataset using torchvision')
        pass

    def sample_labeled_data_batch(self, label, val_bs): # generate val and train batch at the same time
        matched_indices = [id for id, lab in enumerate(self.labels_) if lab==label]
        matched_indices = np.array(matched_indices)
        assert len(matched_indices) > val_bs, 'Make sure the have enough data'
        np.random.shuffle(matched_indices)
        val_indices = matched_indices[:val_bs]

        val_samples, val_labels = zip(*[self[id] for id in val_indices])
        val_samples = list(val_samples)
        val_labels = np.array(val_labels, dtype=np.int32)

        return val_samples, val_labels

class Subset_ImageNet(Subset):
    def __init__(self, dataset, indices):
        super(Subset_ImageNet, self).__init__(dataset, indices)
        self.subset_labels_ = [self.dataset.labels_[k] for k in indices]


    def on_epoch_end(self):
        pass

    def sample_labeled_data_batch(self, label, val_bs):
        matched_indices = [self.indices[id] for id, lab in enumerate(self.subset_labels_) if lab == label]
        matched_indices = np.array(matched_indices)
        assert len(matched_indices) > val_bs, 'Make sure the have enough data'
        np.random.shuffle(matched_indices)
        val_indices = matched_indices[:val_bs]

        val_samples, val_labels = zip(*[self.dataset[id] for id in val_indices])  # applies transforms
        val_samples = list(val_samples)
        val_labels = np.array(val_labels, dtype=np.int32)

        return val_samples, val_labels

def random_split_ImageNet(dataset, lengths, generator=default_generator):
    if sum(lengths) != len(dataset):
        raise ValueError('Sum of input lengths does not equal the length of the input dataset')
    indices = randperm(sum(lengths), generator=generator).tolist()
    return [Subset_ImageNet(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

def get_imagenet_split(val_size=400000, train_sep_size=100000, dataroot='./data', n_GPU=None, seed=300):
    transform = lambda img: np.array(img, dtype=np.uint8)
    total_trainset = ImageNet_DeepAA(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=transform)
    testset = ImageNet_DeepAA(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=transform)

    N_per_shard = (len(total_trainset) - val_size - train_sep_size)//n_GPU
    remaining_data = len(total_trainset) - val_size - train_sep_size - n_GPU * N_per_shard
    if remaining_data > 0:
        splits = [val_size, train_sep_size, *[N_per_shard]*n_GPU, remaining_data]
    else:
        splits = [val_size, train_sep_size, *[N_per_shard]*n_GPU]
    all_ds = random_split_ImageNet(total_trainset,
                               lengths=splits,
                               generator=torch.Generator().manual_seed(seed))
    val_ds = all_ds[0]
    train_ds_sep = all_ds[1]
    pretrain_ds_splits = all_ds[2:2+n_GPU]
    return total_trainset, val_ds, train_ds_sep, pretrain_ds_splits, testset