from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import random
import torchvision.transforms as transforms

class ImagesTextDataset(Dataset):

    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None, train=True):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts
        self.train = train
        if train:
            self.text_paths_dir = '/scratch/users/abaykal20/sam/SAM/mmcelebhq/train_captions/'
        else:
            self.text_paths_dir = '/scratch/users/abaykal20/sam/SAM/mmcelebhq/test_captions/'
        self.text_paths = sorted(data_utils.make_text_dataset(self.text_paths_dir))
        self.rsz_transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        #if self.train and self.opts is not None and self.opts.datasetSize is not None and len(self.source_paths) >= self.opts.datasetSize:
            #return self.opts.datasetSize
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = from_im.convert('RGB')

        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert('RGB')
        orig = self.rsz_transform(from_im)
        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im
      
        txt_file_dir = self.text_paths[index]
        txt_file = open(txt_file_dir,"r")
        txt = txt_file.read().splitlines()
        txt = txt[0] #random.choice(txt)

        return from_im, to_im, txt#, orig # orig if using discriminator
  