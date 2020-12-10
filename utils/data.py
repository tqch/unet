from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os,re
from hashlib import md5
import wget
from zipfile import ZipFile
from sklearn.model_selection import train_test_split


class ButterflyDataset(Dataset):

    resize = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(size=resize),
        transforms.ToTensor()
    ])

    def __init__(self, root, mapping, transform=transform):
        self.root = root
        self.mapping = mapping
        self.transform = transform
        self.length = len(mapping)

    def __getitem__(self, idx):
        item = self.mapping[idx]
        with Image.open(os.path.join(self.root, item["data"])) as image:
            out_image = self.transform(image)
        with Image.open(os.path.join(self.root, item["mask"])) as mask:
            out_mask = self.transform(mask).long()
        return out_image,out_mask

    def __len__(self):
        return self.length


class Datasets:
    urls = {
        "butterfly": "http://www.josiahwang.com/dataset/leedsbutterfly/leedsbutterfly_dataset_v1.0.zip"
    }
    md5s = {
        "butterfly": "d395ad65a68cf6b31c7c9e1e2f735e3c"
    }
    datasets = {
        "butterfly": "leedsbutterfly"
    }

    def __init__(self,root,download=True,dataset="butterfly"):
        self.root = root
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.download = download
        self.dataset = dataset
        self.data_dir = os.path.join(self.root,self.datasets[self.dataset])
        if os.path.exists(self.data_dir):
            self.download = False
        if self.download:
            filename = wget.download(self.urls[self.dataset],out=self.root)
            if self.checkmd5(filename):
                print("MD5 check pass!")
            else:
                print("MD5 check fail!")
                exit()
            with ZipFile(filename) as zip:
                zip.extractall(path=root)
        else:
            print("Dataset exists!")

        self.trainset, self.testset = self.get_dataset()

    def checkmd5(self,file):
        with open(file,"rb") as f:
            result = md5(f.read()).hexdigest() == self.md5s[self.dataset]
        return result

    def get_dataset(self):
        if self.dataset == "butterfly":
            image_file = os.listdir(os.path.join(self.data_dir, "images"))
            image_id = sorted([re.search("(\d+).png", img).group(1) for img in image_file])
            train_id, test_id = train_test_split(image_id, test_size=0.1, shuffle=True, random_state=42)
            train_mapping = {i: {"data": "images/" + idx + ".png",
                                 "mask": "segmentations/" + idx + "_seg0.png"} for i, idx in enumerate(train_id)}
            test_mapping = {i: {"data": "images/" + idx + ".png",
                                "mask": "segmentations/" + idx + "_seg0.png"} for i, idx in enumerate(test_id)}
            return ButterflyDataset(self.data_dir, train_mapping),ButterflyDataset(self.data_dir, test_mapping)