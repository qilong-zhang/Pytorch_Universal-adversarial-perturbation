import torch as t
from torch.utils import data
from PIL import Image
import os
from scipy.misc import imread, imresize


class ImagetNet(data.Dataset):
    # './data/ILSVRC2012/'
    def __init__(self, path_train_imagenet, num_classes, num_imgs_per_batch, transforms=None):
        self.path_img = []
        self.transforms = transforms

        dirs = [x[0] for x in os.walk(path_train_imagenet)]
        dirs = dirs[1:]
        # Sort the directory in alphabetical order (same as synset_words.txt)
        dirs = sorted(dirs)
        it = 0
        Matrix = [0 for x in range(1200)]

        for d in dirs:
            for root, dir, filename in os.walk(d):
                Matrix[it] = filename
            it = it + 1

        for k in range(num_classes):
            for u in range(num_imgs_per_batch):
                img_path = os.path.join(dirs[k], Matrix[k][u])
                # try:
                #     data = Image.open(img_path)
                #     data = self.transforms(data)
                #     self.path_img.append(img_path)
                # except Exception as e:
                #     print(e)
                # print('Processing image number ', it)
                self.path_img.append(img_path)


    def __getitem__(self, index):
        path_img = self.path_img[index]
        data = Image.open(path_img).convert('RGB')
        if self.transforms:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.path_img)


class Val(data.Dataset):
    # '../val/'
    def __init__(self, path_val_imagenet, transforms=None):
        self.path_img = []
        self.transforms = transforms

        filename = [x[2] for x in os.walk(path_val_imagenet)][0]
        for name in filename:
            img_path = os.path.join(path_val_imagenet, name)
            # txt_file = open('val.txt', 'a')
            # try:
            #     data = Image.open(img_path)
            #     data = self.transforms(data)
                # self.path_img.append(img_path)
                # txt_file.write(img_path + '\n')

            # except Exception as e:
            #     print(e)
            # txt_file.close()

        txt_file = open('val.txt', 'r')
        for line in txt_file.readlines():
            img_path = os.path.join(path_val_imagenet, line.rstrip('\n'))
            self.path_img.append(img_path)
        txt_file.close()

    def __getitem__(self, index):
        path_img = self.path_img[index]
        data = Image.open(path_img).convert('RGB')
        if self.transforms:
            try:
                data = self.transforms(data)
            except:
                print(path_img)
        return data

    def __len__(self):
        return len(self.path_img)


