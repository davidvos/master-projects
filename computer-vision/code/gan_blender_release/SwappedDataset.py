from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F


# Helper function to quickly see the values of a list or dictionary of data
def printTensorList(data, detailed=False):
    if isinstance(data, dict):
        print('Dictionary Containing: ')
        print('{')
        for key, tensor in data.items():
            print('\t', key, end='')
            print(' with Tensor of Size: ', tensor.size())
            if detailed:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
        print('}')
    else:
        print('List Containing: ')
        print('[')
        for tensor in data:
            print('\tTensor of Size: ', tensor.size())
            if detailed:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
        print(']')


class SwappedDatasetLoader(Dataset):

    def __init__(self, data_file, prefix, resize=256):
        self.prefix = prefix
        self.resize = resize
        
        file_names = open(data_file, 'r')
        self.file_list = file_names.readlines()
        
        self.transforms = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])

    def __len__(self):
        # Return the length of the datastructure that is your dataset
        return len(self.file_list)

    def __getitem__(self, index):

        filename = self.file_list[index]
        splitted_filename = filename.split('_')
        x = splitted_filename[0]
        y = splitted_filename[2]
        z = splitted_filename[3].split('.')[0]
        
        source = self.transforms(Image.open(self.prefix + x + '_fg_' + z + '.png'))
        target = self.transforms(Image.open(self.prefix + x + '_bg_' + y + '.png'))
        swap = self.transforms(Image.open(self.prefix + x + '_sw_' + y + '_' + z + '.png'))
        mask = self.transforms(Image.open(self.prefix + x + '_mask_' + y + '_' + z + '.png'))

        image_dict = {  'source': source,
                        'target': target,
                        'swap': swap,
                        'mask': mask }

        return filename, image_dict


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    # It is always a good practice to have separate debug section for your
    # functions. Test if your dataloader is working here. This template creates
    # an instance of your dataloader and loads 20 instances from the dataset.
    # Fill in the missing part. This section is only run when the current file
    # is run and ignored when this file is imported.

    # This points to the root of the dataset
    data_root = '../../data_set/data/'
    # This points to a file that contains the list of the filenames to be
    # loaded.
    test_list = '../../data_set/test.str'
    print('[+] Init dataloader')
    # Fill in your dataset initializations
    testSet = SwappedDatasetLoader(test_list, data_root)
    print('[+] Create workers')
    loader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    print('[*] Dataset size: ', len(loader))
    enu = enumerate(loader)
    for i in range(20):
        a = time.time()
        i, (images) = next(enu)
        b = time.time()
        # Uncomment to use a prettily printed version of a dict returned by the
        # dataloader.
        # printTensorList(images[0], True)
        print('[*] Time taken: ', b - a)
