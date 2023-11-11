import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root='./datasets', transform=None, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode

        # Load image paths and corresponding labels
        self.image_paths = []
        self.labels = []

        if self.mode == 'train':
            self.dataset_path = os.path.join(self.root, 'seg_train', 'seg_train')
        elif self.mode == 'val':
            self.dataset_path = os.path.join(self.root, 'seg_test', 'seg_test')
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented yet...")

        # Get class names
        self.class_names = sorted(os.listdir(os.path.join(self.dataset_path)))

        # Load image paths and labels using class names
        for idx, class_name in enumerate(self.class_names):
            for image_name in os.listdir(os.path.join(self.dataset_path, class_name)):
                self.image_paths.append(os.path.join(self.dataset_path, class_name, image_name))
                self.labels.append(idx)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]

        # Apply transform if not None
        if self.transform:
            image = self.transform(image)
        
        return image, label
