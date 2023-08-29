import hashlib
import json
import os
import pickle
from datetime import datetime

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

from configuration import Configuration


class TilesDataset(Dataset):
    """
    This class represent a dataset for a given list of slide ids.
    It is agnostic to the ids context, that is, it is unaware whether
     it is a train or val or test set.
    The ids should be determined by a different function
    """

    def __init__(self, tiles_directory, ids, mode, noise_ratio, pred=[], prob=[], caller=None):
        self.root_dir = tiles_directory
        # self.transform = transform
        self.transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.], std=[255.])
        ])

        self._files = self._load_files(ids)
        self.mode = mode
        self.noise_ratio = noise_ratio
        print("Loading: {} files".format(len(self._files)))
        print("Started loading at: {}".format(datetime.now()))
        if self.mode != "test":
            train_data = []
            train_label = []
            hash = hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()
            if os.path.exists(os.path.join(Configuration.CHECKPOINTS_PATH, hash)):
                with open(os.path.join(Configuration.CHECKPOINTS_PATH, hash), "rb") as f:
                    train_data, train_label = pickle.load(f)
            fail = 0
            c = 0
            for path, label in self._files:
                img_path = os.path.join(self.root_dir, path)
                img_obj = Image.open(img_path)
                img = np.array(Image.open(img_path))
                if img.shape != (512, 512, 3):
                    fail += 1
                else:
                    # reshape to (32, 32, 3)
                    img = np.asarray(img_obj.resize((32, 32)))
                    train_data.append(img)
                    train_label.append(label)
                c += 1
                if c % 1000 == 0:
                    print("loaded {} files, {}".format(c, datetime.now()))
            print(f"failed {fail}, which is {(fail / len(self._files)) * 100}%")
            train_data = np.stack(train_data)
            print("Ended loading at: {}".format(datetime.now()))
            with open(os.path.join(Configuration.CHECKPOINTS_PATH, hash), "wb") as f:
                pickle.dump((train_data, train_label), f)

            if os.path.exists(Configuration.NOISE_FILE):
                noise_label = json.load(open(Configuration.NOISE_FILE, "r"))
            else:
                noise_label = []
                train_size = len(train_data)
                idx = list(range(train_size))
                np.random.shuffle(idx)
                num_noise = int(self.noise_ratio * train_size)
                noise_idx = idx[:num_noise]
                for i in range(train_size):
                    if i in noise_idx:
                        # TODO- currently didn't handle asymmetric noise
                        random_label = np.random.randint(0, 2)
                        noise_label.append(random_label)
                    else:
                        noise_label.append(train_label[i])

                json.dump(noise_label, open(Configuration.NOISE_FILE, "w"))

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def get_num_of_files(self):
        return len(self._files)

    def _load_files(self, ids):
        files = os.listdir(self.root_dir)
        mapped_ids = dict(ids)
        matched_files = [file for file in files if any(file.startswith(prefix) for prefix, _ in ids)]
        labeled_files = [(file, mapped_ids[next(iter([prefix for prefix in mapped_ids.keys() if
                                                      file.startswith(prefix)]))]) for file in matched_files]

        return labeled_files

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        if self.mode == 'all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform_train(img)
            return img, target, index
        else:
            raise NotImplementedError()
