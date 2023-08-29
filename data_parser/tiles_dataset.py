import os

from PIL import Image
from torch.utils.data import Dataset


class TilesDataset(Dataset):
    """
    This class represent a dataset for a given list of slide ids.
    It is agnostic to the ids context, that is, it is unaware whether
     it is a train or val or test set.
    The ids should be determined by a different function
    """

    def __init__(self, tiles_directory, transform, ids, caller=None):
        self.root_dir = tiles_directory
        self.transform = transform
        self._files = self._load_files(ids)
        print("Loading: {} files".format(len(self._files)))

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
        return len(self._files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self._files[index][0])

        # TODO- remove when running on actual env!!!
        if os.path.basename(img_path) == ".DS_Store":
            index += 1
            img_path = os.path.join(self.root_dir, self._files[index][0])

        img = Image.open(img_path)
        img = self.transform(img)

        return img, self._files[index][1]
