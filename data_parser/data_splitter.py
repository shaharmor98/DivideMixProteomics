import random


class DataSplitter(object):
    def __init__(self, dia):
        self.dia = dia

    def split_train_val(self, instances, seed=42, val_proportion=0.3):
        """
        Expecting input to be a list of lists, first element is slide name, second is slide label
        """

        high_instances = [slide for slide, label in instances.items() if label == 1]
        low_instances = [slide for slide, label in instances.items() if label == 0]

        validation_set_size = int(len(high_instances) * val_proportion)

        random.seed(seed)
        random.shuffle(high_instances)
        random.shuffle(low_instances)

        train_records = []
        validation_records = []

        for instance in high_instances[:validation_set_size]:
            validation_records.append((instance, 1))
        for instance in high_instances[validation_set_size:]:
            train_records.append((instance, 1))
        for instance in low_instances[:validation_set_size]:
            validation_records.append((instance, 0))
        for instance in low_instances[validation_set_size:]:
            train_records.append((instance, 0))

        print(f"Train records: {len(train_records)}")
        print(f"Validation records: {len(validation_records)}")
        return train_records, validation_records
