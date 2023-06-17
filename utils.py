import numpy as np
from sklearn.model_selection import train_test_split

def get_classes_map(path):
    class_to_id = {}
    id_to_class = {}
    
    with open(f"{path}/food-101/meta/classes.txt", "r") as f:
        for idx, class_str in enumerate(f):
            class_str = class_str[:-1]
            class_to_id[class_str] = idx
            id_to_class[idx] = class_str
            
    return class_to_id, id_to_class
    
def collect_train_val(path):
    class_to_id, _ = get_classes_map(path)
    
    train_files, val_files = [], []
    train_target, val_target = [], []
    
    def read_from(file, out, target):
        with open(file, "r") as f:
            for s in f:
                s = s[:-1]
                class_str = s.split("/")[0]
                out.append(s)
                target.append(class_to_id[class_str])
                    
    read_from(f"{path}/food-101/meta/train.txt", train_files, train_target)
    read_from(f"{path}/food-101/meta/test.txt", val_files, val_target)
    
    return np.array(train_files), np.array(val_files), np.array(train_target), np.array(val_target)


def split_train(train_files, train_target, colorization_size=0.9, seed=42):
    train_color_files, train_class_files, train_color_target, train_class_target = train_test_split(train_files,
                     train_target, 
                     train_size=colorization_size,
                     random_state=seed,
                     stratify=train_target)
    
    return np.array(train_color_files), np.array(train_class_files), np.array(train_color_target), np.array(train_class_target)
  
def split_in_out_domain(files, targets, out_classes):
    mask = np.isin(targets, out_classes)
    
    return files[~mask], targets[~mask], files[mask], targets[mask]

            