import numpy as np


def load_keel_dat(file_path):
    data, labels, label_map = [], [], {}
    with open(file_path, 'r') as f:
        is_data = False
        for line in f:
            line = line.strip()
            if line.lower().startswith('@data'):
                is_data = True
                continue
            if not is_data or line.startswith('@') or not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            features = [float(x) for x in parts[:-1]]
            label_str = parts[-1]
            if label_str not in label_map:
                label_map[label_str] = len(label_map)
            data.append(features)
            labels.append(label_map[label_str])

    X = np.array(data)
    y = np.array(labels)

    unique, counts = np.unique(y, return_counts=True)
    if counts[0] < counts[1]:
        y = 1 - y

    print(f"Dataset Loaded: {file_path} | Shape: {X.shape} | Distribution: {dict(zip(unique, counts))}")
    return X, y
