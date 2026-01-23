import numpy as np

def load_keel_dat(file_path):
    """
    Parses a KEEL .dat file format.
    Assumes attributes are numeric and the last column is the class.
    
    Returns:
        X (numpy array): Features
        y (numpy array): Labels (encoded as 0 and 1)
    """
    data = []
    labels = []
    label_map = {}
    
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
            
            # Dynamic label encoding: assign numeric IDs based on first appearance
            if label_str not in label_map:
                label_map[label_str] = len(label_map)
            
            data.append(features)
            labels.append(label_map[label_str])
            
    X = np.array(data)
    y = np.array(labels)
    
    # Ensure minority class is encoded as 1, majority class as 0
    unique, counts = np.unique(y, return_counts=True)
    if counts[0] < counts[1]:
        y = 1 - y  # Swap labels if class 0 is the minority
        
    print(f"Dataset Loaded: {file_path} | Shape: {X.shape} | Distribution: {dict(zip(unique, counts))}")
    return X, y