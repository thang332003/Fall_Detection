import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# Scenes
SCENES = ['Coffee_room_01', 'Coffee_room_02', 'Home_01', 'Home_02']

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available, using CPU")

# Pre-trained ResNet
feature_extractor = models.resnet50(pretrained=True).to(device)
feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
feature_extractor.eval()

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def parse_annotation(ann_path, fps=30):
    """Parse .txt annotation file and analyze temporal fall patterns.
    
    Activity codes:
    1. Fall - tình huống ngã
    2. Walk - đi bộ  
    3. Sit down - ngồi xuống
    4. Stand up - đứng dậy
    5. Pick up object - cúi xuống nhặt đồ
    6. Lay down - nằm xuống (có ý, không phải ngã)
    7. Idle / Standing still - đứng yên
    8. Miscellaneous daily activities - hoạt động sinh hoạt khác
    
    Logic:
    - Code 1: Direct fall detection  
    - Code 6: Lying down after fall motion within 10s = fall, otherwise not fall
    """
    with open(ann_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    print(f"Content of {ann_path}: {lines[:5]}...")  # Debug
    
    # Parse frame annotations first
    frame_anns = {}
    fall_start, fall_end = None, None
    
    # Handle files with start/end frames at beginning
    numeric_lines = []
    for i, line in enumerate(lines):
        if line.isdigit():
            numeric_lines.append(int(line))
        else:
            break
    
    # Parse frame-by-frame annotations
    start_idx = len(numeric_lines)
    for line in lines[start_idx:]:
        parts = line.split(',')
        if len(parts) == 6:
            frame = int(parts[0])
            code = int(parts[1])
            x1, y1, x2, y2 = map(int, parts[2:])
            bb = (x1, y1, x2, y2) if x1 + x2 + y1 + y2 > 0 else None
            frame_anns[frame] = {'code': code, 'bb': bb}
    
    # Analyze temporal patterns for fall detection
    fall_labels = analyze_fall_temporal_pattern(frame_anns, fps)
    
    # Update frame_anns with fall labels
    for frame in frame_anns:
        frame_anns[frame]['is_fall'] = fall_labels.get(frame, 0)
    
    # Find overall fall range for compatibility
    fall_frames = [f for f, label in fall_labels.items() if label == 1]
    if fall_frames:
        fall_start = min(fall_frames)
        fall_end = max(fall_frames)
    
    return fall_start, fall_end, frame_anns

def analyze_fall_temporal_pattern(frame_anns, fps=30):
    """Chỉ code = 7 hoặc 8 là fall, các code 1-6 là non-fall."""
    fall_labels = {}
    for frame in frame_anns:
        code = frame_anns[frame]['code']
        if code in [7, 8]:
            fall_labels[frame] = 1
        else:
            fall_labels[frame] = 0
    return fall_labels

def extract_features(video_path, fall_start, fall_end, frame_anns, max_frames=200, batch_size=32):
    """Extract features in batches on GPU, using temporal fall analysis."""
    cap = cv2.VideoCapture(video_path)
    features = []
    labels = []
    frame_count = 0
    batch_frames = []
    batch_labels = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        ann = frame_anns.get(frame_count, {'bb': None, 'is_fall': 0})
        if ann['bb'] is None:
            continue
        x1, y1, x2, y2 = ann['bb']
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Use the temporal fall analysis result instead of simple range check
        label = ann.get('is_fall', 0)
        
        batch_frames.append(crop)
        batch_labels.append(label)
        
        if len(batch_frames) >= batch_size:
            batch_tensor = torch.stack([transform(f).to(device) for f in batch_frames])
            with torch.no_grad():
                batch_feats = feature_extractor(batch_tensor).squeeze().cpu().numpy()  # Ensure 2048-dim
            features.extend(batch_feats)
            labels.extend(batch_labels)
            batch_frames = []
            batch_labels = []
    
    if batch_frames:
        batch_tensor = torch.stack([transform(f).to(device) for f in batch_frames])
        with torch.no_grad():
            batch_feats = feature_extractor(batch_tensor).squeeze().cpu().numpy()
        features.extend(batch_feats)
        labels.extend(batch_labels)
    
    cap.release()
    return np.array(features), np.array(labels)

def load_raw_data():
    """Load and extract features from all video files."""
    all_features = []
    all_labels = []
    
    for scene in SCENES:
        scene_dir = scene  # Scene folders are directly in the workspace
        if not os.path.exists(scene_dir):
            print(f"Warning: Scene directory {scene_dir} not found, skipping")
            continue
        
        # Handle different annotation folder names
        ann_dir = os.path.join(scene_dir, 'Annotation_files')
        if not os.path.exists(ann_dir):
            ann_dir = os.path.join(scene_dir, 'Annotations_files')  # Coffee_room_02 uses this name
        video_dir = os.path.join(scene_dir, 'Videos')
        if not os.path.exists(ann_dir) or not os.path.exists(video_dir):
            print(f"Warning: Annotation_files or Videos not found in {scene_dir}, skipping")
            continue
        for ann_file in os.listdir(ann_dir):
            if ann_file.endswith('.txt'):
                video_name = ann_file.replace('.txt', '')
                ann_path = os.path.join(ann_dir, ann_file)
                video_path = os.path.join(video_dir, video_name + '.avi')
                if not os.path.exists(video_path):
                    print(f"Warning: Video {video_path} not found")
                    continue
                fall_start, fall_end, frame_anns = parse_annotation(ann_path)
                feats, frame_labels = extract_features(video_path, fall_start, fall_end, frame_anns)
                all_features.append(feats)
                all_labels.append(frame_labels)
    
    return all_features, all_labels

if __name__ == "__main__":
    print("Loading raw data...")
    features, labels = load_raw_data()
    print(f"Loaded {len(features)} video files")
    
    # Save raw data as lists (not arrays) to handle different shapes
    np.save('raw_features.npy', np.array(features, dtype=object), allow_pickle=True)
    np.save('raw_labels.npy', np.array(labels, dtype=object), allow_pickle=True)
    print("Raw data saved to raw_features.npy and raw_labels.npy")