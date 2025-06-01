import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import baseline_2d_resnets
import baseline_3d_resnets
from torchvision import transforms
import time

# ===== 설정 =====
VIDEO_PATH = '/workspace/representation-flow-cvpr19/test/test12.mp4'                      # 추론할 비디오 경로
WEIGHTS_PATH = '/workspace/representation-flow-cvpr19/model.pt'     # .pth 모델 가중치 경로
NUM_CLASSES = 7                                   # HMDB51 등 사용한 클래스 수
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = 200                                   # 비디오 프레임 크기
NUM_FRAMES = 16                                    # 사용할 연속 프레임 수

# ===== 전처리 함수 =====
def load_and_preprocess_video(video_path, num_frames=16, size=200):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ⭐ 랜덤 시작 프레임 계산
    if total_frames <= num_frames:
        start_frame = 0
    else:
        start_frame = np.random.randint(0, total_frames - num_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (size, size))
        frame = frame[:, :, ::-1]  # BGR to RGB
        frames.append(frame)
    cap.release()

    # 부족한 프레임은 마지막 프레임 복제
    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames = np.stack(frames, axis=0)  # (T, H, W, C)
    frames = frames.astype(np.float32) / 255.0
    frames = frames.transpose(3, 0, 1, 2)  # -> (C, T, H, W)
    return torch.tensor(frames).unsqueeze(0)  # -> (1, C, T, H, W)

# ===== 모델 정의 및 로드 =====
model = baseline_3d_resnets.resnet50(num_classes=NUM_CLASSES)
state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)

if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=True)  # ✅ strict=True 명시
model = model.to(DEVICE)
model.eval()

# ===== 비디오 불러와서 추론 + 시간 측정 =====
video_tensor = load_and_preprocess_video(VIDEO_PATH, NUM_FRAMES, INPUT_SIZE)
video_tensor = video_tensor.to(DEVICE)

start_time = time.time()
with torch.no_grad():
    outputs = model(video_tensor)
    probs = F.softmax(outputs, dim=1).squeeze()
    pred_class = torch.argmax(probs).item()
end_time = time.time()

# ===== 결과 출력 =====
print(f"✅ Predicted class index: {pred_class}")
print("🔍 Class probabilities:")
for i, prob in enumerate(probs.tolist()):
    print(f"  Class {i}: {prob:.4f}")
print(f"⏱️ Inference time: {end_time - start_time:.3f} seconds")
