import cv2
import mediapipe as mp
import time
import json
import math
from tqdm import tqdm

# 경로 설정
num = 3
input_path = f"videos/{num}.mp4"
output_path = f"videos/{num}_.mp4"
json_path = f"pose_data/{num}.json"
cal = 150
# 비디오 열기
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"영상 열기 실패: {input_path}")

# 프레임 정보 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
tqdm.write(f"🎥 해상도: {width}x{height}, FPS: {fps}, 총 프레임: {frame_count}")

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not out.isOpened():
    cap.release()
    raise IOError("VideoWriter 초기화 실패")

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

# 프레임별 포즈 좌표 저장용 딕셔너리
pose_data = {}
prev_keypoints = None  # 이전 프레임 키포인트 저장


def frame_index_to_timestamp(frame_idx, fps):
    return int((frame_idx / fps) * 1000)


# 처리 시작
t0 = time.time()

with tqdm(total=frame_count, desc="프레임 처리 중", unit="frame") as pbar:
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        keypoints = []
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            for i, lm in enumerate(results.pose_landmarks.landmark):
                x = round(lm.x, 4)
                y = round(lm.y, 4)
                z = round(lm.z, 4)
                score = round(lm.visibility, 4)
                # 이전 프레임과의 차이 계산
                if prev_keypoints is not None and i < len(prev_keypoints):
                    prev = prev_keypoints[i]
                    dx = round(x - prev["x"], 4)
                    dy = round(y - prev["y"], 4)
                    dz = round(z - prev["z"], 4)
                    dist = round(math.sqrt(dx * dx + dy * dy + dz * dz), 4)
                else:
                    dx, dy, dz, dist = 0.0, 0.0, 0.0, 0.0

                keypoints.append(
                    {
                        "id": i,
                        "x": x,
                        "y": y,
                        "z": z,
                        "score": score,
                        "dx": dx,
                        "dy": dy,
                        "dz": dz,
                        "dist": dist,
                    }
                )

        # 결과 저장
        timestamp = frame_index_to_timestamp(frame_idx, fps)
        pose_data[timestamp] = keypoints
        prev_keypoints = keypoints

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

# 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()

# pose_data 중에 빈 값이 있는 경우 제거
pose_data = {k: v for k, v in pose_data.items() if v}

# JSON 저장
with open(json_path, "w") as f:
    json.dump({"cal": cal, "pose_data": pose_data}, f, indent=2)

elapsed = time.time() - t0
tqdm.write(f"✅ 처리 완료! 총 소요 시간: {elapsed:.2f}초")
tqdm.write(f"📦 JSON 저장 완료: {json_path}")
tqdm.write(f"🎬 영상 저장 완료: {output_path}")
