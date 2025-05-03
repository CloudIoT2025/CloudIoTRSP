import json
import time
from bisect import bisect_left, bisect_right
import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial import procrustes
from fastdtw import fastdtw
from tqdm import tqdm

# — 파라미터 설정 —
JSON_PATH                 = 'pose_data/1.json'
SIMILARITY_THRESHOLD      = 0.7      # 0~1, 이 이상이면 '수행'으로 간주
DELAY_TOLERANCE_MS        = 300      # ms 단위 싱크 허용 오차
PAST_WINDOW_MS            = 2000     # 과거 비교 윈도우 크기 (ms)
FUTURE_WINDOW_MS          = 500      # 미래 비교 윈도우 크기 (ms)
MIN_POSES_FOR_DTW         = 3        # DTW 적용 최소 프레임 수
MIN_MOVEMENT_RATIO        = 0.2      # 참조 대비 라이브 이동량 비율 최소치
STATIC_MOVEMENT_THRESHOLD = 0.05     # 정적 구간 판단 임계 이동량
TOP_K_JOINTS              = 20       # 사용할 상위 관절 개수
WARMUP_MS                 = 5000     # 초반 5초(5000ms) 동안 결과는 최종 계산에서 제외
SMOOTH_WINDOW_SIZE        = 5        # 보정용 윈도우 반경(프레임 수)

# 1) JSON 로드 & 전처리
def load_pose_data(path):
    with open(path, 'r') as f:
        temp = json.load(f)
        raw = temp['pose_data']
        cal = temp['cal']
    data = {}
    # tqbm 을 사용하여 진행상황 보기
    for ms_str, lm_list in raw.items():
        ms = int(ms_str)
        frame = {
            p['id']:(p['x'], p['y'], p['z'])
            for p in lm_list
            if p.get('score', 0) > 0.3
        }
        data[ms] = frame
    return cal, dict(sorted(data.items()))

# 2) 관절별 이동량(amplitude) 계산
def compute_joint_amplitudes(pose_data):
    ms_list = list(pose_data.keys())
    joint_ids = set().union(*pose_data.values())
    amp = {j:0.0 for j in joint_ids}
    for t0, t1 in zip(ms_list, ms_list[1:]):
        f0, f1 = pose_data[t0], pose_data[t1]
        for j in joint_ids:
            if j in f0 and j in f1:
                amp[j] += np.linalg.norm(np.array(f1[j]) - np.array(f0[j]))
    return amp

# 3) 가중치 계산 (amplitude → ratio)
def compute_joint_weights(amplitudes):
    total = sum(amplitudes.values()) or 1.0
    return {j: amplitudes[j]/total for j in amplitudes}

# 4) Mediapipe로 라이브 프레임 pose dict 추출
def get_live_pose(pose, frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)
    if not res.pose_landmarks:
        return None
    d = {}
    for idx, lm in enumerate(res.pose_landmarks.landmark):
        if lm.visibility < 0.5:
            continue
        d[idx] = (lm.x, lm.y, lm.z)
    return d

# 5) Procrustes 거리 함수 (벡터화된 두 프레임)
def make_frame_dist(weights_sqrt):
    def frame_dist(A, B):
        A2, B2, _ = procrustes(A * weights_sqrt, B * weights_sqrt)
        return np.sum((A2 - B2)**2)
    return frame_dist

# 6) 시퀀스 내 총 이동량 계산 (벡터 시퀀스)
def compute_seq_movement_vec(seq, weights_vec):
    mv = 0.0
    for v0, v1 in zip(seq, seq[1:]):
        mv += np.sum(weights_vec * np.linalg.norm(v1 - v0, axis=1))
    return mv

# 7) 보정용 시퀀스 스무딩 (median-like majority filter)
def smooth_sequence(seq, k):
    n = len(seq)
    sm = []
    for i in range(n):
        start = max(0, i - k)
        end   = min(n, i + k + 1)
        window = seq[start:end]
        # 다수결: True(1) 개수와 False(0) 개수 비교
        sm.append(True if sum(window) >= (len(window) - sum(window)) else False)
    return sm

# 8) 전체 실행
def run_realtime_evaluation(path):
    # --- 초기 로드 & 벡터 변환 ---
    cal,pose_data = load_pose_data(path)
    ms_list   = list(pose_data.keys())
    amps      = compute_joint_amplitudes(pose_data)
    weights   = compute_joint_weights(amps)

    # 상위 K개 관절 선택
    top_joints = sorted(amps, key=lambda j: amps[j], reverse=True)[:TOP_K_JOINTS]

    # pose_data → 벡터 시퀀스 (각 frame: (K,3) 배열)
    pose_vecs = [
        np.array([ pose_data[t].get(j, (0,0,0)) for j in top_joints ], dtype=float)
        for t in ms_list
    ]

    # joint weights → 벡터, sqrt
    w_vec  = np.array([ weights[j] for j in top_joints ], dtype=float)
    w_sqrt = np.sqrt(w_vec)[:, None]   # (K,1) 모양

    frame_dist = make_frame_dist(w_sqrt)

    # 카메라 & Mediapipe 세팅
    cap     = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    start_ms   = int(time.time()*1000)
    live_vecs  = []
    live_times = []
    results    = []  # (time_ms, ok_flag) 기록

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now_ms = int(time.time()*1000) - start_ms
            # 종료 조건
            if now_ms > ms_list[-1] + DELAY_TOLERANCE_MS:
                break

            lp = get_live_pose(pose, frame)
            if lp is None:
                continue

            # live_pose → 벡터 변환
            live_v = np.array([ lp.get(j, (0,0,0)) for j in top_joints ], dtype=float)
            live_vecs.append(live_v)
            live_times.append(now_ms)

            # 과거 윈도우 유지
            while live_times and live_times[0] < now_ms - PAST_WINDOW_MS:
                live_times.pop(0)
                live_vecs.pop(0)

            # 참조 시퀀스 인덱스 (과거+미래 윈도우)
            ref_start = now_ms - PAST_WINDOW_MS - DELAY_TOLERANCE_MS
            ref_end   = now_ms + FUTURE_WINDOW_MS + DELAY_TOLERANCE_MS
            start_i = bisect_left(ms_list, ref_start)
            end_i   = bisect_right(ms_list, ref_end) - 1
            ref_seq = pose_vecs[start_i:end_i+1]
            live_seq = live_vecs

            if len(ref_seq) < MIN_POSES_FOR_DTW or len(live_seq) < MIN_POSES_FOR_DTW:
                continue

            # DTW 유사도 계산
            dist, _    = fastdtw(ref_seq, live_seq, dist=frame_dist, radius=5)
            avg_dist   = dist / max(len(ref_seq), len(live_seq), 1)
            similarity = max(0.0, 1.0 - avg_dist)

            # 이동량 페널티
            ref_mv  = compute_seq_movement_vec(ref_seq, w_vec)
            live_mv = compute_seq_movement_vec(live_seq, w_vec)
            if ref_mv < STATIC_MOVEMENT_THRESHOLD:
                ok = (similarity >= SIMILARITY_THRESHOLD)
            else:
                move_ratio = live_mv / ref_mv if ref_mv > 0 else 0.0
                ok = (similarity >= SIMILARITY_THRESHOLD and move_ratio >= MIN_MOVEMENT_RATIO)

            # 결과 저장 (나중에 WARMUP_MS 이후만 계산)
            results.append((now_ms, ok))

            # 실시간 로그
            if ref_mv < STATIC_MOVEMENT_THRESHOLD:
                print(f"[{now_ms:4d}ms] static – sim={similarity:.3f} → {'OK' if ok else '✗'}")
            else:
                print(f"[{now_ms:4d}ms] sim={similarity:.3f}, mv_ratio={move_ratio:.3f} → {'OK' if ok else '✗'}")

    finally:
        cap.release()

    # --- WARMUP_MS 제외한 필터링 ---
    filtered = [ok for (t, ok) in results if t >= WARMUP_MS]
    # --- 스무딩 적용 ---
    smoothed = smooth_sequence(filtered, SMOOTH_WINDOW_SIZE)
    # --- 최종 수행 비율 계산 ---
    ratio = sum(smoothed) / len(smoothed) if smoothed else 0.0

    print(f"\n초반 {WARMUP_MS/1000:.1f}초 제외 후, "
          f"스무딩 적용된 수행 비율: {ratio:.2%}")

if __name__ == "__main__":
    run_realtime_evaluation(JSON_PATH)
