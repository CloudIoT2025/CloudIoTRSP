# resilient_subscriber_with_heartbeat.py
import os
import cv2
import time
import json
import numpy as np
import logging
import asyncio
import threading
import mediapipe as mp
from paho.mqtt import client as mqtt_client
from fastdtw import fastdtw
# test.py의 함수 불러오기 
from bisect import bisect_left, bisect_right
from scipy.spatial import procrustes
from pose_detect import (
    load_pose_data,
    compute_joint_amplitudes,
    compute_joint_weights,
    make_frame_dist,
    compute_seq_movement_vec,
    get_live_pose,
    smooth_sequence
)


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

# -------------------
# 전역 상태 변수
# async_test 함수에서 이 변수를 업데이트합니다.
latest_async_data = True

# -------------------
# 설정 파일 로드
# -------------------
CONFIG_PATH = 'config.json'
try:
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
        BROKER = cfg.get('broker', 'localhost')
        PORT   = cfg.get('port', 1883)
        RSPID   = cfg.get('rspId', "None")
except FileNotFoundError:
    raise SystemExit(f"설정 파일 '{CONFIG_PATH}'을(를) 찾을 수 없습니다.")
except (json.JSONDecodeError, TypeError) as e:
    raise SystemExit(f"설정 파일 파싱 오류: {e}")

# 구독할 토픽 (move/start/12345 하나만)
TOPICS    = [('move/start/'+RSPID, 0)]
# 고유 클라이언트 ID 생성
CLIENT_ID = f'rsp-{RSPID}'

# 로거 세팅
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# -------------------
# 비동기 함수 정의
# -------------------
async def async_test(client,data,rsp_id):
    global latest_async_data
    print(f"💡 Async test start with: {data}")
    # 예시로 2초 대기
    # await asyncio.sleep(5)
    move_num, url, userId = data.split(',')
    # pos_data/ 에 move_num.json 존재하는지 확인
    if not os.path.exists(f'pose_data/{move_num}.json'):
        # 파일이 없으면 다운로드

        #s3 url에 접속하여 pose_data/에 move_num.json으로 저장
#  ========================================================================= 해야될꺼 ======================================================================================================================
        # 예시로 pose_data.json 을 복사해서 사용
        os.system(f'cp pose_data.json pose_data/{move_num}.json')

#  =================  실제 pose 추정 비교 실행 ===============================
    # --- 초기 로드 & 벡터 변환 ---
    print(f"💡 Loading pose data from {move_num}.json")
    cal,pose_data = load_pose_data(f'pose_data/{move_num}.json')
    print(f"💡 Loaded {len(pose_data)} frames, {len(pose_data[0])} joints")
    ms_list   = list(pose_data.keys())
    amps      = compute_joint_amplitudes(pose_data)
    weights   = compute_joint_weights(amps)
    print(f"💡 Loaded {len(ms_list)} frames, {len(weights)} joints")

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

    result = client.publish('response/move/start/'+rsp_id, '1,'+userId, qos=2)
    if result.rc != mqtt_client.MQTT_ERR_SUCCESS:
        logger.error(f"❌ Publish failed (response): rc={result.rc}")


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


    result = client.publish(f'move/end/'+rsp_id, str(round(cal * ratio/100, 2))+','+userId, qos=2)
    if result.rc != mqtt_client.MQTT_ERR_SUCCESS:
        logger.error(f"❌ Publish failed (response): rc={result.rc}")

    # 전역 변수 업데이트
    latest_async_data = True
    print(f"💡 Async test done, updated latest_async_data: {latest_async_data}")

# 별도 스레드에서 돌아갈 asyncio 이벤트 루프 시작
async_loop = asyncio.new_event_loop()
threading.Thread(target=lambda: async_loop.run_forever(), daemon=True).start()

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("✅ Connected to MQTT Broker %s:%s", BROKER, PORT)
        client.subscribe(TOPICS)
    else:
        logger.warning(f"❌ Connect failed with code {rc}")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        logger.warning(f"⚠️ Unexpected disconnection (rc={rc})")
    else:
        logger.info("🛑 Graceful disconnect")


def on_message(client, userdata, msg):
    global latest_async_data
    # 메시지 파싱
    try:
        data = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        data = msg.payload.decode()
    logger.info(f"📥 Received `{data}` on `{msg.topic}`")

    # 1) 동기 퍼블리시 (기존 응답)
    # result = client.publish(f'responses/{msg.topic}', 1, qos=2)
    # if result.rc != mqtt_client.MQTT_ERR_SUCCESS:
    #     logger.error(f"❌ Publish failed (responses): rc={result.rc}")

    # 2) move/start/RSPID 전용 응답
    if msg.topic == 'move/start/'+RSPID:
        if latest_async_data:
            logger.info("💡 Async test not running, starting...")
            latest_async_data = False

            # 3) 비동기 함수 스케줄링
            asyncio.run_coroutine_threadsafe(async_test(client,data,RSPID), async_loop)
        else:
            logger.info("💡 Async test already running, skipping...")
            result = client.publish(f'response/{msg.topic}', 0, qos=2)
            if result.rc != mqtt_client.MQTT_ERR_SUCCESS:
                logger.error(f"❌ Publish failed (response): rc={result.rc}")


def on_log(client, userdata, level, buf):
    logger.debug(f"MQTT log: {buf}")


def run():
    client = mqtt_client.Client(client_id=CLIENT_ID)
    client.reconnect_delay_set(min_delay=1, max_delay=5)

    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message
    client.on_log        = on_log

    # 초기 연결 시도 (성공할 때까지 5초 간격 재시도)
    while True:
        try:
            client.connect(BROKER, PORT)
            break
        except Exception as e:
            logger.warning(f"⚠️ Initial connection failed: {e}. Retrying in 5s...")
            time.sleep(5)

    # 네트워크 루프 백그라운드 실행
    client.loop_start()

    # 메인 루프: 하트비트 발행 + 대기
    try:
        while True:
            heartbeat = {
                'client_id': CLIENT_ID,
                'timestamp': time.time()
            }
            client.publish('heartbeat', json.dumps(heartbeat), qos=0)
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("⏹️ Stopping subscriber with heartbeat...")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == '__main__':
    run()