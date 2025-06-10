#!/usr/bin/env python3
import os
import cv2
import time
import json
import numpy as np
import logging
import asyncio
import threading
import requests
import mediapipe as mp
from fastdtw import fastdtw
from awscrt import io, mqtt
from awsiot import mqtt_connection_builder

from pose_detect import (
    load_pose_data,
    compute_joint_amplitudes,
    compute_joint_weights,
    make_frame_dist,
    compute_seq_movement_vec,
    get_live_pose,
    smooth_sequence,
    bisect_left,
    bisect_right
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

detect_task  = None
detect_thread = None

CONFIG_PATH = 'config.json'
try:
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
        BROKER = cfg.get('broker', 'localhost')
        PORT   = cfg.get('port', 1883)
        RSPID   = cfg.get('rspId', "None")
        CERT = cfg.get('cert_filepath')
        PRIKEY = cfg.get('pri_key_filepath')
        CA = cfg.get('ca_filepath')
except FileNotFoundError:
    raise SystemExit(f"설정 파일 '{CONFIG_PATH}'을(를) 찾을 수 없습니다.")
except (json.JSONDecodeError, TypeError) as e:
    raise SystemExit(f"설정 파일 파싱 오류: {e}")

# 구독할 토픽 (move/start/12345 하나만)
TOPICS    = {"move_start":'move/start/'+RSPID,"move_end":'move/end/'+RSPID,"response_move_start":'response/move/start/'+RSPID}

# client.publish(f'move/end/'+rsp_id, str(cal * ratio)+','+userId, qos=1)
# client.publish('response/move/start/'+rsp_id, '1', qos=1)

# 고유 클라이언트 ID 생성
CLIENT_ID = f'rsp-{RSPID}'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_json_from_url(url: str, output_path: str) -> dict:
    """
    Download JSON file from a public or presigned S3 URL using HTTP GET.
    Saves the JSON to output_path and returns the parsed data.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def detect_pose(data,rsp_id):
    global mqtt_connection
    print(f"💡 Async test start with: {data}")
    # 예시로 2초 대기
    # await asyncio.sleep(5)
    move_num, url, userId = data.split(',')
    # pos_data/ 에 move_num.json 존재하는지 확인
    if not os.path.exists(f'pose_data/{move_num}.json'):
        # 파일이 없으면 다운로드
        #s3 url에 접속하여 pose_data/에 move_num.json으로 저장
        download_json_from_url(url, f'pose_data/{move_num}.json')

#  =================  실제 pose 추정 비교 실행 ===============================
    # --- 초기 로드 & 벡터 변환 ---
    print(f"💡 Loading pose data from {move_num}.json")
    cal,pose_data = load_pose_data(f'pose_data/{move_num}.json')
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

    # result = client.publish('response/move/start/'+rsp_id, '1', qos=1)
    publish_future,packet_id = mqtt_connection.publish(
        topic=TOPICS["response_move_start"],
        payload='1',
        qos=mqtt.QoS.AT_LEAST_ONCE,
        retain=True
    )


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
            time.sleep(0.01)
    finally:
        cap.release()

    # --- WARMUP_MS 제외한 필터링 ---
    filtered = [ok for (t, ok) in results if t >= WARMUP_MS]
    # --- 스무딩 적용 ---
    smoothed = smooth_sequence(filtered, SMOOTH_WINDOW_SIZE)
    # --- 최종 수행 비율 계산 ---
    ratio = sum(smoothed) / len(smoothed) if smoothed else 0.0

    print(len(smoothed),"개")
    print(f"\n초반 {WARMUP_MS/1000:.1f}초 제외 후, "
          f"스무딩 적용된 수행 비율: {ratio:.2%}")

    # result = client.publish(f'move/end/'+rsp_id, str(cal * ratio)+','+userId, qos=1)
    publish_future, packet_id = mqtt_connection.publish(
        topic=TOPICS["move_end"],
        payload=str(cal * ratio)+','+userId,
        qos=mqtt.QoS.AT_LEAST_ONCE,
        retain=True
    )

    print(f"💡 Async detect_pose done")



async_loop = asyncio.new_event_loop()
threading.Thread(target=lambda: async_loop.run_forever(), daemon=True).start()


# 1. Init
event_loop_group  = io.EventLoopGroup(1)
host_resolver     = io.DefaultHostResolver(event_loop_group)
bootstrap         = io.ClientBootstrap(event_loop_group, host_resolver)

def on_connection_interrupted(connection, error, **kwargs):
    print(f"[경고] 연결 중단: {error}")
def on_connection_resumed(connection, return_code, session_present, **kwargs):
    print(f"[정보] 연결 재개 (rc={return_code}, session_present={session_present})")

# 2. Build connection
mqtt_connection = mqtt_connection_builder.mtls_from_path(
    endpoint=BROKER,
    cert_filepath=CERT,
    pri_key_filepath=PRIKEY,
    client_bootstrap=bootstrap,
    ca_filepath=CA,
    client_id="RSP-"+RSPID,
    clean_session=False,
    keep_alive_secs=30,
    reconnect_min_sec=1,     # 최소 1초 후 재시도
    reconnect_max_sec=32,    # 최대 32초 후 재시도
    on_connection_interrupted=on_connection_interrupted,
    on_connection_resumed=on_connection_resumed
)



# 3. Define callback
def on_message_received(topic, payload, **kwargs):
    global detect_thread
    global mqtt_connection
    # 메시지 파싱
    print(f"📥 수신된 메시지: {topic} - {payload}")

    
    try:
        data = json.loads(payload.decode())
    except json.JSONDecodeError:
        data = payload.decode()
    logger.info(f"📥 Received `{data}` on `{payload}`")

    # if detect_task is None or detect_task.done():
    #     logger.info("💡 Async test not running, starting...")

    #     # 3) 비동기 함수 스케줄링
    #     detect_task = asyncio.run_coroutine_threadsafe(detect_pose(mqtt_connection,data,RSPID), async_loop)
    if detect_thread is None or not detect_thread.is_alive():
        logger.info("💡 Starting new detect_pose thread")
        detect_thread = threading.Thread(
            target=detect_pose,
            args=(data, RSPID),
            daemon=True
        )
        detect_thread.start()
    else:
        logger.info("💡 Async test already running, skipping...")
        # result = client.publish(f'response/{msg.topic}', 0, qos=1)
        publish_future,packet_id = mqtt_connection.publish(
            topic=TOPICS["response_move_start"],
            payload='0',
            qos=mqtt.QoS.AT_LEAST_ONCE,
            retain=True
        )
        # try:
        #     publish_future.result()  
        #     # print(f"▶️ publish().result() 반환값: {result!r}")  
        #     # 출력: ▶️ publish().result() 반환값: None
        # except Exception as e:
        #     logger.error(f"[오류] publish() 실패: {e}")

        # if result.rc != mqtt_client.MQTT_ERR_SUCCESS:
        #     logger.error(f"❌ Publish failed (response): rc={result.rc}")

    # print(f"[수신] {topic}: {payload.decode()}")

# 4. Connect
mqtt_connection.connect().result()
print("▶️ 연결됨")

# 5. Subscribe
mqtt_connection.subscribe(
    topic=TOPICS["move_start"],
    qos=mqtt.QoS.AT_MOST_ONCE,
    callback=on_message_received
)
print("▶️ 구독 완료")

# # 6. Publish example
# message = {"deviceId":"myPythonDevice","temperature":24.7}
# mqtt_connection.publish(
#     topic="home/sensor/temperature",
#     payload=json.dumps(message),
#     qos=mqtt.QoS.AT_LEAST_ONCE
# )
# print("▶️ 발행 완료")

# 7. Wait for 수신 or sleep
# import time; time.sleep(5)

# 8. Disconnect
# mqtt_connection.disconnect().result()
# print("▶️ 연결 해제")

# 무한 대기 & 종료 처리
try:
    while True:
        # 자신의 상태를 퍼블리쉬하는 코드
        publish_future,packet_id = mqtt_connection.publish(
            topic=f"clientCheckAlive/rsp/{RSPID}",
            # 현제 시간 포함
            payload='1'+ ','+str(int(time.time()*1000)),
            qos=mqtt.QoS.AT_MOST_ONCE
        )
        time.sleep(0.5)
except KeyboardInterrupt:
    print("종료 요청 감지, 연결 해제 중…")
    mqtt_connection.disconnect()
    print("▶️ 연결 해제 완료, 프로그램 종료")