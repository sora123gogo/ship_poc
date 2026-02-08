import cv2, os
from pathlib import Path

VIDEO = "data/raw/Busiest Day so far - 8 Ships in Port - LIVE Replay [u62fnabcShI].mp4"
OUT_DIR = Path("data/frames")
# mkdir(parents=True, exist_ok=True)
# - parents=True: 如果上级目录不存在也一起创建（比如 data/ 也会自动建）
# - exist_ok=True: 目录已经存在也不报错
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 3) 目标抽帧频率：每秒抽几张
#    这里是 1fps = 每秒保存 1 张图片
# =========================
fps_target = 3.0  # 1fps

# =========================
# 4) 打开视频
#    VideoCapture 相当于“把视频文件当成一个可逐帧读取的流”
# =========================
cap = cv2.VideoCapture(VIDEO)
# 读取视频的原始 fps（每秒多少帧）
# CAP_PROP_FPS 是 OpenCV 里代表“帧率”的属性编号
fps = cap.get(cv2.CAP_PROP_FPS)

# =========================
# 5) 计算抽帧步长 step
#    step 的含义：每隔 step 帧保存一帧
#
#    原理：
#      - 如果视频是 30fps，你想要 1fps
#      - 那就是每 30 帧保存 1 帧
#      - step ≈ fps / fps_target
#
#    round：四舍五入得到最接近的整数
#    int(...)：转成整数
#    max(1, ...)：防止出现 step=0（比如 fps_target 比 fps 还大）
# =========================
step = max(1, int(round(fps / fps_target)))

# =========================
# 6) idx: 当前读到第几帧（从 0 开始）
#    saved: 实际保存了多少张图片
# =========================

idx = 0
saved = 0

# =========================
# 7) 逐帧读取
#    cap.read() 每次读一帧
#      - ok: 是否读取成功（False = 到结尾或出错）
#      - frame: 读到的图像（numpy 数组，形状通常是 HxWx3）
# =========================
while True:
    ok, frame = cap.read()
    if not ok:
        # ok 为 False：视频读完了（或者文件打不开/损坏）
        break
    
    if idx % step == 0:
        # 生成文件名：frame_0000000.jpg 这种固定长度
        # 好处：文件按名字排序就等于按时间顺序
        # 写入 jpg
        # cv2.imwrite(路径, 图像)
        cv2.imwrite(str(OUT_DIR / f"frame_{idx:07d}.jpg"), frame)
        saved += 1
    idx += 1

cap.release()
print("saved:", saved)
