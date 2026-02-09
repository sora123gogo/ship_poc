import cv2, os
from pathlib import Path

VIDEO = "data/raw/Busiest Day so far - 8 Ships in Port - LIVE Replay [u62fnabcShI].mp4"
OUT_DIR = Path("data/frames")
# ディレクトリを作成（存在しない場合は親ディレクトリも含めて作成）
# - parents=True : 上位ディレクトリが存在しない場合も同時に作成
# - exist_ok=True: 既に存在していてもエラーにしない
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 3) 抽出フレームレート（1秒あたり何枚保存するか）
#    ここでは 3fps = 1秒あたり3枚保存
# =========================
fps_target = 3.0  # 3fps

# =========================
# 4) 動画ファイルを開く
#    VideoCapture は動画をフレーム単位で読み込むためのインタフェース
# =========================
cap = cv2.VideoCapture(VIDEO)
# 元動画のFPS（1秒あたりのフレーム数）を取得
# CAP_PROP_FPS は OpenCV におけるフレームレートの属性ID
fps = cap.get(cv2.CAP_PROP_FPS)

# =========================
# 5) フレーム間隔 step の計算
#    step の意味：stepフレームごとに1枚保存する
#
#    仕組み：
#      - 元動画が30fpsで、1fpsで抽出したい場合
#      - 30フレームごとに1枚保存すればよい
#      - step ≈ fps / fps_target
#
#    round : 四捨五入して最も近い整数にする
#    int(...) : 整数に変換
#    max(1, ...) : step=0になるのを防止
# =========================
step = max(1, int(round(fps / fps_target)))

# =========================
# 6) フレームカウンタと保存枚数
#    idx   : 現在のフレーム番号（0開始）
#    saved : 保存した画像枚数
# =========================

idx = 0
saved = 0

# =========================
# 7) フレームを1枚ずつ読み込み
#    cap.read() で1フレーム取得
#      - ok    : 読み込み成功フラグ（False=終了またはエラー）
#      - frame : 取得した画像（NumPy配列, HxWx3）
# =========================
while True:
    ok, frame = cap.read()
    if not ok:
        # 読み込み失敗（動画終了、またはエラー）
        break
    
    if idx % step == 0:
        # ファイル名を固定長で生成（例：frame_0000123.jpg）
        # → 名前順に並べると時系列順になる
        #
        # 画像を書き込み
        # cv2.imwrite(パス, 画像)
        cv2.imwrite(str(OUT_DIR / f"frame_{idx:07d}.jpg"), frame)
        saved += 1
    idx += 1
# リソース解放
cap.release()
print("saved:", saved)
