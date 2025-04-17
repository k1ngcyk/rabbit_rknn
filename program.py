import time
import threading
import subprocess
import csv
import pyaudio
import numpy as np
import librosa
import digitalio
import board
from PIL import Image, ImageDraw, ImageFont
from adafruit_rgb_display import st7789
from rknnlite.api import RKNNLite

########################################
# Global parameters and shared variables
########################################

# Audio stream parameters
CHUNK = 1024  # Samples per frame
FORMAT = pyaudio.paInt16  # 16-bit
CHANNELS = 1  # Mono
RATE = 48000  # Recording sample rate (Hz)

# Model input parameters: mono audio @ 16kHz, 15600 samples
TARGET_SR = 16000  # Model sample rate (Hz)
INFER_INPUT_SIZE = 15600  # Required samples for model
# To gather enough samples at 48kHz:
INFER_BUFFER_SIZE = INFER_INPUT_SIZE * (RATE // TARGET_SR)  # e.g. 15600 * 3 = 46800

# Model and class mapping files
MODEL_PATH = "./1.rknn"
CSV_PATH = "./class_map.csv"

# Shared variables for inference results and system info
last_inference_time_ms = 0.0
last_prediction = "N/A"
last_audio_rms = 0.0
system_info = {"OS": "N/A", "Disk": "N/A", "Temp": "N/A"}

# Flag to signal threads to stop
running = True


########################################
# Load class names from CSV file
########################################
def load_class_map(csv_file):
    with open(csv_file) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return np.array([display_name for (_, _, display_name) in reader])


TAGS = load_class_map(CSV_PATH)

########################################
# Initialize RKNNLite Model (global)
########################################
rknn_lite = RKNNLite()
print("--> Load RKNN model")
ret = rknn_lite.load_rknn(MODEL_PATH)
if ret != 0:
    print("Load RKNN model failed")
    exit(ret)
print("done")

print("--> Init runtime environment")
ret = rknn_lite.init_runtime()
if ret != 0:
    print("Init runtime environment failed!")
    exit(ret)
print("done")

########################################
# Setup PyAudio (global)
########################################
p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=0,  # Adjust device index as needed
    frames_per_buffer=CHUNK,
)

########################################
# Setup Display (SSD1351)
########################################
# Setup display pins (adjust to your wiring)
cs_pin = digitalio.DigitalInOut(board.CS0)
dc_pin = digitalio.DigitalInOut(board.GPIO15)
reset_pin = digitalio.DigitalInOut(board.GPIO13)
# Setup SPI bus using hardware SPI:
spi = board.SPI()
# Initialize display with rotation
disp = st7789.ST7789(spi, cs=cs_pin, dc=dc_pin, rst=reset_pin)

# Determine display dimensions based on rotation
if disp.rotation % 180 == 90:
    height = disp.width  # swap dimensions for rotation
    width = disp.height
else:
    width = disp.width
    height = disp.height

# Load fonts (adjust font path as needed)
fontDir = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
font = ImageFont.truetype(fontDir, 14)
fontB = ImageFont.truetype(fontDir, 11)
fontC = ImageFont.truetype(fontDir, 22)
padding = 6
x = 0

# Load emoticon images and convert to RGB
emo_images = {}
emo_images["silent"] = Image.open("emo_1.png").convert("RGB")
emo_images["music"] = Image.open("emo_2.png").convert("RGB")
emo_images["animal"] = Image.open("emo_3.png").convert("RGB")
emo_images["speech"] = Image.open("emo_4.png").convert("RGB")
emo_images["whistle"] = Image.open("emo_5.png").convert("RGB")
emo_images["default"] = Image.open("emo_6.png").convert("RGB")


########################################
# Thread: Audio Capture & Inference
########################################
def audio_inference_worker():
    global last_inference_time_ms, last_prediction, last_audio_rms, running
    audio_buffer = np.array([], dtype=np.int16)
    while running:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
        except Exception as e:
            print("Audio read error:", e)
            continue
        samples = np.frombuffer(data, dtype=np.int16)
        # Update current audio level (RMS)
        last_audio_rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        # Append new samples to buffer
        audio_buffer = np.concatenate((audio_buffer, samples))

        # When enough samples are accumulated for an inference:
        if len(audio_buffer) >= INFER_BUFFER_SIZE:
            infer_chunk = audio_buffer[:INFER_BUFFER_SIZE]
            audio_buffer = audio_buffer[INFER_BUFFER_SIZE:]
            # Normalize and downsample from 48kHz to 16kHz
            infer_chunk_float = infer_chunk.astype(np.float32) / 32768.0
            infer_chunk_resampled = librosa.resample(
                infer_chunk_float, orig_sr=RATE, target_sr=TARGET_SR
            )
            # Ensure the resampled audio has exactly INFER_INPUT_SIZE samples
            if len(infer_chunk_resampled) != INFER_INPUT_SIZE:
                if len(infer_chunk_resampled) > INFER_INPUT_SIZE:
                    infer_chunk_resampled = infer_chunk_resampled[:INFER_INPUT_SIZE]
                else:
                    infer_chunk_resampled = np.pad(
                        infer_chunk_resampled,
                        (0, INFER_INPUT_SIZE - len(infer_chunk_resampled)),
                        mode="constant",
                    )
            st = time.time()
            output = rknn_lite.inference(inputs=[infer_chunk_resampled])
            en = time.time()
            last_inference_time_ms = (en - st) * 1000
            prediction = np.argmax(output[0])
            last_prediction = TAGS[prediction]
            print(
                f"Prediction: {last_prediction}, Inference Time: {last_inference_time_ms:.2f} ms"
            )


########################################
# Thread: System Info Updater (once per second)
########################################
def system_info_worker():
    global system_info, running
    while running:
        try:
            cmd = "cat /etc/os-release | head -n 1 | cut -d '\"' -f 2"
            os_info = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        except Exception:
            os_info = "Unknown OS"
        try:
            cmd = 'df -h | awk \'$NF=="/"{printf "Disk %d/%d GB", $3,$2}\''
            disk_info = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        except Exception:
            disk_info = "Disk N/A"
        try:
            cmd = "cat /sys/class/thermal/thermal_zone0/temp |  awk '{printf \"Temp %.1f ºC\", $(NF-0) / 1000}'"
            temp_info = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        except Exception:
            temp_info = "Temp N/A"
        system_info = {"OS": os_info, "Disk": disk_info, "Temp": temp_info}
        time.sleep(1)


########################################
# Thread: UI Update (20 fps)
########################################
def ui_update_worker():
    global last_inference_time_ms, last_prediction, last_audio_rms, running, system_info
    while running:
        # Create a new blank image
        bg = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(bg)
        draw.rectangle([0, 0, width, height], fill=(0, 0, 0))
        y = padding

        # Draw system info texts
        os_info = system_info.get("OS", "OS: N/A")
        disk_info = system_info.get("Disk", "Disk: N/A")
        temp_info = system_info.get("Temp", "Temp: N/A")
        draw.text((x, y), os_info, font=fontB, fill="#FF6699")
        y += fontB.getbbox(os_info)[3]
        draw.text((x, y), disk_info, font=font, fill="#99FF99")
        y += font.getbbox(disk_info)[3]
        draw.text((x, y), temp_info, font=font, fill="#FFCCCC")
        y += font.getbbox(temp_info)[3] + 5

        # Draw inference info
        inference_text = f"Inference: {last_inference_time_ms:.2f} ms"
        label_text = f"Prediction: {last_prediction}"
        draw.text((x, y), inference_text, font=fontB, fill="#FFFFFF")
        y += fontB.getbbox(inference_text)[3] + 2
        draw.text((x, y), label_text, font=fontB, fill="#FFFF00")
        y += fontB.getbbox(label_text)[3] + 2

        # Show current audio RMS level
        rms_text = f"RMS: {last_audio_rms:.2f}"
        draw.text((x, y), rms_text, font=font, fill="#00FF00")
        y += font.getbbox(rms_text)[3] + 5

        # Determine which emoticon image to display based on last_prediction (case-insensitive, substring check)
        pred_lower = last_prediction.lower()
        if "silent" in pred_lower:
            selected_emo = emo_images["silent"]
        elif "music" in pred_lower:
            selected_emo = emo_images["music"]
        elif "animal" in pred_lower:
            selected_emo = emo_images["animal"]
        elif "speech" in pred_lower:
            selected_emo = emo_images["speech"]
        elif "whistle" in pred_lower:
            selected_emo = emo_images["whistle"]
        else:
            selected_emo = emo_images["default"]

        # Paste the selected emoticon image below the texts
        bg.paste(selected_emo, (x, y))

        # Update the display
        disp.image(bg)
        time.sleep(0.05)  # ~20fps (50ms per frame)


########################################
# Main: Start all threads and run until interrupted
########################################
if __name__ == "__main__":
    try:
        # Create and start threads (set as daemons so they exit when main thread exits)
        audio_thread = threading.Thread(target=audio_inference_worker, daemon=True)
        sysinfo_thread = threading.Thread(target=system_info_worker, daemon=True)
        ui_thread = threading.Thread(target=ui_update_worker, daemon=True)
        audio_thread.start()
        sysinfo_thread.start()
        ui_thread.start()

        print("System running. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)  # Main thread idle loop

    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        # Give threads a moment to exit cleanly
        time.sleep(0.2)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        rknn_lite.release()
