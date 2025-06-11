import argparse
import itertools
import queue
import subprocess
import pathlib
import threading
import time
import sys
import logging
import shutil

from datetime import datetime

OUTPUT_LOG = "benchmark_results.log"
VIDEO_MP4_PATH = "sdk/samples/videos/test.mp4"
TIMEOUT_SECONDS = 180

def find_model_files(dir: str):
    return sorted(itertools.chain(
            pathlib.Path(dir).glob("**/tflite/*.tflite"),
            pathlib.Path(dir).glob("**/onnx/*.onnx")
        )
    )

def print_progress(percent: float, last_line: str, bar_ratio = 0.3):
    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns

    # 建立 progress bar
    bar_len = int(terminal_width * bar_ratio)
    filled_len = int(round(bar_len * percent / 100))
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    progress = f"[{bar}] {percent:5.1f}%"

    # 計算剩餘空間
    remaining_width = terminal_width - len(progress) - 1  # -1 for spacing
    trimmed_line = last_line[-remaining_width:] if remaining_width > 0 else ''
    
    sys.stdout.write(f"\r{progress} {trimmed_line}")
    sys.stdout.flush()

def run_inference(model_path: pathlib.Path) -> str:
    cmd = [
        "python3",
        "./main.py",
        f"--sample-path={VIDEO_MP4_PATH}",
        f"--model-path={model_path}"
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    output_lines = []
    output_queue = queue.Queue()

    def reader_thread():
        for line in proc.stdout:  # type: ignore
            output_queue.put(line)

    thread = threading.Thread(target=reader_thread, daemon=True)
    thread.start()

    start_time = time.time()
    last_line = ""
    try:
        while True:
            try:
                line = output_queue.get()
                line = line.strip()
                output_lines.append(line)
                last_line = line
            except queue.Empty:
                pass

            elapsed = time.time() - start_time
            percent = min(elapsed / TIMEOUT_SECONDS * 100, 100)
            print_progress(percent, last_line)

            if proc.poll() is not None:
                break
            
            if elapsed > TIMEOUT_SECONDS:
                proc.kill()
                print_progress(100, "[Timeout]")
                return f"{model_path.name}: {output_lines[-1]}"

        last_line = output_lines[-1] if output_lines else "[No output]"
        print_progress(100, last_line)
        
        return f"{model_path.name}: {output_lines[-1]}"
    
    except Exception as e:
        proc.kill()
        print_progress(100, f"[Error: {e}]")
        return f"{model_path.name}: [Error: {e}]"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, default="", help="ex: sdk/models/object-detection")
    parser.add_argument("-v", "--sample-video", type=str, default="", help="ex: sdk/samples/videos/test.mp4")

    args = parser.parse_args()
    
    if (not args.model_path or not args.sample_video):
        parser.print_help()
        return None
    
    model_files = find_model_files(args.model_path)
    
    with open(OUTPUT_LOG, "w") as log_file:
        for model_path in model_files:
            print(f"\nRunning {model_path.name}")
            result = run_inference(model_path)
            log_file.write(result + "\n")
        print()  # newline after last progress bar

if __name__ == "__main__":
    main()
