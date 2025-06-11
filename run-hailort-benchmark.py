import argparse
import subprocess
import pathlib
import re
import sys
import shutil

HEF_ROOT = pathlib.Path("sdk/models/object-detection")
OUTPUT_LOG = "benchmark_summary.log"

ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def clean_ansi(line: str) -> str:
    return ansi_escape.sub('', line)

def find_hef_files(dir: str):
    return sorted(pathlib.Path(dir).glob("*.hef"))

def print_progress(line: str):
    line = clean_ansi(line)
    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    sys.stdout.write(f"\r{line[-terminal_width:]}")
    # sys.stdout.write(":".join("{:02x}".format(ord(c)) for c in line))
    sys.stdout.flush()

def run_benchmark(hef_path: pathlib.Path) -> str:
    cmd = ["hailortcli", "benchmark", "-t 60", str(hef_path)]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
		)
        
        output_lines = []
        
        for line in proc.stdout: # type: ignore
            line = line.strip()
            output_lines.append(line)
            print_progress(line)
            
        proc.wait()
        output = "\n".join(output_lines)

        # Parse Summary Section
        fps_match = re.search(r"FPS\s+\(hw_only\)\s+=\s+([\d.]+)", output)
        latency_match = re.search(r"Latency\s+\(hw\)\s+=\s+([\d.]+)", output)

        fps = fps_match.group(1) if fps_match else "N/A"
        latency = latency_match.group(1) if latency_match else "N/A"

        return f"{hef_path.name}: FPS(hw_only)={fps}, Latency(hw)={latency} ms"

    except subprocess.TimeoutExpired:
        return f"{hef_path.name}: [Timeout during benchmark]"
    except Exception as e:
        return f"{hef_path.name}: [Error: {e}]"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, default="", help="ex: sdk/models/object-detection/yolo/hailo-8-hef")

    args = parser.parse_args()
    
    if (not args.model_path):
        parser.print_help()
        return None
        
    hef_files = find_hef_files(args.model_path)
    
    with open(OUTPUT_LOG, "w") as log_file:
        for hef_path in hef_files:
            print(f"Benchmarking {hef_path.name} ...")
            result = run_benchmark(hef_path)
            print(result)
            log_file.write(result + "\n")
    print(f"\nAll results written to {OUTPUT_LOG}")

if __name__ == "__main__":
    main()
