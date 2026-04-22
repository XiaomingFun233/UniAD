import re
from collections import defaultdict
from datetime import datetime

iter_per_line = 10  # 每行包含的 iteration 数量
warmup_iters = 50  
warmup_lines = warmup_iters // iter_per_line   # 计算需要跳过的行数


def format_duration(delta):
    total_seconds = int(delta.total_seconds())
    days, remainder = divmod(total_seconds, 24 * 3600)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    day_format = f"{days}天{hours}时{minutes}分{seconds}秒"
    total_hours = total_seconds // 3600
    hms_format = f"{total_hours:02d}:{minutes:02d}:{seconds:02d}"
    return day_format, hms_format

def parse_log(log_path, warmup_lines):
    # 存储每个 epoch 的 time 和 data_time 列表
    epoch_stats = defaultdict(lambda: {"times": [], "data_times": [], "iter_count": 0, "memory": []})
    # 正则匹配：Epoch [epoch][iter/total], ... , time: X, data_time: Y, memory: Z, ...
    pattern = re.compile(
        r"Epoch \[(\d+)\]\[(\d+)/(\d+)\].*?time: ([\d.]+), data_time: ([\d.]+), memory: ([\d.]+),"
    )
    timestamp_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
    first_timestamp = None
    last_timestamp = None
    with open(log_path, "r") as f:
        for line in f:
            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                current_timestamp = datetime.strptime(
                    timestamp_match.group(1), "%Y-%m-%d %H:%M:%S,%f"
                )
                if first_timestamp is None:
                    first_timestamp = current_timestamp
                last_timestamp = current_timestamp

            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                iter_num = int(match.group(2))
                iter_total = int(match.group(3))
                time_val = float(match.group(4))
                data_time_val = float(match.group(5))
                memory_val = float(match.group(6))
                epoch_stats[epoch]["times"].append(time_val)
                epoch_stats[epoch]["data_times"].append(data_time_val)
                epoch_stats[epoch]["memory"].append(memory_val)
                epoch_stats[epoch]["iter_count"] = iter_total  # 记录当前 epoch 的 iteration 数量
    
    # 计算每个 epoch 的平均值（或首值、中位数，按需选择）
    result = {}
    for epoch, stats in epoch_stats.items():
        avg_time = sum(stats["times"][warmup_lines:]) / len(stats["times"][warmup_lines:]) 
        avg_data_time = sum(stats["data_times"][warmup_lines:]) / len(stats["data_times"][warmup_lines:])
        avg_memory = sum(stats["memory"][warmup_lines:]) / len(stats["memory"][warmup_lines:])
        max_memory = max(stats["memory"][warmup_lines:])
        # 记录首步耗时（第一个 iteration 的 time/data_time）
        first_time = stats["times"][0] if stats["times"] else None
        first_data_time = stats["data_times"][0] if stats["data_times"] else None
        iter_count = stats["iter_count"]
        result[epoch] = {
            "avg_time": avg_time,
            "avg_data_time": avg_data_time,
            "first_time": first_time,
            "first_data_time": first_data_time,
            "iter_count": iter_count,
            "avg_memory": avg_memory,
            "max_memory": max_memory
        }

    total_runtime = None
    if first_timestamp is not None and last_timestamp is not None:
        total_runtime = last_timestamp - first_timestamp

    return result, total_runtime

if __name__ == "__main__":
    log_path = "/home/UniAD/projects/work_dirs/stage1_track_map/base_track_map/20260414_095651.log" 
    # 第一阶段的log目录是：   "\home\UniAD\projects\work_dirs\stage1_track_map\base_track_map\20260414_095651.log"
    stats, total_runtime = parse_log(log_path, warmup_lines)
    
    # 打印每个 epoch 的统计结果
    for epoch, data in stats.items():
        print(f"Epoch {epoch}:")
        print(f"  Warm-up Iters: {warmup_iters} (skipped {warmup_lines} lines)")
        print(f"  Avg Time (stable): {data['avg_time']:.4f}s, Avg Data Time (stable): {data['avg_data_time']:.4f}s")
        print(f"  First Time: {data['first_time']:.4f}s, First Data Time: {data['first_data_time']:.4f}s")
        print(f"  Avg Memory (stable): {data['avg_memory']:.2f}MB, Max Memory: {data['max_memory']:.2f}MB")
        print(f"  Iter Count: {data['iter_count']}")
        
        print("-" * 70)

    if total_runtime is not None:
        day_format, hms_format = format_duration(total_runtime)
        print("Total Runtime:")
        print(f"  {day_format}")
        print(f"  {hms_format}")