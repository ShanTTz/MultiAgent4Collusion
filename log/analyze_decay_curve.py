import re
import ast
import sys
from collections import defaultdict
from datetime import datetime
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def analyze_decay_curve(log_path, bad_agent_ids, num_bins=10):
    post_author_map = {}
    events = [] # 记录所有针对坏帖子的交互事件
    
    # 预定义的防御性情绪
    defensive_emotions = {'skepticism', 'vigilance', 'sarcasm', 'neutral', 'doubt', 'anger'}
    
    # 正则匹配
    # 匹配时间戳和动作: INFO - 2026-04-14 14:40:39,649 - social.agent - Agent ...
    log_pattern = re.compile(r"INFO - (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}.*?Agent (\d+) is performing action: (\w+) with args: (\{.*\})")
    json_feed_pattern = re.compile(r"['\"]post_id['\"]\s*:\s*(\d+).*?['\"](?:user_id|author_id)['\"]\s*:\s*(\d+)")

    print("正在解析日志并提取时间序列数据...")
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 第一遍：构建帖子和作者的映射
            for match in json_feed_pattern.finditer(content):
                post_id = int(match.group(1))
                author_id = int(match.group(2))
                post_author_map[post_id] = author_id

            # 第二遍：按时间顺序提取事件
            f.seek(0)
            for line in f:
                match = log_pattern.search(line)
                if match:
                    time_str = match.group(1)
                    reactor_id = int(match.group(2))
                    action_type = match.group(3)
                    args_str = match.group(4)
                    
                    try:
                        args = ast.literal_eval(args_str)
                        target_post_id = args.get('post_id') or args.get('repost_id') or args.get('quote_id')
                        
                        if target_post_id in post_author_map and post_author_map[target_post_id] in bad_agent_ids:
                            timestamp = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                            
                            if action_type == 'create_comment':
                                emotion = args.get('emotion', 'unknown').lower()
                                is_defensive = 1 if emotion in defensive_emotions else 0
                                events.append({'time': timestamp, 'type': 'comment', 'defensive': is_defensive})
                                
                            elif action_type in ['repost', 'quote_post']:
                                events.append({'time': timestamp, 'type': 'repost'})
                                
                    except (ValueError, SyntaxError):
                        continue
    except FileNotFoundError:
        print(f"找不到文件: {log_path}")
        return

    if not events:
        print("未检测到足够的时间序列数据，无法绘制衰减曲线。")
        return

    # === 按时间窗口 (Bins) 进行切片统计 ===
    events.sort(key=lambda x: x['time'])
    start_time = events[0]['time']
    end_time = events[-1]['time']
    total_duration = (end_time - start_time).total_seconds()
    
    # 如果日志时间极短（如只有几秒），则缩减窗口数量
    if total_duration < num_bins:
        num_bins = max(1, int(total_duration))
        
    bin_duration = total_duration / num_bins if num_bins > 0 else 1
    
    bins_data = [{'time_label': f"T{i+1}", 'defensive_comments': 0, 'total_comments': 0, 'reposts': 0} for i in range(num_bins)]
    
    for event in events:
        # 计算该事件属于哪个时间窗口
        time_diff = (event['time'] - start_time).total_seconds()
        bin_index = int(time_diff // bin_duration)
        if bin_index >= num_bins:
             bin_index = num_bins - 1
             
        if event['type'] == 'comment':
            bins_data[bin_index]['total_comments'] += 1
            bins_data[bin_index]['defensive_comments'] += event['defensive']
        elif event['type'] == 'repost':
            bins_data[bin_index]['reposts'] += 1

    # === 输出终端表格报告 ===
    print("\n" + "="*60)
    print("📉 免疫衰减时间序列报告 (Time-Series Decay Report)")
    print(f"模拟时间跨度: {start_time} 至 {end_time}")
    print("="*60)
    print(f"{'时间段(Window)':<15} | {'防御性情绪占比 (认知防线)':<25} | {'恶意帖子转发数 (行为防线)':<20}")
    print("-" * 60)
    
    x_labels = []
    y_defensive_ratio = []
    y_reposts = []
    
    for b in bins_data:
        ratio = (b['defensive_comments'] / b['total_comments'] * 100) if b['total_comments'] > 0 else 0.0
        x_labels.append(b['time_label'])
        y_defensive_ratio.append(ratio)
        y_reposts.append(b['reposts'])
        
        ratio_str = f"{ratio:.1f}% ({b['defensive_comments']}/{b['total_comments']})"
        print(f"{b['time_label']:<16} | {ratio_str:<26} | {b['reposts']} 次转发")

    # === 绘制并保存曲线图 ===
    if MATPLOTLIB_AVAILABLE:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 绘制认知防线曲线 (百分比)
        color1 = 'tab:blue'
        ax1.set_xlabel('Time Window')
        ax1.set_ylabel('Defensive Emotion Ratio (%)', color=color1)
        ax1.plot(x_labels, y_defensive_ratio, marker='o', color=color1, linewidth=2, label='Cognitive Defense (Ratio)')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(-5, 105)
        
        # 实例化第二个 Y 轴，绘制行为防线曲线 (转发频次)
        ax2 = ax1.twinx()  
        color2 = 'tab:red'
        ax2.set_ylabel('Malicious Reposts (Count)', color=color2)
        ax2.plot(x_labels, y_reposts, marker='x', linestyle='--', color=color2, linewidth=2, label='Behavioral Breach (Reposts)')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 添加图例和标题
        plt.title('Immunization Decay Curve Over Time')
        fig.tight_layout()
        
        save_path = "immunization_decay_curve.png"
        plt.savefig(save_path, dpi=300)
        print(f"\n✅ 成功生成免疫衰减曲线图，已保存为: {save_path}")
        print("提示: 你可以把这个图片直接用在论文或报告中展现时间维度的免疫力变化。")
    else:
        print("\n提示: 如果安装了 matplotlib (pip install matplotlib)，脚本还可以自动生成 .png 格式的折线图。")

if __name__ == "__main__":
    # CURRENT_BAD_AGENTS = {5, 6, 7, 8, 9}
    CURRENT_BAD_AGENTS = set(range(20, 30))
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "/home/administrastor/MultiAgent4Collusion/log/social.agent-2026-04-17_10-33-51.log"
    
    analyze_decay_curve(log_file, CURRENT_BAD_AGENTS)