import re
import json
from collections import defaultdict

def analyze_social_agent_log(log_path, bad_agent_ids):
    # 初始化统计容器
    action_counts = defaultdict(int)
    emotion_counts = defaultdict(int)
    # 按照帖子类型区分立场统计
    stance_counts = {
        '对坏帖子 (Bad Posts)': {'Agree': 0, 'Disagree': 0},
        '对好帖子 (Good Posts)': {'Agree': 0, 'Disagree': 0},
        '对未知帖子 (Unknown Posts)': {'Agree': 0, 'Disagree': 0}
    }
    post_interactions = defaultdict(lambda: defaultdict(int))
    mute_counts = defaultdict(int) # 统计谁被 mute 了多少次
    post_author_map = {} # 映射 post_id -> author_id
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"找不到文件: {log_path}")
        return

    # ==========================================
    # 第一步：提取 post_id 与 author_id 的映射
    # ==========================================
    # 策略 1: "post_id": 123, ... "user_id": 45 
    for match in re.finditer(r'"post_id"\s*:\s*(\d+)[^}]*?"(?:user_id|author_id)"\s*:\s*(\d+)', content):
        post_author_map[int(match.group(1))] = int(match.group(2))
        
    # 策略 2: "user_id": 45, ... "post_id": 123
    for match in re.finditer(r'"(?:user_id|author_id)"\s*:\s*(\d+)[^}]*?"post_id"\s*:\s*(\d+)', content):
        post_author_map[int(match.group(2))] = int(match.group(1))

    # 策略 3: 纯文本匹配 (如 "Post 124 by Agent 96")
    for match in re.finditer(r'[Pp]ost\s+(\d+)\s+(?:by|from)\s+(?:[Aa]gent|user)?\s*(\d+)', content):
        post_author_map[int(match.group(1))] = int(match.group(2))

    # ==========================================
    # 第二步：基于游动上下文提取动作、情绪与立场
    # ==========================================
    # 匹配大模型输出的 JSON actions
    action_matches = re.finditer(r'"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*\{(.*?)\}', content, re.DOTALL)
    
    for match in action_matches:
        action_type = match.group(1)
        args_str = match.group(2)
        
        # 1. 统计动作总数
        action_counts[action_type] += 1
        
        # 向上回溯最多 3000 个字符，寻找这是哪个 Agent 做出的动作
        preceding = content[max(0, match.start() - 3000):match.start()]
        agents = re.findall(r'Agent\s+(\d+)', preceding)
        if not agents:
            continue
        
        # 解析参数字符串中的关键字段
        # 提取 post_id
        pid_match = re.search(r'"(?:post_id|repost_id|quote_id)"\s*:\s*(\d+)', args_str)
        post_id = int(pid_match.group(1)) if pid_match else None
        
        # 2. 统计针对特定帖子的交互
        if post_id is not None:
            post_interactions[post_id][action_type] += 1
            
        # 3. 解析评论中的“情绪”和“立场”
        if action_type == 'create_comment':
            # 提取 emotion
            emo_match = re.search(r'"emotion"\s*:\s*"([^"]+)"', args_str)
            if emo_match:
                emotion = emo_match.group(1).lower()
                emotion_counts[emotion] += 1
                
            # 提取 agree (boolean)
            agree_match = re.search(r'"agree"\s*:\s*(true|false)', args_str, re.IGNORECASE)
            if agree_match:
                agree_val = agree_match.group(1).lower()
                
                # 判定这个帖子属于坏人还是好人
                if post_id in post_author_map:
                    author_id = post_author_map[post_id]
                    post_type = '对坏帖子 (Bad Posts)' if author_id in bad_agent_ids else '对好帖子 (Good Posts)'
                else:
                    post_type = '对未知帖子 (Unknown Posts)'

                if agree_val == 'true':
                    stance_counts[post_type]['Agree'] += 1
                elif agree_val == 'false':
                    stance_counts[post_type]['Disagree'] += 1
                    
        # 4. 统计网络隔离 (Mute)
        if action_type == 'mute':
            mute_match = re.search(r'"mutee_id"\s*:\s*(\d+)', args_str)
            if mute_match:
                mutee_id = int(mute_match.group(1))
                mute_counts[mutee_id] += 1

    # === 打印分析报告 ===
    print("="*50)
    print(f"📊 社交网络模拟日志分析报告 (通用容错解析)")
    print(f"靶向恶意节点: {bad_agent_ids}")
    print(f"文件: {log_path}")
    print("="*50)

    print("\n[1] 动作频率总计 (Action Frequencies):")
    if not action_counts:
        print("  - 未检测到任何动作记录。")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {action}: {count} 次")

    print("\n[2] 评论情绪分布 (Emotion Distribution):")
    total_emotions = sum(emotion_counts.values())
    if total_emotions == 0:
        print("  - 未检测到情绪数据。")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_emotions * 100) if total_emotions > 0 else 0
        print(f"  - {emotion}: {count} 次 ({percentage:.1f}%)")

    print("\n[3] 评论立场分布 (Stance Distribution):")
    has_stance_data = False
    for post_type, counts in stance_counts.items():
        total_stance = counts['Agree'] + counts['Disagree']
        if total_stance > 0:
            has_stance_data = True
            print(f"  👉 {post_type} (总计 {total_stance} 次立场表达):")
            for stance, count in counts.items():
                percentage = (count / total_stance * 100) if total_stance > 0 else 0
                print(f"      - {stance}: {count} 次 ({percentage:.1f}%)")
    
    if not has_stance_data:
        print("  - 未检测到立场数据。")

    print("\n[4] 账号被静音/屏蔽排行 (Top Muted Users):")
    if not mute_counts:
        print("  - 未检测到任何 Mute 行为。")
    for user_id, count in sorted(mute_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - User ID {user_id}: 被 mute 了 {count} 次")

if __name__ == "__main__":
    # 定义坏节点集合以便区分帖子阵营
    CURRENT_BAD_AGENTS = set(range(90, 100))
    log_file = "/home/administrastor/MultiAgent4Collusion/log/social.agent-2026-04-17_10-54-28.log"
    analyze_social_agent_log(log_file, CURRENT_BAD_AGENTS)