import os
import json
import glob
import logging
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

KG_DATA_DIR = os.path.join(os.path.dirname(__file__), 'kg_data')
SUMMARY_DIR = os.path.join(KG_DATA_DIR, 'summary')
os.makedirs(SUMMARY_DIR, exist_ok=True)

logger = logging.getLogger("kg_data_analyzer")
logger.setLevel(logging.INFO)

# 统计实体、关系、错误日志
entity_counter = Counter()
relation_counter = Counter()
error_counter = Counter()
llm_fail_counter = 0
llm_fail_examples = []

entity_files = glob.glob(os.path.join(KG_DATA_DIR, 'entities_*.json'))
relation_files = glob.glob(os.path.join(KG_DATA_DIR, 'relations_*.json'))
error_files = glob.glob(os.path.join(KG_DATA_DIR, 'error_*.log'))

# 统计实体
for ef in entity_files:
    try:
        with open(ef, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for ent in data:
                entity_counter[ent.get('type', 'Unknown')] += 1
    except Exception as e:
        logger.warning(f"实体文件 {ef} 解析失败: {e}")

# 统计关系
for rf in relation_files:
    try:
        with open(rf, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for rel in data:
                relation_counter[rel.get('relation_type', 'Unknown')] += 1
    except Exception as e:
        logger.warning(f"关系文件 {rf} 解析失败: {e}")

# 统计错误日志
for errf in error_files:
    try:
        with open(errf, 'r', encoding='utf-8') as f:
            for line in f:
                if 'LLM调用失败' in line or 'LLM' in line and '失败' in line:
                    llm_fail_counter += 1
                    llm_fail_examples.append(line.strip())
                error_counter[line.split(':')[0].strip()] += 1
    except Exception as e:
        logger.warning(f"错误日志文件 {errf} 解析失败: {e}")

# 结构化 summary 输出
summary = {
    'entity_type_count': dict(entity_counter),
    'relation_type_count': dict(relation_counter),
    'error_type_count': dict(error_counter),
    'llm_fail_count': llm_fail_counter,
    'llm_fail_examples': llm_fail_examples[:10],
    'entity_files': entity_files,
    'relation_files': relation_files,
    'error_files': error_files
}
with open(os.path.join(SUMMARY_DIR, 'kg_data_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# 可视化
if entity_counter:
    plt.figure(figsize=(8,4))
    plt.bar(entity_counter.keys(), entity_counter.values())
    plt.title('实体类型分布')
    plt.ylabel('数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, 'entity_type_dist.png'))
    plt.close()
if relation_counter:
    plt.figure(figsize=(8,4))
    plt.bar(relation_counter.keys(), relation_counter.values())
    plt.title('关系类型分布')
    plt.ylabel('数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, 'relation_type_dist.png'))
    plt.close()

# 边界/异常检测
warnings = []
if sum(entity_counter.values()) < 5:
    warnings.append('实体总数极少，可能抽取异常！')
if sum(entity_counter.values()) > 10000:
    warnings.append('实体总数极多，需检查是否重复或异常！')
if sum(relation_counter.values()) < 2:
    warnings.append('关系总数极少，可能抽取异常！')
if sum(relation_counter.values()) > 10000:
    warnings.append('关系总数极多，需检查是否重复或异常！')
if llm_fail_counter > 0:
    warnings.append(f'LLM调用失败次数: {llm_fail_counter}，请检查 last_re_prompt.txt 和 last_re_llm_response.txt')

if warnings:
    logger.warning('检测到异常/边界情况：')
    for w in warnings:
        logger.warning(w)
    with open(os.path.join(SUMMARY_DIR, 'warnings.txt'), 'w', encoding='utf-8') as f:
        for w in warnings:
            f.write(w + '\n')

print('kg_data 统计与可视化已完成，结果见 summary 目录。')
