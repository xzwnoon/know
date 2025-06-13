# kg_graph_viz.py
"""
自动化生成单文献"成分-工艺-性能"关系知识图谱图片
- 支持从PDF文件直接提取信息
- 支持从Neo4j数据库获取已有信息
- 使用DeepSeek LLM进行知识抽取
- 生成可视化知识图谱
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import json
import pdfplumber
import requests
from typing import Dict, List, Any, Optional
import logging
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import networkx as nx
from src.neo4j_manager import Neo4jManager
import re
from src import utils

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  
DEEPSEEK_API_KEY = "sk-0bb6e4dae7ed44b58a8585b363c95dfb"  # 请替换为实际的API密钥
SAVE_DIR = r"C:\Users\Xiang\Desktop\workflow\know\data\kg_data"
PDF_DIR = r"C:\Users\Xiang\Desktop\knowledge paper"  # PDF文件目录

def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    从PDF文件中提取文本和表格内容
    
    Args:
        pdf_path: PDF文件路径
    Returns:
        包含文本和表格内容的字典
    """
    content = {
        "text": "",
        "tables": []
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 提取文本
                content["text"] += page.extract_text() + "\n"
                
                # 提取表格
                tables = page.extract_tables()
                if tables:
                    content["tables"].extend(tables)
                    
        return content
    except Exception as e:
        logger.error(f"PDF提取错误: {str(e)}")
        return content

def get_llm_response(prompt: str) -> Dict:
    """
    调用DeepSeek LLM API进行知识抽取
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "max_tokens": 2000
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        print(f"API响应状态码: {response.status_code}")  # 调试信息
        print(f"API响应内容: {response.text}")  # 调试信息
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"LLM原始响应: {response_data}")
        if 'choices' in response_data and len(response_data['choices']) > 0:
            content = response_data['choices'][0].get('message', {}).get('content', '{}')
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {e}, 原始内容: {content}")
                return {}
        return {}
    except Exception as e:
        logger.error(f"LLM API调用错误: {str(e)}")
        return {"error": str(e)}

def get_llm_response_with_retry(prompt: str, max_retries: int = 3) -> Dict:
    """
    调用DeepSeek LLM API进行知识抽取，支持重试机制
    """
    for attempt in range(max_retries):
        try:
            response = get_llm_response(prompt)
            if response and not response.get("error"):
                return response
        except Exception as e:
            logger.error(f"LLM API调用错误: {str(e)}，尝试次数: {attempt + 1}")
    return {"error": "LLM API调用失败，已重试多次"}

def parse_material_info(content: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    使用LLM解析材料相关信息
    
    Args:
        content: PDF提取的内容
    Returns:
        解析后的材料信息列表
    """
    prompt = f"""请从以下材料科学文献中提取关键信息，格式要求：
    1. 材料成分：包括合金主元素及其含量
    2. 热处理工艺：包括具体温度、时间、介质等参数
    3. 材料性能：包括力学性能（
2. 制备工艺
3. 材料性能
4. 它们之间的关系

文本内容:
{content['text'][:2000]}  # 限制文本长度

请以JSON格式返回,包含以下字段:
{{
    "composition": [{{
        "name": "成分名称",
        "value": "具体数值"
    }}],
    "process": [{{
        "name": "工艺名称",
        "parameters": "工艺参数"
    }}],
    "properties": [{{
        "name": "性能指标",
        "value": "具体数值"
    }}],
    "relationships": [{{
        "from": "起始节点",
        "to": "目标节点",
        "type": "关系类型"
    }}]
}}"""

    # 调用LLM进行解析
    response = get_llm_response(prompt)
    
    # 处理响应
    try:
        if isinstance(response, str):
            response = json.loads(response)
        return response
    except:
        logger.error("LLM响应解析错误")
        return []

def extract_conditions(text):
    """
    从工艺描述中提取温度、时间等实验条件，返回dict
    """
    if not text:
        return {}
    cond = {}
    # 温度（如 170℃、545 °C、190C）
    m = re.search(r'(\d+)[ ]*[°℃C]+', text)
    if m:
        cond['temp'] = int(m.group(1))
    # 时间（如 45min、12 h、30分钟）
    m = re.search(r'(\d+)[ ]*(min|分钟|h|小时)', text, re.I)
    if m:
        cond['time'] = int(m.group(1))
    return cond

def condition_similarity(cond1, cond2):
    """
    计算两个实验条件的相似度，越小越相近
    """
    if not cond1 or not cond2:
        return 9999
    score = 0
    if 'temp' in cond1 and 'temp' in cond2:
        score += abs(cond1['temp'] - cond2['temp'])
    if 'time' in cond1 and 'time' in cond2:
        score += abs(cond1['time'] - cond2['time'])
    return score

def infer_process_order(process_list, custom_keywords=None):
    """
    工艺链顺序推断：
    - 若有Step/order字段，直接用
    - 否则用常见工艺关键词优先级排序
    - 若无关键词，按原顺序
    支持外部自定义keywords
    """
    # 支持外部传入keywords，便于配置化
    keywords = custom_keywords or [
        ('固溶', 1), ('solution', 1), ('SHT', 1), ('溶解', 1), ('dissolution', 1),
        ('淬火', 2), ('quench', 2), ('Q', 2), ('冷却', 2), ('water quench', 2),
        ('时效', 3), ('aging', 3), ('A', 3), ('ageing', 3), ('age', 3),
        ('人工', 3), ('artificial', 3), ('T6', 3), ('T5', 3), ('T4', 3),
        ('自然', 3), ('natural', 3), ('NA', 3), ('室温', 3), ('RT', 3),
        ('双级', 4), ('two-stage', 4), ('double', 4), ('二级', 4), ('T78', 4),
        ('峰值', 5), ('peak', 5), ('peak aging', 5), ('peak-aged', 5),
        ('过时效', 6), ('overaged', 6), ('over-aging', 6),
        ('预时效', 0), ('pre-aging', 0), ('preaged', 0),
        ('回火', 7), ('tempering', 7),
        ('退火', 8), ('annealing', 8),
        ('均匀化', 0), ('homogenization', 0),
        ('冷处理', 9), ('cold treatment', 9),
        ('热处理', 2), ('heat treatment', 2),
        # 可继续扩展...
    ]
    def get_order(proc):
        # 优先用Step/order字段
        for k in ['Step', 'order', 'Order', 'step']:
            if proc.get(k) is not None:
                try:
                    return int(proc[k])
                except Exception:
                    pass
        name = proc.get('Process', '') or ''
        for k, v in keywords:
            if k.lower() in name.lower():
                return v
        # 若Process为空，尝试Property等
        if proc.get('Property'):
            return 99
        return 999  # 未知工艺排最后
    return sorted(process_list, key=get_order)

def draw_kg_for_doc(doc_id: str, pdf_path: Optional[str] = None):
    """
    为指定文献生成知识图谱
    
    Args:
        doc_id: 文献ID
        pdf_path: PDF文件路径(可选)
    """
    logger.info(f"开始为文献 {doc_id} 生成知识图谱")
    print(f"\n===== 调试信息 =====\nPDF路径: {pdf_path}")
    
    # 创建图实例
    G = nx.DiGraph()
    
    # 如果提供了PDF文件,从PDF中提取信息
    material_info = {}
    if pdf_path:
        logger.info(f"正在处理PDF文件: {pdf_path}")
        material_info = extract_and_analyze_pdf(pdf_path)
        print(f"提取的material_info内容:\n{json.dumps(material_info, indent=2, ensure_ascii=False)}")
        if not material_info:
            logger.warning("未能从PDF中提取有效信息")
            return
        if material_info and not isinstance(material_info, dict):
            logger.error(f"LLM返回了无效的数据格式: {type(material_info)}")
            return
        if material_info.get("error"):
            logger.error(f"提取信息时出错: {material_info['error']}")
            return
    
    # 添加成分节点
    composition_details = "\n".join([f"{comp['name']}: {comp['value']}" for comp in material_info.get("composition", [])])
    if composition_details:
        G.add_node("合金成分", type="composition", details=composition_details)
    
    # 添加工艺节点
    process_list = material_info.get("process", [])
    for proc in process_list:
        proc_name = proc["name"]
        proc_params = proc.get("parameters", "未提供")
        G.add_node(proc_name, type="process", parameters=proc_params)
    
    # 添加性能节点
    properties = material_info.get("properties", [])
    for prop in properties:
        pname = prop.get("name", "性能")
        pval = prop.get("value", "")
        label = f"{pname}\n{pval}" if pval else pname
        G.add_node(label, type="property", value=pval)
    
    # 添加关系
    relationships = material_info.get("relationships", [])
    for rel in relationships:
        if rel["from"] in G.nodes and rel["to"] in G.nodes:
            G.add_edge(rel["from"], rel["to"], relationship=rel.get("type", "影响"))
    
    # 自动连接节点（如果关系为空）
    if not relationships:
        # 连接成分到第一个工艺
        if "合金成分" in G.nodes and process_list:
            G.add_edge("合金成分", process_list[0]["name"], relationship="原材料")
        
        # 连接工艺链
        for i in range(len(process_list) - 1):
            from_proc = process_list[i]["name"]
            to_proc = process_list[i+1]["name"]
            G.add_edge(from_proc, to_proc, relationship="工艺流程")
        
        # 连接最后一个工艺到性能
        if process_list and properties:
            last_process = process_list[-1]["name"]
            for prop in properties:
                pname = prop.get("name", "性能")
                pval = prop.get("value", "")
                label = f"{pname}\n{pval}" if pval else pname
                G.add_edge(last_process, label, relationship="影响")
    
    # 设置节点颜色
    node_colors = []
    for node in G.nodes:
        node_type = G.nodes[node].get("type", "")
        if node_type == "composition":
            node_colors.append("lightblue")
        elif node_type == "process":
            node_colors.append("lightgreen")
        elif node_type == "property":
            node_colors.append("lightpink")
        else:
            node_colors.append("gray")

    # 绘制图形
    plt.figure(figsize=(12, 8))
    
    # 使用合适的布局算法
    try:
        if len(G.nodes) > 10:
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        elif len(G.nodes) > 5:
            pos = nx.circular_layout(G)
        else:
            pos = nx.shell_layout(G)
    except Exception:
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, width=1.5, alpha=0.7)
    
    # 优化标签显示
    def node_label(node):
        ntype = G.nodes[node].get("type", "")
        if ntype == "composition":
            return f"{node}\n{G.nodes[node].get('details', '')}"
        elif ntype == "process":
            return f"{node}\n{G.nodes[node].get('parameters', '')}"
        else:
            return node
    
    nx.draw_networkx_labels(G, pos, labels={node: node_label(node) for node in G.nodes}, 
                           font_size=10, font_family='SimHei')
    
    # 添加边标签
    edge_labels = nx.get_edge_attributes(G, "relationship")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    # 添加标题和图例
    plt.title(f"文献知识图谱: {doc_id}", fontsize=14)
    
    # 保存图片
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{doc_id}_kg.png")
    logger.info(f"尝试保存图谱到: {save_path}")
    
    try:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"知识图谱已保存至: {save_path}")
        print(f"图谱保存成功: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"保存图谱失败: {str(e)}")
        print(f"保存图谱失败: {str(e)}")
        return None

def main():
    """
    主函数,处理命令行参数并执行图谱生成
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="生成材料科学文献知识图谱")
    parser.add_argument("doc_id", help="文献ID")
    parser.add_argument("--pdf", help="PDF文件路径", default=None)
    args = parser.parse_args()
    
    # 处理PDF路径
    pdf_path = args.pdf
    if pdf_path:
        if not os.path.isabs(pdf_path):
            # 尝试不同的文件扩展名组合
            possible_paths = [
                os.path.join(PDF_DIR, pdf_path),
                os.path.join(PDF_DIR, f"{pdf_path}.pdf"),
                os.path.join(PDF_DIR, f"{pdf_path}.PDF")
            ]
            
            found_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_path = path
                    logger.info(f"找到PDF文件: {path}")
                    break
                    
            if not found_path:
                logger.error(f"错误：找不到PDF文件，尝试过以下路径:\n" + "\n".join(possible_paths))
                return
                
            pdf_path = found_path
        else:
            if not os.path.exists(pdf_path):
                logger.error(f"错误：找不到PDF文件 '{pdf_path}'")
                return
    
    # 生成知识图谱
    result_path = draw_kg_for_doc(args.doc_id, pdf_path)
    
    if result_path:
        print(f"知识图谱生成成功: {result_path}")
    else:
        print("知识图谱生成失败")

def extract_and_analyze_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    优化：严格分步抽取，保证链条起点为合金成分，依次为工艺、组织/第二相、腐蚀/性能，并梳理链式关系。
    增加LLM返回类型和字段名兼容性处理，防止类型错误。
    """
    logger.info(f"开始提取和分析PDF: {pdf_path}")

    # 提取PDF文本和表格
    content = extract_text_from_pdf(pdf_path)
    if not content or not content.get("text"):
        logger.warning("PDF文本提取失败或为空")
        return {}
    
    # 保存提取的原始内容到临时文件
    os.makedirs(SAVE_DIR, exist_ok=True)
    temp_path = os.path.join(SAVE_DIR, f"{os.path.basename(pdf_path)}_extracted.txt")
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(content['text'])
        if content.get('tables'):
            f.write("\n\n=== 表格内容 ===\n")
            for i, table in enumerate(content['tables']):
                f.write(f"\n表格{i+1}:\n")
                for row in table:
                    f.write("\t".join(str(cell) for cell in row) + "\n")
    logger.info(f"已保存提取内容到: {temp_path}")

    # 拼接表格内容（如有）
    table_str = ""
    if content.get("tables"):
        for i, table in enumerate(content["tables"]):
            table_str += f"\n表格{i+1}:\n"
            for row in table:
                table_str += "\t".join([str(cell) for cell in row]) + "\n"

    def ensure_list(data, *possible_keys):
        # 兼容LLM返回的list/str/dict/字段名不符等情况，自动兼容单复数
        key_set = set(possible_keys)
        # 自动补全单复数
        for k in list(key_set):
            if k.endswith('s'):
                key_set.add(k[:-1])
            else:
                key_set.add(k + 's')
        # 常见拼写补充
        key_set.update({'composition', 'compositions', 'alloy_composition', 'alloy_compositions',
                        'process', 'processes', 'heat_treatment_process', 'heat_treatment_processes',
                        'relationships', 'causal_chain', 'properties', 'performance'})
        
        # 如果已经是列表直接返回
        if isinstance(data, list):
            return data
        
        # 如果是字符串尝试解析为JSON
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return []
        
        # 如果是字典，查找可能的键
        if isinstance(data, dict):
            for k in key_set:
                if k in data:
                    value = data[k]
                    if isinstance(value, list):
                        return value
                    elif isinstance(value, dict):
                        return [value]
                    elif isinstance(value, str):
                        try:
                            return json.loads(value)
                        except:
                            return [value]
            # 检查整个字典是否是所需结构
            if all(k in data for k in ["from", "to", "type"]):
                return [data]
        
        return []

    # 步骤1：抽取合金成分
    comp_prompt = (
        "请从下述材料文献内容（含表格）中，严格提取所有合金成分及其含量，返回JSON数组，格式如下：\n"
        "[{\"name\": \"成分名称\", \"value\": \"具体数值\"}]\n"
        "例如：[{\"name\": \"Al\", \"value\": \"0.7\"}, {\"name\": \"Mg\", \"value\": \"0.95\"}, {\"name\": \"Si\", \"value\": \"0.35\"}, {\"name\": \"Cu\", \"value\": \"0.35\"}]\n"
        f"正文：\n{content['text'][:1000]}\n{table_str}"
    )
    composition_raw = get_llm_response_with_retry(comp_prompt)
    composition = ensure_list(composition_raw, 'composition', 'alloy_compositions')
    logger.info(f"提取的成分数据: {composition}")

    # 步骤2：抽取热处理工艺
    proc_prompt = (
        "请从下述内容中，严格提取所有热处理工艺，返回JSON数组，格式如下：\n"
        "[{\"name\": \"工艺名称\", \"parameters\": \"工艺参数\"}]\n"
        "例如：[{\"name\": \"固溶处理\", \"parameters\": \"545℃/45min\"}, {\"name\": \"双级时效\", \"parameters\": \"170℃/4h + 190℃/4h\"}]\n"
        f"正文：\n{content['text'][:1200]}\n{table_str}"
    )
    process_raw = get_llm_response_with_retry(proc_prompt)
    process = ensure_list(process_raw, 'process', 'heat_treatment_processes')
    logger.info(f"提取的工艺数据: {process}")

    # 步骤3：抽取工艺链条/实验流程因果关系
    chain_prompt = (
        "请从下述内容中，严格梳理所有热处理工艺的实验流程顺序（即工艺链条/因果关系），返回JSON数组，格式如下：\n"
        "[{\"from\": \"起始工艺名称\", \"to\": \"目标工艺名称\", \"type\": \"实验流程\"}]\n"
        "例如：[{\"from\": \"固溶处理\", \"to\": \"淬火\", \"type\": \"实验流程\"}, {\"from\": \"淬火\", \"to\": \"时效\", \"type\": \"实验流程\"}]\n"
        f"正文：\n{content['text'][:1200]}\n{table_str}"
    )
    chain_raw = get_llm_response_with_retry(chain_prompt)
    relationships = ensure_list(chain_raw, 'relationships', 'causal_chain')
    logger.info(f"提取的关系数据: {relationships}")

    # 步骤4：抽取性能数据
    prop_prompt = (
        "请从下述内容中，严格提取材料性能数据，返回JSON数组，格式如下：\n"
        "[{\"name\": \"性能指标名称\", \"value\": \"具体数值\"}]\n"
        "例如：[{\"name\": \"屈服强度\", \"value\": \"320 MPa\"}, {\"name\": \"抗拉强度\", \"value\": \"350 MPa\"}, {\"name\": \"延伸率\", \"value\": \"12%\"}]\n"
        f"正文：\n{content['text'][:1200]}\n{table_str}"
    )
    prop_raw = get_llm_response_with_retry(prop_prompt)
    properties = ensure_list(prop_raw, 'properties', 'performance')
    logger.info(f"提取的性能数据: {properties}")

    # 推断工艺链条顺序
    process_ordered = infer_process_order(process) if process else []

    # 返回结果
    result = {
        "composition": composition,
        "process": process_ordered,
        "relationships": relationships,
        "properties": properties
    }
    return result

if __name__ == "__main__":
    main()