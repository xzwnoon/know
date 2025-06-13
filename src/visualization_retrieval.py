# src/visualization_retrieval.py

import os
import sys
import json
import pprint
from typing import List, Dict, Any, Optional, Tuple
import logging
import argparse

# 配置 visualization_retrieval 的日志记录
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # INFO 级别显示检索和保存进度，DEBUG 更详细

# 导入 Pyvis 库用于图谱可视化
try:
    from pyvis.network import Network
    logger.debug("成功导入 pyvis 库。")
except ImportError:
    logger.error("错误：未安装 'pyvis' 库。无法生成图谱可视化。请安装：pip install pyvis")
    Network = None # 如果导入失败，将 Network 类设为 None

# 导入同目录下的 neo4j_manager 和 utils 模块
import neo4j_manager # 数据库接口
import utils # 通用辅助函数 (如 load_config)


def build_property_html(properties: Dict[str, Any]) -> str:
    """
    将节点的属性字典转换为 HTML 字符串，用于 Pyvis 节点的 'title' 属性。
    当鼠标悬停在节点上时，会显示这些属性信息。

    Args:
        properties: 节点的属性字典（从 Neo4j 查询结果中获取）。

    Returns:
        包含属性信息的 HTML 字符串。
    """
    if not properties:
        return ""

    html = "<strong>属性:</strong><ul>"
    for key, value in properties.items():
        # 简单处理属性值，避免显示过长的字符串或复杂的嵌套结构
        # 将字典或列表属性转换为 JSON 字符串显示一部分
        if isinstance(value, (dict, list)):
             try:
                 # 使用 json.dumps 格式化字典/列表，ensure_ascii=False 支持中文
                 value_str = json.dumps(value, ensure_ascii=False, indent=2)
                 # 限制显示的长度，避免 title 过长
                 value_str_truncated = value_str[:500] + "..." if len(value_str) > 500 else value_str
                 # 使用 <pre> 标签保留 JSON 格式的缩进和换行
                 html += f"<li><strong>{key}:</strong> <pre>{value_str_truncated}</pre></li>"
             except TypeError:
                  # 如果转换失败，显示提示
                  html += f"<li><strong>{key}:</strong> (复杂结构，无法显示)</li>"
        else:
            # 将基本类型（字符串、数字、布尔值等）直接转换为字符串显示
            # 限制字符串长度，避免过长的文本撑开悬停框
            value_str = str(value)
            value_str_truncated = value_str[:100] + "..." if len(value_str) > 100 else value_str
            html += f"<li><strong>{key}:</strong> {value_str_truncated}</li>"

    html += "</ul>"
    return html

def visualize_subgraph(nodes_data: List[Dict], relationships_data: List[Dict], output_filename: str = 'knowledge_graph.html'):
    """
    使用 Pyvis 可视化 Neo4j 查询结果中的节点和关系子图。

    将节点和关系添加到 Pyvis Network 对象，并配置节点/关系的显示样式和悬停信息，
    然后生成一个交互式的 HTML 文件。

    Args:
        nodes_data: 从 Neo4j 查询结果中提取的节点数据列表。每个字典应包含 'elementId', 'labels', 'properties'。
        relationships_data: 从 Neo4j 查询结果中提取的关系数据列表。每个字典应包含 'elementId', 'startNodeElementId', 'endNodeElementId', 'type', 'properties'。
        output_filename: 输出的 HTML 文件名。文件将保存在 data/kg_data/ 目录下。
    """
    # 检查 Pyvis 库是否成功导入
    if Network is None:
        logger.error("Pyvis 库未就绪，无法生成可视化。")
        # 即使无法可视化，也尝试创建一个空的 HTML 文件，表示操作已执行但无内容
        try:
            empty_html_content = """
            <html><body><p>Pyvis 库未安装，无法显示图谱。请安装：<code>pip install pyvis</code></p></body></html>
            """
            output_path = os.path.join('data', 'kg_data', output_filename)
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                 f.write(empty_html_content)
            print(f"图谱可视化失败，生成一个空 HTML 文件表示：{output_path}")
        except Exception as e:
            logger.error(f"无法生成空的 HTML 文件 {output_path}: {e}")
        return


    if not nodes_data and not relationships_data:
        logger.info("没有节点或关系数据可用于可视化。")
        # 如果没有数据，生成一个空的图谱文件
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
        net.set_options("""
        var options = {
          "layout": { "hierarchical": { "enabled": false } },
          "physics": { "enabled": true }
        }
        """)
        output_path = os.path.join('data', 'kg_data', output_filename)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir, exist_ok=True)
        net.save_graph(output_path)
        print(f"没有数据，生成一个空图谱可视化文件：{output_path}")
        logger.info(f"空图谱可视化已保存到文件: {output_path}")
        return


    # 创建一个 Pyvis Network 对象
    # height 和 width 设置图谱在 HTML 页面中的大小
    # bgcolor 和 font_color 设置背景和字体颜色
    # notebook=False 表示生成独立的 HTML 文件，而不是在 Jupyter Notebook 中显示
    # directed=True 表示图谱有方向（对应 Neo4j 的有向关系）
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False, directed=True)

    # 设置一些可视化选项，控制图谱的布局、物理模拟效果等
    # 这里的选项可以根据需要进行调整，以获得更好的视觉效果
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 10
        },
         // 可以根据节点数量调整节点大小的缩放范围
        "scaling": {
          "min": 10,
          "max": 30
        }
      },
      "edges": {
        "font": {
          "size": 8,
          "align": "middle" // 关系标签显示在关系中间
        },
        "smooth": {
          "forceDirection": "none" // 关系的弯曲方向
        },
         "arrows": { // 设置箭头样式
             "to": { "enabled": true, "scaleFactor": 0.7 }
         },
        "color": { // 关系颜色
             "inherit": "from" // 继承自源节点颜色
        }
      },
      "physics": {
        // barnesHut 算法适用于中等大小的图谱，largeGraphMode 适用于更大图谱
        "barnesHut": {
          "gravitationalConstant": -2000, // 节点之间的排斥力
          "centralGravity": 0.1, // 中心引力
          "springLength": 90, // 关系弹簧的自然长度
          "springConstant": 0.08, // 关系弹簧的强度
          "damping": 0.9, // 物理模拟的阻尼，避免振荡
          "avoidOverlap": 0.5 // 避免节点重叠
        },
        "maxVelocity": 30, // 节点最大移动速度
        "minVelocity": 0.5, // 节点最小移动速度，低于此速度认为静止
        "solver": "barnesHut", // 使用的物理模拟算法
        "stabilization": { // 稳定化过程，让图谱在初始布局时达到一个稳定状态
           "enabled": true,
           "iterations": 500, // 稳定化迭代次数
           "updateInterval": 25 // 更新显示间隔
         }
      },
      "interaction": {
         "tooltipDelay": 200, // 悬停显示 tooltip 的延迟时间 (毫秒)
         "hideEdgesOnDrag": true, // 拖动节点时隐藏关系，提高性能
         "hoverConnectedEdges": true // 悬停节点时高亮相关关系
      }
    }
    """) # 可以在这里自定义更复杂的 Pyvis/Vis.js 选项


    # 添加节点到 Pyvis 网络对象
    added_nodes_pyvis_ids = set() # 记录已经添加到 Pyvis 的节点ID (Pyvis内部使用的ID)
    # 创建一个映射表：从 Neo4j 的 elementId 到 Pyvis 内部使用的节点ID
    # 这是因为 Pyvis 添加节点时可以指定 ID，关系需要使用这些 ID 来连接节点
    neo4j_element_id_to_pyvis_id = {}

    # 使用一个计数器为节点生成简单的 Pyvis ID (另一种方式是直接使用 Neo4j 的 elementId 作为 Pyvis ID)
    # 直接使用 elementId 作为 Pyvis ID 更简单且能保证唯一性，推荐这种方式。
    # pyvis_node_counter = 0

    for node_data in nodes_data:
        # 从 Neo4j 节点数据中获取 elementId, labels, properties
        node_neo4j_id = node_data.get('elementId')
        if not node_neo4j_id:
             logger.warning(f"跳过缺少 elementId 的节点数据: {node_data}")
             continue

        # 直接使用 Neo4j 的 elementId 作为 Pyvis 的节点 ID
        pyvis_node_id = node_neo4j_id

        # 检查节点是否已经添加到 Pyvis 网络中（通过其 Pyvis ID）
        if pyvis_node_id in added_nodes_pyvis_ids:
             continue # 已经添加过，跳过

        properties = node_data.get('properties', {})
        # 选择合适的节点显示标签，优先中文名称 'name_zh'，然后是 'text' 属性，最后使用 Neo4j elementId
        label = properties.get('name_zh') or properties.get('text') or node_neo4j_id
        # 对标签进行简单清理，移除特殊字符如换行符，避免显示问题
        label = str(label).replace('\n', ' ')
        label = (label[:50] + '...') if len(label) > 50 else label # 截断过长的标签

        # 根据节点标签（类型）设置颜色、大小或形状等可视化样式
        color = "#ffcc99" # 默认节点颜色 (浅橙色)
        # 根据节点标签映射颜色（根据 Schema 定义的实体类型）
        if node_data.get('labels'):
            # PyMuPDF 返回的标签是 frozenset，需要转换为 list
            labels_list = list(node_data['labels'])
            # 查找主要的标签（排除通用的 'Entity' 标签如果使用了的话）
            main_label = [lbl for lbl in labels_list if lbl != 'Entity']
            main_label = main_label[0] if main_label else (labels_list[0] if labels_list else None) # 如果没有特定标签，使用第一个；如果都没有，为 None

            if main_label == "MaterialAlloy":
                color = "#add8e6" # 浅蓝色
            elif main_label == "HeatTreatmentProcess":
                color = "#90ee90" # 浅绿色
            elif main_label == "MechanicalProperty":
                color = "#ffb6c1" # 浅粉色
            elif main_label == "AlloyComponent":
                color = "#ffff99" # 浅黄色
            elif main_label == "ResearchMethod":
                color = "#dda0dd" # 梅红色
            elif main_label == "Application":
                color = "#ff7f50" # 珊瑚色
            elif main_label == "SourceDocument":
                 color = "#cccccc" # 灰色
            # 可以添加更多标签类型和对应的颜色


        # 构建节点悬停时显示的 HTML 属性信息
        title_html = build_property_html(properties)
        # 在 title 开头加上节点类型信息
        title_html = f"<strong>类型:</strong> {', '.join(node_data.get('labels', []))}<br>" + title_html

        # 将节点添加到 Pyvis 网络中
        net.add_node(pyvis_node_id, label=label, color=color, title=title_html, size=15)
        # 记录已经添加的节点 Pyvis ID
        added_nodes_pyvis_ids.add(pyvis_node_id)
        # 记录 Neo4j elementId 到 Pyvis ID 的映射
        neo4j_element_id_to_pyvis_id[node_neo4j_id] = pyvis_node_id


    # 添加关系到 Pyvis 网络对象
    added_edges_pyvis_ids = set() # 记录已经添加到 Pyvis 的关系ID (Pyvis内部使用的ID)
    # 同样使用 Neo4j 的 elementId 作为 Pyvis 的关系 ID，或者生成简单 ID
    # 直接使用 elementId 更方便关联 Neo4j 数据

    for rel_data in relationships_data:
        # 从 Neo4j 关系数据中获取 elementId, type, properties, startNodeElementId, endNodeElementId
        rel_neo4j_id = rel_data.get('elementId')
        if not rel_neo4j_id:
             logger.warning(f"跳过缺少 elementId 的关系数据: {rel_data}")
             continue

        # 直接使用 Neo4j 的 elementId 作为 Pyvis 的关系 ID
        pyvis_edge_id = rel_neo4j_id

        # 检查关系是否已经添加到 Pyvis 网络中
        if pyvis_edge_id in added_edges_pyvis_ids:
            continue # 已经添加过，跳过

        source_neo4j_id = rel_data.get('startNodeElementId')
        target_neo4j_id = rel_data.get('endNodeElementId')
        rel_type = rel_data.get('type')

        # 确保关系的起始和结束节点已经添加到图中
        # 使用之前建立的映射表查找对应的 Pyvis 节点 ID
        source_pyvis_id = neo4j_element_id_to_pyvis_id.get(source_neo4j_id)
        target_pyvis_id = neo4j_element_id_to_pyvis_id.get(target_neo4j_id)

        if source_pyvis_id is None or target_pyvis_id is None:
            # 如果关系的任何一端节点未在 nodes_data 列表中找到并添加到 Pyvis，则跳过此关系
            logger.warning(f"警告: 关系 '{rel_type}' (ID: {rel_neo4j_id}) 的端点节点未在可视化列表中找到，跳过此关系。源节点ID: {source_neo4j_id}, 目标节点ID: {target_neo4j_id}")
            continue


        properties = rel_data.get('properties', {})

        # 构建关系悬停时显示的 HTML 属性信息
        title_html = build_property_html(properties)
        # 在 title 开头加上关系类型信息
        title_html = f"<strong>类型:</strong> {rel_type}<br>" + title_html

        # 将关系添加到 Pyvis 网络中
        # 使用节点的 Pyvis ID 来定义关系
        net.add_edge(source_pyvis_id, target_pyvis_id, label=rel_type, title=title_html)
        # 记录已经添加的关系 Pyvis ID
        added_edges_pyvis_ids.add(pyvis_edge_id)

    # 生成可视化 HTML 文件
    try:
        # 将文件保存在 data/kg_data/ 目录下
        output_path = os.path.join('data', 'kg_data', output_filename)
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir, exist_ok=True)
             logger.info(f"创建输出目录: {output_dir}")

        net.save_graph(output_path)
        logger.info(f"图谱可视化已保存到文件: {output_path}")
        print(f"图谱可视化已保存到文件: {output_path}") # 同时打印到控制台提示用户
    except Exception as e:
        logger.error(f"保存图谱可视化文件失败 {output_path}: {e}")
        print(f"错误: 保存图谱可视化文件失败 {output_path}: {e}")


def search_and_visualize_kg(manager: neo4j_manager.Neo4jManager, keyword: str, hops: int = 1, output_filename: str = 'search_results_graph.html'):
    """
    执行基于关键词的图谱检索，并可视化检索结果及其指定跳数内的邻居子图。

    通过 Neo4j 数据库查询与关键词匹配的节点，然后从这些节点出发，扩展指定跳数，
    获取一个子图（包含节点和关系），最后使用 visualize_subgraph 函数进行可视化。

    Args:
        manager: 已连接的 Neo4jManager 实例。
        keyword: 用于检索的关键词。将用于匹配节点的 text 或 name_zh/name_en 等属性。
        hops: 从匹配到的节点向外扩展的跳数。
              0 表示只显示匹配的节点（没有关系）。
              1 表示显示匹配节点及其直接相连的节点和连接它们的关系。
              2 表示显示匹配节点、它们的一跳邻居、以及它们到二跳邻居的路径上的所有节点和关系。
        output_filename: 输出的 HTML 文件名。将保存在 data/kg_data/ 目录下。
    """
    # 检查数据库连接是否可用
    if not manager or not manager.driver:
        logger.error("数据库连接不可用，无法执行检索和可视化。")
        # 如果无法连接，生成一个空的 HTML 文件并返回
        visualize_subgraph([], [], output_filename)
        return

    logger.info(f"开始检索关键词: '{keyword}', 扩展跳数: {hops}...")
    print(f"开始检索关键词: '{keyword}', 扩展跳数: {hops}...")


    # Step 1: 查找与关键词匹配的初始节点
    # 使用 Cypher 全文索引查询（效率高，但需要先创建索引）或 LIKE 模糊匹配
    # 优先使用 LIKE 匹配，因为它不需要额外的数据库配置（无需创建索引）
    # 匹配节点的 text, name_zh, name_en, 或 description_zh 属性
    # toLower() 函数用于进行不区分大小写的匹配
    search_query = """
    MATCH (n)
    WHERE toLower(n.text) CONTAINS toLower($keyword)
       OR (n.name_zh IS NOT NULL AND toLower(n.name_zh) CONTAINS toLower($keyword))
       OR (n.name_en IS NOT NULL AND toLower(n.name_en) CONTAINS toLower($keyword))
       OR (n.description_zh IS NOT NULL AND toLower(n.description_zh) CONTAINS toLower($keyword))
    RETURN DISTINCT elementId(n) AS nodeId, n.text AS nodeText, labels(n) AS nodeLabels, properties(n) AS nodeProperties
    """
    search_parameters = {'keyword': keyword}

    # 执行查询
    try:
        initial_nodes_result = manager.query(search_query, parameters=search_parameters)
    except Exception as e:
         logger.error(f"执行关键词匹配查询失败: {e}")
         print(f"错误：执行关键词匹配查询失败: {e}")
         visualize_subgraph([], [], output_filename) # 查询失败，生成空图
         return


    if not initial_nodes_result:
        logger.warning(f"未找到与关键词 '{keyword}' 匹配的任何实体。")
        print(f"未找到与关键词 '{keyword}' 匹配的任何实体。")
        visualize_subgraph([], [], output_filename) # 未找到匹配，生成空图
        return

    logger.info(f"找到 {len(initial_nodes_result)} 个匹配的初始实体。")
    print(f"找到 {len(initial_nodes_result)} 个匹配的初始实体。")

    # 打印匹配到的实体列表（DEBUG级别）
    logger.debug("--- 匹配实体列表 ---")
    for record in initial_nodes_result:
        # record 是一个字典，键对应 RETURN 子句中的别名
        logger.debug(f"- {record.get('nodeText', 'N/A')} ({', '.join(record.get('nodeLabels', []))})")
    logger.debug("----------------------")


    # 提取匹配节点的 elementId 列表，用于后续的子图扩展查询
    # elementId 是 Neo4j 内部标识节点/关系的唯一ID，在查询时使用 elementId() 函数获取
    start_node_ids = [record['nodeId'] for record in initial_nodes_result if 'nodeId' in record]

    # Step 2: 从匹配节点出发，扩展指定跳数，获取子图
    # MATCH (start_node) WHERE elementId(start_node) IN $start_node_ids 找到起始节点
    # MATCH path = (start_node)-[r*0..$hops]-(neighbor) 查找从起始节点出发，路径长度在 0 到 hops 之间的所有路径
    # *0..$hops 表示路径可以包含 0 个或更多关系，最多 hops 个
    # neighbor 是路径的另一端节点
    # RETURN nodes(path) 和 relationships(path) 返回路径上的所有节点和关系对象
    subgraph_query = """
    MATCH (start_node)
    WHERE elementId(start_node) IN $start_node_ids
    MATCH path = (start_node)-[r*0..$hops]-(neighbor)
    // 可以添加 WHERE 子句进一步过滤 neighbor 节点的类型或 r 关系的类型
    RETURN nodes(path) AS nodes, relationships(path) AS relationships
    """
    subgraph_parameters = {
        'start_node_ids': start_node_ids,
        'hops': hops
    }

    # 执行子图扩展查询
    try:
        subgraph_result = manager.query(subgraph_query, parameters=subgraph_parameters)
    except Exception as e:
        logger.error(f"执行子图扩展查询失败: {e}")
        print(f"错误：执行子图扩展查询失败: {e}")
        visualize_subgraph([], [], output_filename) # 查询失败，生成空图
        return


    if not subgraph_result:
        logger.warning("扩展查询未返回任何节点或关系。")
        visualize_subgraph([], [], output_filename) # 结果为空，生成空图
        return

    # Step 3: 处理查询结果，提取唯一的节点和关系列表用于可视化
    # Neo4j 查询返回的结果是以路径为单位的，同一个节点或关系可能出现在多条路径中
    # 需要对结果进行去重，构建唯一的节点和关系列表
    all_nodes = {} # 使用字典，以节点的 elementId 作为键，存储唯一的节点数据
    all_relationships = {} # 使用字典，以关系的 elementId 作为键，存储唯一的关系数据

    # 遍历查询结果的每一条记录（每条路径）
    for record in subgraph_result:
        # 从 Record 对象中获取 nodes 和 relationships 列表
        # Neo4j Python Driver Result 对象包含各种方法，data() 方法返回字典列表，
        # 但如果直接访问 record['nodes']，它可能是包含 Node/Relationship 对象的列表。
        # Node/Relationship 对象有 .element_id, .labels, .properties, .type, .start_node, .end_node 属性。
        # 这里的代码假设 record.get('nodes', []) 和 record.get('relationships', []) 直接返回这些对象列表。
        nodes_in_path = record.get('nodes', [])
        rels_in_path = record.get('relationships', [])

        # 遍历路径中的节点，并添加到 all_nodes 字典中进行去重
        for node in nodes_in_path:
            # 检查 node 是否是有效的 Node 对象且有 element_id
            if hasattr(node, 'element_id'):
                 node_id = node.element_id
                 # 如果节点尚未添加
                 if node_id not in all_nodes:
                      all_nodes[node_id] = {
                          'elementId': node_id,
                          'labels': list(node.labels), # labels 是 frozenset，转换为 list
                          'properties': dict(node) # properties 是属性字典
                      }
            else:
                 logger.warning(f"跳过无效的节点对象: {node}")


        # 遍历路径中的关系，并添加到 all_relationships 字典中进行去重
        for rel in rels_in_path:
             # 检查 rel 是否是有效的 Relationship 对象且有 element_id
             if hasattr(rel, 'element_id'):
                rel_id = rel.element_id
                # 如果关系尚未添加
                if rel_id not in all_relationships:
                    # 提取关系的头尾节点 element_id
                    start_node_element_id = rel.start_node.element_id if hasattr(rel, 'start_node') and hasattr(rel.start_node, 'element_id') else None
                    end_node_element_id = rel.end_node.element_id if hasattr(rel, 'end_node') and hasattr(rel.end_node, 'element_id') else None

                    # 确保关系数据完整
                    if start_node_element_id is not None and end_node_element_id is not None and hasattr(rel, 'type'):
                         all_relationships[rel_id] = {
                             'elementId': rel_id,
                             'type': rel.type,
                             'startNodeElementId': start_node_element_id,
                             'endNodeElementId': end_node_element_id,
                             'properties': dict(rel) # properties 是属性字典
                         }
                    else:
                         logger.warning(f"跳过数据不完整的关系对象: {rel}")
             else:
                  logger.warning(f"跳过无效的关系对象: {rel}")


    logger.info(f"提取子图共包含 {len(all_nodes)} 个节点和 {len(all_relationships)} 个关系。")

    # Step 4: 可视化提取的子图
    # 将字典的 value 转为列表，传递给 visualize_subgraph
    visualize_subgraph(list(all_nodes.values()), list(all_relationships.values()), output_filename)
    # visualize_subgraph 函数内部会打印保存成功的消息


# Example execution block for script (这部分代码只在直接运行 visualization_retrieval.py 文件时执行)
if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="铝合金知识图谱可视化与检索脚本。")
    parser.add_argument(
        "-k", "--keyword",
        required=True, # 关键词是必需的参数
        help="用于检索知识图谱的关键词。脚本将查找包含此关键词的节点并可视化其邻居。"
    )
    parser.add_argument(
        "-n", "--hops",
        type=int, # 参数类型为整数
        default=1, # 默认扩展跳数为 1
        help="从匹配节点向外扩展的跳数 (例如：0表示只显示匹配节点，1表示显示直接邻居及关系)。默认为 1。"
    )
    parser.add_argument(
        "-c", "--config",
        default='config/settings.yaml', # 默认配置文件路径
        help="项目配置文件路径 (默认为 config/settings.yaml，相对于项目根目录)。"
    )
    parser.add_argument(
        "-o", "--output",
        default='search_results_graph.html', # 默认输出文件名
        help="输出的 HTML 图谱文件名。文件将保存在 data/kg_data/ 目录下。默认为 search_results_graph.html。"
    )

    args = parser.parse_args()

    # 配置日志级别，使其在命令行执行时显示 INFO 级别消息
    # 在脚本的 __main__ 块中设置 basicConfig，通常只在此处生效一次
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- 知识图谱检索与可视化脚本开始 ---")
    logger.info(f"命令行参数: 关键词='{args.keyword}', 跳数={args.hops}, 配置文件='{args.config}', 输出文件='{args.output}'")


    # 加载配置并初始化 Neo4jManager
    # utils.load_config 内部会处理查找配置文件路径
    config = utils.load_config(args.config)

    neo4j_manager_instance = None
    if config:
        # 将配置路径传递给 Neo4jManager，以便它加载自己的 Neo4j 连接配置
        neo4j_manager_instance = neo4j_manager.Neo4jManager(config_path=args.config)

        # 检查 Neo4j 连接是否成功
        if neo4j_manager_instance.driver:
            logger.info("成功连接到 Neo4j 数据库。")
            # 执行检索和可视化功能
            search_and_visualize_kg(
                manager=neo4j_manager_instance, # 传递 Neo4jManager 实例
                keyword=args.keyword,
                hops=args.hops,
                output_filename=args.output
            )
        else:
            # 如果连接失败，错误信息已在 Neo4jManager 中记录
            logger.error("无法连接到 Neo4j 数据库，无法执行检索和可视化。请检查 config/settings.yaml 中的连接信息和数据库是否已启动。")
            # 即使连接失败，也尝试生成一个空图，避免脚本因错误终止
            visualize_subgraph([], [], args.output)
            logger.info("已生成表示连接失败的空图文件。")
    else:
        logger.error("配置文件加载失败，无法连接到数据库进行检索。请检查 config/settings.yaml 文件路径和内容。")
        # 配置加载失败，生成空图
        visualize_subgraph([], [], args.output)
        logger.info("已生成表示配置加载失败的空图文件。")


    # 在脚本结束前关闭数据库连接
    if neo4j_manager_instance:
        neo4j_manager_instance.close_db()
        logger.info("数据库连接已关闭。")

    logger.info("--- 知识图谱检索与可视化脚本结束 ---")

    print("--- 知识图谱检索与可视化脚本结束 ---") # 同时打印到控制台