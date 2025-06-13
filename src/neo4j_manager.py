# src/neo4j_manager.py

import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json

# 导入 Neo4j 驱动相关的类
from neo4j import GraphDatabase, Driver, Session, Transaction, Result
from neo4j.exceptions import Neo4jError, ServiceUnavailable

# 导入同目录下的 utils 模块，用于加载配置
import utils

# 配置 neo4j_manager 的日志记录
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # INFO 级别显示连接和批量导入进度，DEBUG 更详细

class Neo4jManager:
    """
    Neo4j数据库管理器的简化实现
    用于生成知识图谱原型
    """
    def __init__(self, config):
        """
        初始化Neo4jManager
        支持直接传入配置字典
        """
        self.config = config
        self.driver = None
        logger.info("使用Neo4j模拟数据模式")
        # 尝试从config中获取neo4j部分
        if self.config:
            # 兼容老的config_path逻辑
            if isinstance(self.config, dict) and 'neo4j' in self.config:
                self._neo4j_config = self.config['neo4j']
            else:
                self._neo4j_config = None
            if self._neo4j_config:
                self.driver = self.connect_db()
                if self.driver:
                    self._create_uniqueness_constraints()
            else:
                logger.error("Neo4jManager 初始化失败：无法加载或验证配置（缺少'neo4j'部分）。")
        else:
            logger.error("Neo4jManager 初始化失败：未传入有效配置。")


    def _load_and_validate_config(self) -> Optional[Dict]:
        """
        加载并验证配置文件中的 Neo4j 部分。

        Returns:
            Neo4j 配置的字典部分，如果加载或验证失败则返回 None。
        """
        # 使用 utils.load_config 函数加载整个配置文件
        config = utils.load_config(self.config_path)
        if not config:
            logger.error("无法加载配置文件，无法获取 Neo4j 配置。")
            return None

        # 检查配置文件中是否存在 'neo4j' 部分
        neo4j_config = config.get('neo4j')
        if not neo4j_config:
            logger.error("配置文件中缺少 'neo4j' 部分。")
            return None

        # 检查 Neo4j 连接必需的详细信息是否存在
        uri = neo4j_config.get('uri')
        user = neo4j_config.get('user')
        password = neo4j_config.get('password')

        if not uri or not user or not password:
            logger.error("配置文件中缺少 Neo4j 连接详细信息 (uri, user, 或 password)。")
            return None

        # 存储验证通过的 Neo4j 配置部分
        self._neo4j_config = neo4j_config
        logger.info("Neo4j 配置加载成功。")
        return self._neo4j_config # 返回 Neo4j 的配置字典


    def connect_db(self, retries: int = 5, delay: int = 5) -> Optional[Driver]:
        """
        建立与 Neo4j 数据库的连接，并返回 Neo4j 驱动对象。

        连接参数从初始化时加载的配置中获取。包含连接失败时的重试机制。

        Args:
            retries: 连接重试次数。
            delay: 每次重试之间的等待时间（秒）。

        Returns:
            成功连接时返回 Neo4j 驱动对象；连接失败达到最大重试次数后返回 None。
        """
        # 检查是否成功加载了 Neo4j 配置
        if not hasattr(self, '_neo4j_config') or not self._neo4j_config:
            logger.error("无法建立连接：Neo4j 配置未加载或验证失败。")
            return None

        uri = self._neo4j_config['uri']
        user = self._neo4j_config['user']
        password = self._neo4j_config['password']

        # 尝试连接数据库，并在失败时重试
        for i in range(retries):
            try:
                # 使用 GraphDatabase.driver 创建驱动对象
                driver = GraphDatabase.driver(uri, auth=(user, password))
                # 调用 verify_connectivity() 立即检查连接是否可用
                driver.verify_connectivity()
                logger.info(f"成功连接到 Neo4j 数据库 {uri} (尝试 {i+1}/{retries})")
                return driver # 连接成功，返回驱动对象
            except ServiceUnavailable as e:
                logger.warning(f"无法连接到 Neo4j 数据库 {uri} (尝试 {i+1}/{retries}): 服务不可用。{e}")
                if i < retries - 1:
                    logger.info(f"等待 {delay} 秒后重试...")
                    time.sleep(delay)
                else:
                    logger.error(f"达到最大重试次数 ({retries})，无法连接到 Neo4j 数据库 {uri}。请检查数据库是否已启动并可访问。")
                    return None # 重试耗尽，连接失败
            except Exception as e:
                # 捕获其他建立连接时可能发生的意外错误
                logger.error(f"建立 Neo4j 连接时发生意外错误: {e}")
                # 对于非服务不可用错误，通常不重试，直接失败
                return None
        # 理论上，如果 retries > 0 且没有异常，代码不会执行到这里，但作为安全返回
        return None

    def close_db(self):
        """关闭数据库连接。"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 数据库连接已关闭。")
            self.driver = None # 将驱动对象设为 None，表示连接已不再可用

    def query(self, query: str, parameters: Optional[Dict] = None, database: Optional[str] = None) -> List[Dict]:
        """
        执行一个 Cypher 查询（读或写）。返回查询结果数据。

        这是一个通用的查询方法，适用于简单的读写操作。对于批量导入，推荐使用 batch_import。

        Args:
            query: Cypher 查询字符串。
            parameters: 查询参数字典，用于安全地传递值到查询中，避免注入风险。
            database: 要操作的数据库名称 (例如 "neo4j")。None 使用默认数据库。

        Returns:
            查询结果列表，每个元素是一个代表一行记录的字典。
            如果数据库连接不可用或查询失败，返回空列表 []。
        """
        if not self.driver:
            logger.error("数据库连接不可用，无法执行查询。")
            return []

        try:
            # 使用 driver.session() 获取一个会话，with 语句确保会话结束后资源被释放
            with self.driver.session(database=database) as session:
                 # session.run() 执行查询。Result 对象代表查询结果。
                 result: Result = session.run(query, parameters)
                 # 使用 result.data() 获取所有记录，并转换为字典列表
                 records_data = [record.data() for record in result]
                 logger.debug(f"查询执行成功。返回 {len(records_data)} 条记录。查询: {query[:100]}...")
                 return records_data

        except ServiceUnavailable as e:
            logger.error(f"查询执行失败：数据库服务不可用。{e}")
            return []
        except Neo4jError as e:
            logger.error(f"查询执行失败：Neo4j 错误。查询: {query[:200]}... 错误: {e}")
            return []
        except Exception as e:
            logger.error(f"查询执行失败：意外错误。查询: {query[:200]}... 错误: {e}")
            return []

    # 定义允许的实体和关系类型列表，用于导入前的基本验证
    # 这些类型应与 Schema 设计中的节点标签和关系类型一致
    allowed_entity_types = [
        "MaterialAlloy", "AlloyComponent", "HeatTreatmentProcess",
        "ResearchSubject", "ResearchMethod", "MechanicalProperty",
        "MicrostructureFeature", "Application", "SourceDocument" # SourceDocument 可以表示文献本身
    ]
    allowed_relation_types = [
         "HAS_COMPONENT", "EXHIBITS_PROPERTY", "UNDERGOES_TREATMENT",
         "INVESTIGATED_BY", "HAS_MICROSTRUCTURE", "HAS_APPLICATION",
         "MENTIONED_IN" # MENTIONED_IN 关系可以连接实体到 SourceDocument 节点
    ]
    # 关系唯一性约束字段定义（头尾实体 text + 关系类型）
    relation_unique_keys = [
        ("HAS_COMPONENT",    ["head_entity_text", "tail_entity_text", "relation_type"]),
        ("EXHIBITS_PROPERTY",["head_entity_text", "tail_entity_text", "relation_type"]),
        ("UNDERGOES_TREATMENT",["head_entity_text", "tail_entity_text", "relation_type"]),
        ("INVESTIGATED_BY",  ["head_entity_text", "tail_entity_text", "relation_type"]),
        ("HAS_MICROSTRUCTURE",["head_entity_text", "tail_entity_text", "relation_type"]),
        ("HAS_APPLICATION",  ["head_entity_text", "tail_entity_text", "relation_type"]),
        ("MENTIONED_IN",     ["head_entity_text", "tail_entity_text", "relation_type"]),
    ]

    # --------------------------------------------------------------
    # 批量导入相关的内部事务函数 (_create_nodes_batch_tx, _create_relationships_batch_tx)
    # 这些函数在 batch_import 方法内部被调用，用于在单个事务中处理数据批次。
    # 它们设计为由 driver.execute_write() 方法执行。
    # Cypher 查询使用了 UNWIND 子句来处理输入列表，提高了批量导入效率。
    # 注意：这些查询使用了 APOC 过程 (apoc.map.clean, apoc.map.merge, coalesce)。
    # APOC 需要在 Neo4j 数据库中额外安装和启用。如果无法使用 APOC，需要修改 Cypher 查询。
    # --------------------------------------------------------------

    def _flatten_attributes(self, item: dict) -> dict:
        # 将 attributes 字段平铺到顶层，避免写入 dict
        flat = dict(item)
        attrs = flat.pop('attributes', None)
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                # 只平铺基本类型
                if isinstance(v, (str, int, float, bool)) or v is None:
                    flat[k] = v
                else:
                    flat[k] = str(v)
        return flat


    def _create_nodes_batch_tx(self, tx: Transaction, entities_batch: List[Dict]) -> Tuple[int, int, List[Dict]]:
        """
        事务函数：使用 UNWIND 批量创建或合并节点。
        期望处理的是同一类型实体的批次。

        Args:
            tx: Neo4j 事务对象。
            entities_batch: 同一类型实体字典的列表。

        Returns:
            元组：(已处理数量, 已创建/更新数量, 错误列表)
        """
        if not entities_batch:
            return 0, 0, []

        # 检查批次中实体类型的一致性（虽然 batch_import 应该已经分组）
        first_type = entities_batch[0].get('type')
        if not first_type or not all(e.get('type') == first_type for e in entities_batch):
             error_msg = "批量导入实体批次包含混合或缺失类型！此事务函数需要单一实体类型。"
             logger.error(error_msg)
             errors = [{"error": error_msg, "entity_data": e} for e in entities_batch]
             return 0, 0, errors

        entity_type = first_type

        # 在构建查询前再次验证实体类型是否在允许列表中
        if entity_type not in self.allowed_entity_types:
             error_msg = f"未知或不允许的实体类型 '{entity_type}'，跳过此批次导入。"
             logger.warning(error_msg)
             errors = [{"error": error_msg, "entity_data": e} for e in entities_batch]
             return 0, 0, errors

        # 平铺 attributes 字段
        flat_batch = [self._flatten_attributes(e) for e in entities_batch]

        # 构建 Cypher 查询（只写入基本类型属性）
        query = f"""
        UNWIND $entity_list AS entity_data
        WITH entity_data WHERE entity_data.text IS NOT NULL AND entity_data.text <> ''
        MERGE (n:{entity_type} {{ text: entity_data.text }})
        SET n += entity_data,
            n.created_at = coalesce(n.created_at, timestamp()),
            n.updated_at = timestamp(),
            n.batch_id = entity_data.batch_id
        """

        parameters = {'entity_list': flat_batch}
        try:
            tx.run(query, parameters)
            processed_count = len(flat_batch)
            upserted_count = processed_count
            logger.debug(f"成功处理 {processed_count} 个 '{entity_type}' 实体导入。")
            return processed_count, upserted_count, []
        except Neo4jError as e:
            logger.error(f"Neo4jError 在批量导入实体 '{entity_type}' 批次时发生: {e}")
            errors = [{"error": str(e), "batch_start_text": flat_batch[0].get('text', 'N/A') if flat_batch else "N/A"}]
            return 0, 0, errors
        except Exception as e:
            logger.error(f"意外错误在批量导入实体 '{entity_type}' 批次时发生: {e}")
            errors = [{"error": str(e), "batch_start_text": flat_batch[0].get('text', 'N/A') if flat_batch else "N/A"}]
            return 0, 0, errors


    def _create_relationships_batch_tx(self, tx: Transaction, relationships_batch: List[Dict]) -> Tuple[int, int, List[Dict]]:
        """
        事务函数：批量创建或合并关系（无 APOC，静态拼接 Cypher，兼容所有 Neo4j 社区版）。

        Args:
            tx: Neo4j 事务对象。
            relationships_batch: 同一关系类型实体字典的列表。

        Returns:
            元组：(已处理数量, 已创建/更新数量, 错误列表)
        """
        processed_count = 0
        upserted_count = 0
        errors = []
        for rel in relationships_batch:
            try:
                head_label = rel.get('head_entity_type', 'Entity')
                tail_label = rel.get('tail_entity_type', 'Entity')
                rel_type = rel.get('relation_type', 'RELATED_TO')
                head_text = rel.get('head_entity_text', '')
                tail_text = rel.get('tail_entity_text', '')
                rel_data = self._flatten_attributes(rel)
                cypher = (
                    f"MATCH (source:`{head_label}` {{ text: $head_text }}) "
                    f"MATCH (target:`{tail_label}` {{ text: $tail_text }}) "
                    f"MERGE (source)-[r:`{rel_type}`]->(target) "
                    f"SET r += $rel_data, "
                    f"r.created_at = coalesce(r.created_at, timestamp()), "
                    f"r.updated_at = timestamp(), "
                    f"r.batch_id = $batch_id"
                )
                params = {
                    "head_text": head_text,
                    "tail_text": tail_text,
                    "rel_data": rel_data,
                    "batch_id": rel.get('batch_id')
                }
                tx.run(cypher, params)
                processed_count += 1
                upserted_count += 1
            except Exception as e:
                errors.append({
                    "error": str(e),
                    "relationship_data": rel,
                    "cypher": cypher,
                    "params": params
                })
        return processed_count, upserted_count, errors


    def _get_neo4j_version(self) -> str:
        """
        获取当前 Neo4j 数据库的主版本号（如 '4', '5'），用于判断是否支持关系唯一性约束。
        """
        if not self.driver:
            return ""
        try:
            with self.driver.session() as session:
                result = session.run("CALL dbms.components() YIELD versions RETURN versions[0] AS version")
                record = result.single()
                if record:
                    version = record["version"]
                    logger.info(f"Neo4j 版本: {version}")
                    return version.split(".")[0]
        except Exception as e:
            logger.warning(f"获取 Neo4j 版本失败: {e}")
        return ""


    def _create_uniqueness_constraints(self):
        """
        为所有实体类型的 text 字段添加唯一性约束，防止重复导入。
        Neo4j 5.x+ 时为关系添加三元组唯一性约束。
        """
        if not self.driver:
            return
        with self.driver.session() as session:
            for label in self.allowed_entity_types:
                try:
                    constraint_name = f"uniq_{label}_text"
                    cypher = f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.text IS UNIQUE"
                    session.run(cypher)
                    logger.info(f"唯一性约束已确保: {constraint_name}")
                except Exception as e:
                    logger.warning(f"添加唯一性约束 {label} 失败: {e}")
            # 检查是否为 Neo4j 5.x 及以上，支持关系唯一性约束
            version_major = self._get_neo4j_version()
            if version_major and int(version_major) >= 5:
                for rel_type in self.allowed_relation_types:
                    try:
                        constraint_name = f"uniq_{rel_type}_triple"
                        cypher = (
                            f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                            f"FOR ()-[r:{rel_type}]-() REQUIRE (r.head_entity_text, r.tail_entity_text, r.relation_type) IS UNIQUE"
                        )
                        session.run(cypher)
                        logger.info(f"关系唯一性约束已确保: {constraint_name}")
                    except Exception as e:
                        logger.warning(f"添加关系唯一性约束 {rel_type} 失败: {e}")

    def rollback_batch(self, batch_id: str) -> Dict:
        """
        回滚（删除）指定 batch_id 的所有节点和关系。
        """
        if not self.driver:
            logger.error("数据库连接不可用，无法回滚批次。")
            return {"success": False, "error": "数据库连接不可用"}
        try:
            with self.driver.session() as session:
                # 先删除关系，再删除节点
                rel_result = session.run(
                    "MATCH ()-[r]-() WHERE r.batch_id = $batch_id DELETE r RETURN count(r) as rel_deleted",
                    {"batch_id": batch_id}
                )
                rel_deleted = rel_result.single()["rel_deleted"] if rel_result.single() else 0
                node_result = session.run(
                    "MATCH (n) WHERE n.batch_id = $batch_id DETACH DELETE n RETURN count(n) as node_deleted",
                    {"batch_id": batch_id}
                )
                node_deleted = node_result.single()["node_deleted"] if node_result.single() else 0
                logger.info(f"批次回滚完成: batch_id={batch_id}, 删除关系 {rel_deleted} 条, 节点 {node_deleted} 个。")
                return {"success": True, "rel_deleted": rel_deleted, "node_deleted": node_deleted}
        except Exception as e:
            logger.error(f"批次回滚失败: {e}")
            return {"success": False, "error": str(e)}

    def query_batch(self, batch_id: str) -> Dict:
        """
        查询指定 batch_id 的所有节点和关系。
        """
        if not self.driver:
            logger.error("数据库连接不可用，无法查询批次。")
            return {"success": False, "error": "数据库连接不可用"}
        try:
            with self.driver.session() as session:
                nodes = session.run(
                    "MATCH (n) WHERE n.batch_id = $batch_id RETURN n", {"batch_id": batch_id}
                )
                node_list = [record["n"] for record in nodes]
                rels = session.run(
                    "MATCH ()-[r]-() WHERE r.batch_id = $batch_id RETURN r", {"batch_id": batch_id}
                )
                rel_list = [record["r"] for record in rels]
                logger.info(f"批次查询: batch_id={batch_id}, 节点 {len(node_list)} 个, 关系 {len(rel_list)} 条。")
                return {"success": True, "nodes": node_list, "relationships": rel_list}
        except Exception as e:
            logger.error(f"批次查询失败: {e}")
            return {"success": False, "error": str(e)}

    def record_batch_history(self, batch_id: str, operation: str, status: str, summary: str = "", user_note: str = "", doc_id: str = None, stats: dict = None):
        """
        记录批次操作历史（BatchHistory 节点），包括导入/回滚/更新等。
        operation: import/rollback/update
        status: running/completed/failed/rolledback/superseded/active
        summary: 摘要
        user_note: 用户备注
        doc_id: 关联文档ID（如有）
        stats: 统计信息（如节点/关系数等）
        """
        if not self.driver:
            logger.error("数据库连接不可用，无法记录批次历史。")
            return False
        try:
            with self.driver.session() as session:
                cypher = (
                    "MERGE (b:BatchHistory {batch_id: $batch_id}) "
                    "SET b.operation = $operation, b.status = $status, b.summary = $summary, "
                    "b.user_note = $user_note, b.doc_id = $doc_id, b.stats = $stats, "
                    "b.updated_at = timestamp(), b.created_at = coalesce(b.created_at, timestamp())"
                )
                # stats 必须为字符串
                stats_str = None
                if stats is not None:
                    if isinstance(stats, str):
                        stats_str = stats
                    else:
                        try:
                            stats_str = json.dumps(stats, ensure_ascii=False)
                        except Exception:
                            stats_str = str(stats)
                else:
                    stats_str = "{}"
                params = {
                    "batch_id": batch_id,
                    "operation": operation,
                    "status": status,
                    "summary": summary,
                    "user_note": user_note,
                    "doc_id": doc_id,
                    "stats": stats_str
                }
                session.run(cypher, params)
                logger.info(f"批次历史已记录: batch_id={batch_id}, operation={operation}, status={status}")
                return True
        except Exception as e:
            logger.error(f"记录批次历史失败: {e}")
            return False

    def update_batch_status(self, batch_id: str, status: str, summary: str = "", user_note: str = ""):
        """
        更新批次状态（如 completed/failed/rolledback/superseded/active），可附带摘要和备注。
        """
        return self.record_batch_history(batch_id, operation=None, status=status, summary=summary, user_note=user_note)

    def get_batch_history(self, batch_id: str = None, doc_id: str = None, limit: int = 20) -> list:
        """
        查询批次历史记录。可按 batch_id 或 doc_id 查询，默认返回最新 limit 条。
        """
        if not self.driver:
            logger.error("数据库连接不可用，无法查询批次历史。")
            return []
        try:
            with self.driver.session() as session:
                if batch_id:
                    cypher = "MATCH (b:BatchHistory {batch_id: $batch_id}) RETURN b ORDER BY b.created_at DESC LIMIT $limit"
                    params = {"batch_id": batch_id, "limit": limit}
                elif doc_id:
                    cypher = "MATCH (b:BatchHistory {doc_id: $doc_id}) RETURN b ORDER BY b.created_at DESC LIMIT $limit"
                    params = {"doc_id": doc_id, "limit": limit}
                else:
                    cypher = "MATCH (b:BatchHistory) RETURN b ORDER BY b.created_at DESC LIMIT $limit"
                    params = {"limit": limit}
                result = session.run(cypher, params)
                return [record["b"] for record in result]
        except Exception as e:
            logger.error(f"查询批次历史失败: {e}")
            return []

    def mark_batch_superseded(self, batch_id: str):
        """
        标记某批次为 superseded（被新批次覆盖/废弃）。
        """
        return self.update_batch_status(batch_id, status="superseded")

    def get_active_batch_by_doc_id(self, doc_id: str) -> list:
        """
        查询某文档当前 active 状态的批次（未被 superseded/rolledback）。
        """
        if not self.driver:
            logger.error("数据库连接不可用，无法查询 active 批次。")
            return []
        try:
            with self.driver.session() as session:
                cypher = (
                    "MATCH (b:BatchHistory {doc_id: $doc_id}) "
                    "WHERE b.status IN ['active', 'completed'] "
                    "RETURN b ORDER BY b.created_at DESC"
                )
                result = session.run(cypher, {"doc_id": doc_id})
                return [record["b"] for record in result]
        except Exception as e:
            logger.error(f"查询 active 批次失败: {e}")
            return []

    def soft_delete_batch(self, batch_id: str):
        """
        软删除批次（仅将 BatchHistory 节点状态标记为 deleted，不实际删除数据）。
        """
        return self.update_batch_status(batch_id, status="deleted")

    def _standardize_entity_type(self, entity_type: str) -> str:
        """
        实体类型标准化映射（可扩展为配置化）
        """
        mapping = {
            "Alloy": "MaterialAlloy",
            "合金": "MaterialAlloy",
            "工艺": "HeatTreatmentProcess",
            "性能": "MechanicalProperty",
            "方法": "ResearchMethod",
            "试验": "ResearchMethod",
            "结构": "MicrostructureFeature",
            # ...可扩展...
        }
        return mapping.get(entity_type, entity_type)

    def _standardize_relation_type(self, rel_type: str) -> str:
        """
        关系类型标准化映射（可扩展为配置化）
        """
        mapping = {
            "HAS_PROPERTY": "EXHIBITS_PROPERTY",
            "HAS_MICROSTRUCTURE": "HAS_MICROSTRUCTURE",
            "TREATED_BY": "UNDERGOES_TREATMENT",
            "INVESTIGATED": "INVESTIGATED_BY",
            # ...可扩展...
        }
        return mapping.get(rel_type, rel_type)

    def _extract_relation_attributes(self, rel: dict) -> dict:
        """
        针对常见属性短语自动正则解析温度/时间等结构化属性
        """
        import re
        attrs = rel.get("attributes", {}) or {}
        # 解析温度
        if "temperature" not in attrs:
            temp_match = re.search(r"(\d+\s*[℃°C])", rel.get("object", "") + rel.get("subject", ""))
            if temp_match:
                attrs["temperature"] = temp_match.group(1).replace(" ", "")
        # 解析时间
        if "time" not in attrs:
            time_match = re.search(r"(\d+\s*[hH小时min分])", rel.get("object", "") + rel.get("subject", ""))
            if time_match:
                attrs["time"] = time_match.group(1).replace(" ", "")
        return attrs

    def batch_import(self, entities: List[Dict], relationships: List[Dict], batch_id: Optional[str] = None, batch_size: int = 100, resume: bool = False, skip_if_exists: bool = False, user_note: str = "", doc_id: str = None) -> Dict:
        """
        批量导入实体和关系，支持分批事务、批次ID、去重、日志、错误捕获、属性扁平化、批次统计返回。
        Args:
            entities: 实体列表（dict，需包含 type/text/attributes 等字段）
            relationships: 关系列表（dict，需包含 head_entity_text/head_entity_type/relation_type/tail_entity_text/tail_entity_type/attributes 等字段）
            batch_id: 可选，批次ID。若未指定自动生成。
            batch_size: 每批事务处理的数量。
            resume: 是否断点续导（如遇到已存在的批次ID时跳过已导入部分）。
            skip_if_exists: 若批次ID已存在则跳过导入。
            user_note: 用户备注。
            doc_id: 关联文档ID。
        Returns:
            统计与日志字典，含处理数、去重数、错误数、批次ID等。
        """
        import math
        import copy
        import json
        from collections import defaultdict
        if not self.driver:
            logger.error("数据库连接不可用，无法批量导入。")
            return {"success": False, "error": "数据库连接不可用"}

        if not batch_id:
            batch_id = f"batch_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        logger.info(f"开始批量导入: batch_id={batch_id}, 实体{len(entities)}个, 关系{len(relationships)}条, batch_size={batch_size}")

        if skip_if_exists:
            existing = self.get_batch_history(batch_id=batch_id)
            if existing:
                logger.warning(f"批次ID {batch_id} 已存在，跳过导入。")
                return {"success": True, "skipped": True, "batch_id": batch_id}

        self.record_batch_history(batch_id, operation="import", status="running", summary="批量导入开始", user_note=user_note, doc_id=doc_id)

        # 实体标准化、去重
        seen_entity_keys = set()
        unique_entities = []
        entity_type_errors = []
        for e in entities:
            etype = self._standardize_entity_type(e.get("type", ""))
            text = e.get("text")
            key = (etype, text)
            if not etype or not text:
                entity_type_errors.append({"error": "实体type或text缺失", "entity_data": e})
                continue
            if key not in seen_entity_keys:
                seen_entity_keys.add(key)
                ent = copy.deepcopy(e)
                ent["type"] = etype
                ent["batch_id"] = batch_id
                if doc_id:
                    ent["source_doc_id"] = doc_id
                unique_entities.append(ent)
        entities = unique_entities

        # 关系自动标准化、补全类型、属性正则解析
        entity_text2type = {e["text"]: e["type"] for e in entities if e.get("text") and e.get("type")}
        seen_rel_keys = set()
        unique_relationships = []
        for r in relationships:
            rel_type = self._standardize_relation_type(r.get("relation_type", ""))
            head_text = r.get("head_entity_text")
            tail_text = r.get("tail_entity_text")
            key = (head_text, tail_text, rel_type)
            if not rel_type or not head_text or not tail_text:
                continue
            rel = copy.deepcopy(r)
            rel["relation_type"] = rel_type
            if not rel.get("head_entity_type"):
                rel["head_entity_type"] = entity_text2type.get(head_text, "Entity")
            if not rel.get("tail_entity_type"):
                rel["tail_entity_type"] = entity_text2type.get(tail_text, "Entity")
            # 自动正则补全属性
            rel["attributes"] = self._extract_relation_attributes(rel)
            if key not in seen_rel_keys:
                seen_rel_keys.add(key)
                rel["batch_id"] = batch_id
                if doc_id:
                    rel["source_doc_id"] = doc_id
                unique_relationships.append(rel)
        relationships = unique_relationships

        # 若无有效实体和关系，直接返回 skipped 结构
        if not entities and not relationships:
            logger.warning(f"批次ID {batch_id} 无有效实体和关系，导入跳过。")
            summary = "无有效实体或关系，导入跳过"
            stats = {
                "entities_processed": 0,
                "entities_upserted": 0,
                "entities_errors": len(entity_type_errors),
                "relationships_processed": 0,
                "relationships_upserted": 0,
                "relationships_errors": 0
            }
            self.record_batch_history(batch_id, operation="import", status="skipped", summary=summary, user_note=user_note, doc_id=doc_id, stats=json.dumps(stats))
            return {
                "success": False,
                "skipped": True,
                "batch_id": batch_id,
                "entities_processed": 0,
                "entities_upserted": 0,
                "entities_errors": entity_type_errors,
                "relationships_processed": 0,
                "relationships_upserted": 0,
                "relationships_errors": [],
                "summary": summary,
                "stats": stats
            }

        # 按type分组实体，确保每批type一致，且只导入允许类型
        type2entities = defaultdict(list)
        for ent in entities:
            if ent["type"] in self.allowed_entity_types:
                type2entities[ent["type"]].append(ent)
            else:
                entity_type_errors.append({"error": "实体type不在允许列表", "entity_data": ent})

        entities_processed = 0
        entities_upserted = 0
        entities_errors = list(entity_type_errors)  # 先加上类型/字段错误
        for ent_type, ents in type2entities.items():
            for i in range(0, len(ents), batch_size):
                batch = ents[i:i+batch_size]
                try:
                    with self.driver.session() as session:
                        processed, upserted, errors = session.execute_write(self._create_nodes_batch_tx, batch)
                        entities_processed += processed
                        entities_upserted += upserted
                        if errors:
                            entities_errors.extend(errors)
                except Exception as e:
                    logger.error(f"实体批次导入异常: {e}")
                    entities_errors.append({"error": str(e), "batch_index": i, "entity_type": ent_type})

        # 分批导入关系（只导入允许类型）
        relationships_processed = 0
        relationships_upserted = 0
        relationships_errors = []
        for i in range(0, len(relationships), batch_size):
            batch = [r for r in relationships[i:i+batch_size] if r.get("relation_type") in self.allowed_relation_types]
            if not batch:
                continue
            try:
                with self.driver.session() as session:
                    processed, upserted, errors = session.execute_write(self._create_relationships_batch_tx, batch)
                    relationships_processed += processed
                    relationships_upserted += upserted
                    if errors:
                        relationships_errors.extend(errors)
            except Exception as e:
                logger.error(f"关系批次导入异常: {e}")
                relationships_errors.append({"error": str(e), "batch_index": i})

        summary = f"实体: 处理{entities_processed}，成功{entities_upserted}，错误{len(entities_errors)}；关系: 处理{relationships_processed}，成功{relationships_upserted}，错误{len(relationships_errors)}"
        stats = {
            "entities_processed": entities_processed,
            "entities_upserted": entities_upserted,
            "entities_errors": len(entities_errors),
            "relationships_processed": relationships_processed,
            "relationships_upserted": relationships_upserted,
            "relationships_errors": len(relationships_errors)
        }
        # 记录批次历史（completed/failed），stats序列化为json字符串
        status = "completed" if not entities_errors and not relationships_errors else "failed"
        self.record_batch_history(batch_id, operation="import", status=status, summary=summary, user_note=user_note, doc_id=doc_id, stats=json.dumps(stats))

        logger.info(f"批量导入完成: batch_id={batch_id}, {summary}")
        return {
            "success": status == "completed",
            "batch_id": batch_id,
            "entities_processed": entities_processed,
            "entities_upserted": entities_upserted,
            "entities_errors": entities_errors,
            "relationships_processed": relationships_processed,
            "relationships_upserted": relationships_upserted,
            "relationships_errors": relationships_errors,
            "summary": summary,
            "stats": stats
        }

    def get_doc_relations(self, doc_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取指定文献的所有相关节点和关系
        
        Args:
            doc_id: 文献ID
            
        Returns:
            包含nodes和relationships的字典
        """
        # 数据库不可用时返回mock数据
        if not self.driver:
            logger.warning(f"数据库不可用，返回文献 {doc_id} 的mock数据")
            return {
                "nodes": [
                    {
                        "name": "Al-Mg-Si-Cu合金",
                        "type": "composition",
                        "properties": {"value": "baseline"}
                    },
                    {
                        "name": "固溶处理",
                        "type": "process",
                        "properties": {"parameters": "540°C/1h"}
                    },
                    {
                        "name": "时效处理",
                        "type": "process",
                        "properties": {"parameters": "175°C/8h"}
                    },
                    {
                        "name": "抗拉强度",
                        "type": "property",
                        "properties": {"value": "385MPa"}
                    }
                ],
                "relationships": [
                    {
                        "from": "Al-Mg-Si-Cu合金",
                        "to": "固溶处理",
                        "type": "UNDERGOES_TREATMENT"
                    },
                    {
                        "from": "固溶处理",
                        "to": "时效处理",
                        "type": "FOLLOWED_BY"
                    },
                    {
                        "from": "时效处理",
                        "to": "抗拉强度",
                        "type": "RESULTS_IN"
                    }
                ]
            }

        result = {"nodes": [], "relationships": []}
        try:
            # 查询与文献相关的所有节点和关系的Cypher查询
            query = """
            MATCH (doc:SourceDocument {id: $doc_id})-[r]-(n)
            RETURN n, r, TYPE(r) as relType,
                   CASE WHEN STARTNODE(r) = doc THEN 'out' ELSE 'in' END as direction
            """
            # 执行查询
            data = self.query(query, {"doc_id": doc_id})
            # 处理查询结果
            for record in data:
                node = record.get("n")
                rel = record.get("r")
                rel_type = record.get("relType")
                direction = record.get("direction")
                # 处理节点
                if node:
                    node_data = {
                        "name": node.get("name") or node.get("text") or node.get("id", "未命名"),
                        "type": next(iter(node.get("labels", [])), node.get("type", "unknown")),
                        "properties": node
                    }
                    if node_data not in result["nodes"]:
                        result["nodes"].append(node_data)
                # 处理关系
                if rel and rel_type:
                    rel_data = {
                        "from": doc_id if direction == "out" else node.get("name", node.get("text", "未命名")),
                        "to": node.get("name", node.get("text", "未命名")) if direction == "out" else doc_id,
                        "type": rel_type
                    }
                    result["relationships"].append(rel_data)
            logger.info(f"返回文献 {doc_id} 的真实数据库数据，共{len(result['nodes'])}节点，{len(result['relationships'])}关系")
        except Exception as e:
            logger.error(f"获取文献关系时发生错误: {str(e)}，返回空结果")
        return result

