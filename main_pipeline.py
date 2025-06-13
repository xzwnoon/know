# -*- coding: utf-8 -*-
# main_pipeline.py

"""
铝合金知识图谱构建主流程脚本。
"""
import sys
import os
import os
import sys
import logging
import json
# 添加src目录到sys.path
src_dir = os.path.join(os.path.dirname(__file__), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)



# 添加项目根目录到 Python 路径
print(f"当前文件路径: {__file__}")
know_dir = os.path.dirname(os.path.abspath(__file__))
print(f"当前文件路径: {__file__}")
print(f"计算的项目目录: {know_dir}")
print(f"当前 sys.path: {sys.path}")
if know_dir not in sys.path:
    sys.path.insert(0, know_dir)
    print(f"添加后的 sys.path: {sys.path}")
import argparse
import glob
import json
import re
from typing import Any, Dict, List, Optional

# 导入模块
try:
    from src import utils
    from src import pdf_parser
    from src import ner_module
    from src import re_module
    from src import neo4j_manager

    load_config = utils.load_config
    PDFParser = pdf_parser.PDFParser
    NERExtractor = ner_module.NERExtractor
    REExtractor = re_module.REExtractor
    Neo4jManager = neo4j_manager.Neo4jManager

    logger = logging.getLogger('KGPipeline')

except ImportError as e:
    print(f"错误：无法导入项目模块: {e}")
    sys.exit(1)

# 配置日志
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
logger.propagate = False

def process_document(file_path: str, config: Dict[str, Any],
                     parser: Any,
                     ner_extractor: Any,
                     re_extractor: Any,
                     neo4j_manager_instance: Any,
                     resume_step: Optional[str] = None,
                     force_overwrite: bool = False) -> Dict[str, Any]:
    """
    处理单个PDF文件，使用TextChunker进行语义分块
    """
    filename = os.path.basename(file_path)
    logger.info(f"--- 开始处理文件: {filename} ---")

    processing_summary = {
        "filename": filename,
        "status": "Processing",
        "steps_completed": [],
        "entity_count": 0,
        "relationship_count": 0,
        "import_summary": None,
        "error": None
    }

    # 中间结果路径
    kg_data_dir = os.path.join(r"C:\Users\Xiang\Desktop\workflow\know", "kg_data")
    if not os.path.exists(kg_data_dir):
        os.makedirs(kg_data_dir, exist_ok=True)
    doc_id = os.path.splitext(filename)[0]
    entities_path = os.path.join(kg_data_dir, f"{doc_id}_entities.json")
    relations_path = os.path.join(kg_data_dir, f"{doc_id}_relations.json")

    try:
        # 阶段1: PDF解析/NER复用
        if resume_step in ["NER", "RE", "NEO4J"] and os.path.exists(entities_path) and not force_overwrite:
            logger.info(f"[{filename}] 检测到实体中间结果，跳过PDF解析和NER")
            with open(entities_path, "r", encoding="utf-8") as f:
                entities_list = json.load(f)
            processing_summary["steps_completed"].extend(["PDF_Parsing", "NER_Extraction"])
            processing_summary["entity_count"] = len(entities_list)
            
            if resume_step in ["RE", "NEO4J"] and os.path.exists(relations_path) and not force_overwrite:
                logger.info(f"[{filename}] 检测到关系中间结果，跳过RE")
                with open(relations_path, "r", encoding="utf-8") as f:
                    relationships_list = json.load(f)
                processing_summary["steps_completed"].append("RE_Extraction")
                processing_summary["relationship_count"] = len(relationships_list)
        else:
            # 阶段1: PDF解析
            logger.info(f"[{filename}] 阶段1: 开始PDF解析...")
            structured_text_data = parser.process_single_pdf(file_path)
            
            if not structured_text_data or not structured_text_data.get("sections"):
                raise ValueError("PDF解析失败或未提取到有效文本结构")
                
            processing_summary["steps_completed"].append("PDF_Parsing")
            paragraph_count = sum(len(s.get('paragraphs', [])) for s in structured_text_data.get('sections', []))
            logger.info(f"[{filename}] 阶段1: PDF解析完成。提取到{paragraph_count}个段落")

            # 阶段2: NER抽取
            logger.info(f"[{filename}] 阶段2: 开始NER抽取...")
            try:
                # 使用TextChunker进行语义分块
                chunker = utils.TextChunker(config)
                all_entities = []
                
                for section in structured_text_data.get("sections", []):
                    for para in section.get("paragraphs", []):
                        text = para.get("text", "")
                        if not text.strip():
                            continue
                            
                        try:
                            # 使用TextChunker分块，直接返回符合NER模块要求的格式
                            chunk_data_list = chunker.chunk_text(text)
                            logger.debug(f"文本分块完成，共{len(chunk_data_list)}块")
                            
                            for chunk_data in chunk_data_list:
                                # 添加section标题
                                if 'sections' in chunk_data and chunk_data['sections']:
                                    chunk_data['sections'][0]['title'] = section.get("title", "")
                                chunk_entities = ner_extractor.extract_entities_from_text(chunk_data)
                                if chunk_entities:
                                    all_entities.extend(chunk_entities)
                        except Exception as chunk_e:
                            logger.error(f"[{filename}] NER分块抽取异常: {chunk_e}")
                            # 回退到整段处理，但保持正确的数据结构
                            try:
                                chunk_data = {
                                    "sections": [{
                                        "title": section.get("title", ""),
                                        "paragraphs": [{"text": text}]
                                    }]
                                }
                                chunk_entities = ner_extractor.extract_entities_from_text(chunk_data)
                                if chunk_entities:
                                    all_entities.extend(chunk_entities)
                            except Exception as fallback_e:
                                logger.error(f"[{filename}] 回退处理也失败: {fallback_e}")
                
                # 去重
                seen = set()
                entities_list = []
                for ent in all_entities:
                    ent_key = json.dumps(ent, ensure_ascii=False, sort_keys=True)
                    if ent_key not in seen:
                        seen.add(ent_key)
                        entities_list.append(ent)
                        
            except Exception as ner_e:
                logger.error(f"[{filename}] NER抽取阶段发生异常: {ner_e}", exc_info=True)
                entities_list = []
                processing_summary["error"] = f"NER抽取异常: {ner_e}"
                
            processing_summary["steps_completed"].append("NER_Extraction")
            processing_summary["entity_count"] = len(entities_list)
            logger.info(f"[{filename}] 阶段2: NER抽取完成。提取到{len(entities_list)}个实体")
            
            # 保存实体结果
            with open(entities_path, "w", encoding="utf-8") as f:
                json.dump(entities_list, f, ensure_ascii=False, indent=2)

            # 阶段3: RE抽取
            logger.info(f"[{filename}] 阶段3: 开始RE抽取...")
            try:
                if entities_list and len(entities_list) > 1:
                    all_relations = []
                    
                    # 再次使用TextChunker进行语义分块
                    for section in structured_text_data.get("sections", []):
                        for para in section.get("paragraphs", []):
                            text = para.get("text", "")
                            if not text.strip():
                                continue
                                
                            try:
                                # 获取当前段落中的实体
                                para_entities = [e for e in entities_list if e.get('source_text', '').strip() in text.strip()]
                                if len(para_entities) < 2:
                                    continue
                                    
                                # 使用TextChunker分块
                                text_segments = chunker.chunk_text(text)
                                logger.debug(f"文本分块完成，共{len(text_segments)}块")
                                
                                for seg in text_segments:
                                    # 获取当前分块中的实体
                                    seg_entities = [e for e in para_entities if e.get('text', '') in seg]
                                    if len(seg_entities) >= 2:
                                        chunk_data = {"sections": [{"title": section.get("title", ""), "paragraphs": [seg]}]}
                                        chunk_rels = re_extractor.extract_relations_from_text(chunk_data, seg_entities)
                                        if chunk_rels:
                                            all_relations.extend(chunk_rels)
                            except Exception as chunk_e:
                                logger.error(f"[{filename}] RE分块抽取异常: {chunk_e}")
                                
                    # 去重
                    seen_rel = set()
                    relationships_list = []
                    for rel in all_relations:
                        rel_key = json.dumps(rel, ensure_ascii=False, sort_keys=True)
                        if rel_key not in seen_rel:
                            seen_rel.add(rel_key)
                            relationships_list.append(rel)
                            
                    # 规则兜底 - 仅补充re_module.py未提取的关系
                    if not relationships_list:
                        logger.warning(f"[{filename}] RE解析无结果，自动规则兜底")
                        rule_rels = rule_based_relation_extraction(entities_list)
                        # 过滤掉已存在的重复关系
                        relationships_list = [
                            rel for rel in rule_rels
                            if not any(
                                r['head_entity_text'] == rel['head_entity_text'] and 
                                r['tail_entity_text'] == rel['tail_entity_text']
                                for r in all_relations
                            )
                        ]
                        logger.info(f"[{filename}] 规则兜底提取到{len(rule_rels)}个关系，过滤后新增{len(relationships_list)}个")
                else:
                    relationships_list = []
                    logger.warning(f"[{filename}] 未提取到足够实体({len(entities_list)}<2)，跳过关系抽取")
                    
            except Exception as re_e:
                logger.error(f"[{filename}] RE抽取阶段发生异常: {re_e}", exc_info=True)
                relationships_list = []
                processing_summary["error"] = (processing_summary.get("error", "") + f" RE抽取异常: {re_e}")
                
            processing_summary["steps_completed"].append("RE_Extraction")
            processing_summary["relationship_count"] = len(relationships_list)
            logger.info(f"[{filename}] 阶段3: RE抽取完成。提取到{len(relationships_list)}个关系")
            
            # 保存关系结果
            with open(relations_path, "w", encoding="utf-8") as f:
                json.dump(relationships_list, f, ensure_ascii=False, indent=2)

        # 阶段4: Neo4j导入
        logger.info(f"[{filename}] 阶段4: 开始导入Neo4j数据库...")
        if neo4j_manager_instance and neo4j_manager_instance.driver:
            if entities_list or relationships_list:
                import_results = neo4j_manager_instance.batch_import(entities_list, relationships_list)
                processing_summary["import_summary"] = import_results
                processing_summary["steps_completed"].append("Neo4j_Import")
                logger.info(f"[{filename}] 阶段4: Neo4j导入完成")
            else:
                logger.info(f"[{filename}] 没有实体或关系需要导入")
                processing_summary["import_summary"] = {"entities_processed": 0, "entities_upserted": 0, "entities_errors": [], 
                                                       "relationships_processed": 0, "relationships_upserted": 0, "relationships_errors": []}
                processing_summary["steps_completed"].append("Neo4j_Import_Skipped")
        else:
            logger.error(f"[{filename}] Neo4jManager未连接，跳过导入")
            processing_summary["error"] = "Neo4j数据库连接失败，导入被跳过"
            processing_summary["status"] = "Failed_Import"
            processing_summary["entities_list"] = entities_list if entities_list is not None else []
            processing_summary["relationships_list"] = relationships_list if relationships_list is not None else []
            return processing_summary

        # 阶段5: 完成
        processing_summary["status"] = "Completed"
        logger.info(f"--- 文件处理成功完成: {filename} ---")

    except Exception as e:
        logger.error(f"[{filename}] 处理过程中发生错误: {e}", exc_info=True)
        processing_summary["status"] = "Failed"
        processing_summary["error"] = str(e)
        logger.info(f"--- 文件处理失败: {filename} ---")

    processing_summary["entities_list"] = entities_list if entities_list is not None else []
    processing_summary["relationships_list"] = relationships_list if relationships_list is not None else []
    return processing_summary

def rule_based_relation_extraction(entities: list) -> list:
    """
    基于实体类型的规则关系抽取
    """
    rels = []
    type_map = {}
    for ent in entities:
        t = ent.get('type')
        if t not in type_map:
            type_map[t] = []
        type_map[t].append(ent)
        
    # 工艺-性能
    for proc in type_map.get('HeatTreatmentProcess', []):
        for prop in type_map.get('MechanicalProperty', []):
            rels.append({
                'head_entity_text': proc['text'],
                'head_entity_type': proc['type'],
                'tail_entity_text': prop['text'], 
                'tail_entity_type': prop['type'],
                'relation_type': 'EXHIBITS_PROPERTY',
                'attributes': {'source': 'rule'}
            })
            
    # 其他关系类型...
    
    return rels

def main():
    """
    主执行函数
    """
    parser = argparse.ArgumentParser(description="铝合金知识图谱构建主流程脚本")
    parser.add_argument("-i", "--input", required=True, help="待处理的PDF文件路径或目录")
    parser.add_argument("-c", "--config", default='config/settings.yaml', help="配置文件路径")
    parser.add_argument("--resume_step", choices=["PDF", "NER", "RE", "NEO4J"], default=None)
    parser.add_argument("--force_overwrite", action="store_true")

    args = parser.parse_args()
    logger.info("--- 知识图谱构建流程开始 ---")
    logger.info(f"启动参数: 输入路径='{args.input}', 配置文件='{args.config}'")

    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    if not config:
        logger.error("无法加载配置文件，程序终止")
        sys.exit(1)
    logger.info("配置文件加载成功")

    # 初始化模块
    logger.info("初始化处理模块...")
    try:
        parser = PDFParser(config)
        ner_extractor = NERExtractor(config)
        re_extractor = REExtractor(config)
        neo4j_manager_instance = Neo4jManager(config)

        if not neo4j_manager_instance or not neo4j_manager_instance.driver:
            logger.error("Neo4j数据库连接失败，程序终止")
            sys.exit(1)

        logger.info("所有处理模块初始化成功")

    except Exception as e:
        logger.error(f"初始化处理模块时发生错误: {e}", exc_info=True)
        sys.exit(1)

    # 确定待处理文件
    input_path = args.input
    files_to_process = []

    if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        files_to_process.append(input_path)
        logger.info(f"输入路径指定了单个PDF文件: {input_path}")
    elif os.path.isdir(input_path):
        abs_input_path = os.path.abspath(input_path)
        pdf_pattern = os.path.join(abs_input_path, '**', '*.pdf')
        try:
            files_to_process = list(glob.iglob(pdf_pattern, recursive=True))
        except Exception as e:
            logger.error(f"列出目录中的文件时发生错误: {e}")
            
        if not files_to_process:
            logger.warning(f"在目录中未找到任何PDF文件")
    else:
        logger.error(f"输入路径无效")
        if neo4j_manager_instance: neo4j_manager_instance.close_db()
        sys.exit(1)

    if not files_to_process:
        logger.info("没有需要处理的PDF文件")
        if neo4j_manager_instance: neo4j_manager_instance.close_db()
        sys.exit(0)

    logger.info(f"总共找到{len(files_to_process)}个PDF文件待处理")

    # 批量处理文件
    total_files = len(files_to_process)
    processed_count = 0
    failed_count = 0
    overall_summary = []

    for i, file_path in enumerate(files_to_process):
        logger.info(f"[{i+1}/{total_files}] 正在处理文件: {os.path.basename(file_path)}")
        summary = process_document(
            file_path=file_path,
            config=config,
            parser=parser,
            ner_extractor=ner_extractor,
            re_extractor=re_extractor,
            neo4j_manager_instance=neo4j_manager_instance,
            resume_step=args.resume_step,
            force_overwrite=args.force_overwrite
        )
        overall_summary.append(summary)
        if summary["status"] == "Completed":
            processed_count += 1
        else:
            failed_count += 1

    logger.info("--- 所有文件处理完成 ---")
    logger.info(f"总文件数: {total_files}, 成功处理: {processed_count}, 失败: {failed_count}")

    # 打印处理总结
    print("\n--- 处理总结 ---")
    print(f"{'状态':<8} | {'文件名':<30} | {'实体数':<6} | {'关系数':<6} | {'导入结果':<25} | {'错误信息'}")
    print("-" * 100)
    for summary in overall_summary:
        status_icon = "✅ 完成" if summary['status'] == 'Completed' else ("❌ 失败" if summary['status'] == 'Failed' else "⚠️ 部分失败")
        filename_display = (summary['filename'][:27] + '...') if len(summary['filename']) > 30 else summary['filename']
        entity_count = summary.get('entity_count', 0)
        relation_count = summary.get('relationship_count', 0)
        error_display = (summary['error'][:50] + '...') if summary.get('error') else ''        
        print(f"{status_icon:<8} | {filename_display:<30} | {entity_count:<6} | {relation_count:<6} | {'':<25} | {error_display}")

    print("-" * 100)

if __name__ == "__main__":
    main()
