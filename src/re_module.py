# src/re_module.py

import json
from typing import List, Dict, Any, Optional
import os
import utils
from utils import LLMClient
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class REExtractor:
    def __init__(self, config: Dict = None):
        if config is None:
            config = utils.load_config()
        self.config = config
        self.llm_client = LLMClient(self.config.get('llm', {}))
        self.re_prompt_template = self.config.get('prompts', {}).get('re_template')
        if not self.re_prompt_template:
            self.re_prompt_template = """Extract relations (JSON array) from text. Output format: [{"head_entity_text": "...", "tail_entity_text": "...", "relation_type": "..."}]. Return empty array [] if no relations found.
            
Text:
{text_segment}

Entities:
{entities_list}

Relations:
"""

    def build_re_prompt(self, text_segment: str, entities: List[Dict]) -> str:
        entities_list = "\n".join([f"- {e['text']} ({e['type']})" for e in entities])
        return self.re_prompt_template.format(
            text_segment=text_segment,
            entities_list=entities_list
        )

    def parse_re_response(self, llm_output: str, source_text_segment: str) -> List[Dict]:
        try:
            llm_output_cleaned = llm_output.strip()
            if llm_output_cleaned.startswith('```json'):
                llm_output_cleaned = llm_output_cleaned[7:-3].strip()
            obj = json.loads(llm_output_cleaned)
            if isinstance(obj, dict) and "relations" in obj:
                return obj["relations"]
            return obj if isinstance(obj, list) else []
        except Exception as e:
            logger.error(f"解析 LLM 输出时发生异常: {e}\n原始LLM输出片段: {llm_output[:200]}")
            return []

    @staticmethod
    def standardize_entity(text: str) -> str:
        return text.strip().lower()

    @staticmethod
    def rule_based_extract(text, para_entities):
        import re
        relations = []
        property_patterns = [
            (r'(屈服强度|yield strength|YS)[^\d]{0,10}([0-9]+\.?[0-9]*)\s*MPa', '屈服强度', 'MPa'),
            (r'(抗拉强度|tensile strength|UTS)[^\d]{0,10}([0-9]+\.?[0-9]*)\s*MPa', '抗拉强度', 'MPa'),
            (r'(硬度|hardness|HV)[^\d]{0,10}([0-9]+\.?[0-9]*)\s*HV', '硬度', 'HV'),
            (r'(伸长率|elongation|延伸率)[^\d]{0,10}([0-9]+\.?[0-9]*)\s*%', '伸长率', '%'),
            (r'(断裂韧性|fracture toughness|KIC)[^\d]{0,10}([0-9]+\.?[0-9]*)\s*MPa·m\^?0\.?5', '断裂韧性', 'MPa·m^0.5')
        ]

        process_patterns = [
            (r'(T6|T4|双级|单级)\s*(热处理|时效)', '热处理工艺'),
            (r'(固溶|时效|退火|淬火|回火)\s*处理', '热处理工艺'),
            (r'(\d+℃)[^\d]{0,5}(\d+[小时h])', '热处理参数')
        ]

        entity_texts = [REExtractor.standardize_entity(e['text']) for e in para_entities if 'text' in e]
        
        for pattern, prop_type, unit in property_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(2)
                for ent_text in entity_texts:
                    if ent_text == prop_type:
                        relations.append({
                            "relation_type": "EXHIBITS_PROPERTY",
                            "head_entity_text": ent_text,
                            "tail_entity_text": f"{value}{unit}",
                            "confidence_score": 0.9,
                            "source_sentence": text
                        })

        for pattern, process_type in process_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                process_desc = match.group(0)
                for ent_text in entity_texts:
                    if ent_text in process_desc:
                        relations.append({
                            "relation_type": "UNDERGOES_TREATMENT",
                            "head_entity_text": ent_text,
                            "tail_entity_text": process_desc,
                            "confidence_score": 0.85,
                            "source_sentence": text
                        })

        return relations

    def extract_relations_from_text(self, structured_text_data, entities):
        """
        输入结构化文本数据和实体列表，返回关系列表。
        优先使用规则提取，LLM提取作为补充。
        """
        all_relations = []
        chunker = utils.TextChunker(self.config)
        
        sections = structured_text_data.get('sections', [])
        for section in sections:
            for para in section.get('paragraphs', []):
                text = para.get('text', '')
                if not text.strip():
                    continue
                
                para_entities = [e for e in entities if e.get('source_text', '').strip() in text.strip()]
                if len(para_entities) < 2:  # 至少需要2个实体才能建立关系
                    continue
                
                try:
                    text_segments = chunker.chunk_text(text)
                    logger.info(f"文本分块完成，共{len(text_segments)}块")
                except Exception as e:
                    logger.error(f"文本分块失败: {e}, 回退到单段落处理")
                    text_segments = [text]
                
                for seg in text_segments:
                    # 1. 优先使用规则提取
                    rule_relations = self.rule_based_extract(seg, para_entities)
                    all_relations.extend(rule_relations)
                    
                    # 2. 如果规则提取结果不足，再使用LLM提取
                    if len(rule_relations) < len(para_entities) / 2:  # 启发式阈值
                        prompt = self.build_re_prompt(seg, para_entities)
                        response = self.llm_client.complete(prompt)
                        logger.info(f"LLM原始输出(关系提取):\n{response}")
                        llm_relations = self.parse_re_response(response, seg)
                        
                        # 验证LLM提取的关系是否使用已识别的实体
                        validated_llm_relations = []
                        for rel in llm_relations:
                            head_ent = next((e for e in para_entities 
                                           if e['text'] == rel['head_entity_text']), None)
                            tail_ent = next((e for e in para_entities 
                                           if e['text'] == rel['tail_entity_text']), None)
                            if head_ent and tail_ent:
                                validated_llm_relations.append(rel)
                        
                        # 过滤掉与规则提取重复的关系
                        unique_llm_relations = [
                            rel for rel in validated_llm_relations 
                            if not any(
                                r['head_entity_text'] == rel['head_entity_text'] and 
                                r['tail_entity_text'] == rel['tail_entity_text']
                                for r in rule_relations
                            )
                        ]
                        all_relations.extend(unique_llm_relations)
                    
                    logger.debug(f"分块文本({len(seg)}字符): {seg[:100]}...")
                    logger.debug(f"抽取到{len(rule_relations)}个规则关系 + {len(all_relations)-len(rule_relations)}个LLM关系")
        
        return all_relations
