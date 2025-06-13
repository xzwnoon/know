# src/ner_module.py

import json
from typing import List, Dict, Any, Optional
import os
# 导入同目录下的 utils 模块，其中包含 LLMClient 和 load_config
import utils
# 从 utils 模块导入 LLMClient 类
from utils import LLMClient

# 配置 ner_module 的日志记录
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # INFO 级别显示处理进度，DEBUG 更详细


class NERExtractor:
    """
    使用 LLM Prompts 处理命名实体识别 (Named Entity Recognition, NER)。
    从文本中提取铝合金领域的特定命名实体，例如材料名称、工艺、性能等。
    """

    def __init__(self, config: Dict = None):
        """
        使用配置和 LLM 客户端初始化 NERExtractor。

        Args:
            config: 配置字典，通常从 settings.yaml 加载，包含 'llm' 和 'prompts' 部分。
                    如果为 None，则尝试调用 utils.load_config() 加载默认配置。
        """
        # 如果没有提供配置，尝试加载默认配置
        if config is None:
             config = utils.load_config()
             if config is None:
                  logger.error("NERExtractor: 无法加载配置。使用一个非常基本的备用配置，功能可能受限。")
                  # 提供一个最基本的备用配置，确保类能实例化，但功能可能无法正常工作
                  config = {
                      'llm': {
                          'provider': 'dummy', # 使用 dummy 提供商
                          'model_name': 'fallback-model',
                          'api_key': 'dummy_key'
                      },
                      'prompts': {
                          'ner_template': """提取以下文本中的实体 (格式为 JSON 数组): {text_segment}。实体类型包括 MaterialAlloy, MechanicalProperty 等。输出格式: [{"text": "...", "type": "...", "attributes": {}}]"""
                      }
                  }
        self.config = config

        # 初始化 LLM 客户端，将 LLM 配置部分传递给它
        self.llm_client = LLMClient(self.config.get('llm', {}))

        # 从配置中加载 NER Prompt 模板
        self.ner_prompt_template = self.config.get('prompts', {}).get('ner_template')
        if not self.ner_prompt_template:
            logger.error("NER Prompt 模板未在配置中找到 (key 'prompts' -> 'ner_template')。使用一个基本的备用模板。")
            # 提供一个基本的备用 Prompt 模板
            self.ner_prompt_template = """Extract entities (JSON array) from the following text. Entity types: MaterialAlloy, AlloyComponent, HeatTreatmentProcess, ResearchSubject, ResearchMethod, MechanicalProperty, MicrostructureFeature, Application. 
Output format: [{"text": "...", "type": "...", "attributes": {}}]. 如无实体请返回空数组[]，不要省略任何字段。

Text:
{text_segment}

Entities:
"""

        # TODO: 如果 Prompt 设计使用了 Few-shot Examples，可以在这里加载
        # few_shot_examples 的结构取决于 LLM API 的 Few-shot 机制，例如对于 Chat 模型是消息列表
        self.few_shot_examples = self.config.get('prompts', {}).get('ner_few_shot_examples', []) # 假定配置中可能包含此键


        logger.info("NERExtractor 初始化成功。")

    def build_ner_prompt(self, text_segment: str, few_shot_examples: List[Dict] = None) -> str:
        """
        构建用于发送给 LLM 的 NER Prompt 字符串。

        Args:
            text_segment: 需要提取实体的文本片段（通常是一个段落）。
            few_shot_examples: (可选) 用于 Few-shot Learning 的示例列表。

        Returns:
            格式化好的 Prompt 字符串。
        """
        try:
            # 预处理文本片段 - 移除多余空白和特殊字符
            cleaned_text = text_segment.strip()
            cleaned_text = ' '.join(cleaned_text.split())  # 合并多个空白
            
            # 确保文本片段不为空
            if not cleaned_text:
                logger.warning("构建NER Prompt时传入空文本片段")
                cleaned_text = "无内容"

            # 使用配置中的 Prompt 模板格式化文本片段
            prompt = self.ner_prompt_template.format(text_segment=cleaned_text)
            
            # 验证Prompt是否包含文本片段
            if cleaned_text not in prompt:
                logger.error(f"Prompt格式化失败，文本片段未正确插入。模板可能缺少{{text_segment}}占位符")
                return f"{self.ner_prompt_template}\n\n文本片段:\n{cleaned_text}"

            return prompt
        except KeyError as e:
            logger.error(f"Prompt模板缺少必要占位符: {e}")
            return f"{self.ner_prompt_template}\n\n文本片段:\n{text_segment}"
        except Exception as e:
            logger.error(f"构建NER Prompt时发生异常: {e}")
            return f"请从以下文本中提取实体:\n{text_segment}"

    def parse_ner_response(self, llm_output: str, source_text_segment: str) -> List[Dict]:
        """
        解析 LLM 返回的原始输出字符串（期望是 JSON 格式）为实体字典列表。
        进行严格的格式验证，并为每个实体添加源文本信息。
        支持多行/多对象JSON的健壮解析。
        """
        import re
        def robust_json_parse(llm_output: str):
            # 空字符串直接返回空
            if not llm_output or not llm_output.strip():
                logger.warning("LLM输出为空")
                return []
            
            # 预处理：移除可能的代码块标记
            text = llm_output.strip()
            if text.startswith('```json'):
                text = text.replace('```json', '').replace('```', '').strip()
            
            # 1. 先尝试标准JSON数组
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    if "entities" in obj:  # 处理{"entities": [...]}格式
                        return obj["entities"]
                    return [obj]
                return obj if isinstance(obj, list) else []
            except json.JSONDecodeError as e:
                logger.debug(f"标准JSON解析失败: {e}")
            
            # 2. 尝试补全为数组
            text = re.sub(r'^[,\s]+|[,\s]+$', '', text)
            if text.startswith('{') and not text.startswith('['):
                text = f'[{text}]'
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return [obj]
                return obj if isinstance(obj, list) else []
            except json.JSONDecodeError as e:
                logger.debug(f"补全JSON解析失败: {e}")
            
            # 3. 精确处理多行对象
            matches = re.findall(r'\{[\s\S]*?\}', text)
            objs = []
            for m in matches:
                try:
                    obj = json.loads(m)
                    # 严格验证必需字段
                    if (isinstance(obj, dict) and 
                        'text' in obj and isinstance(obj['text'], str) and
                        'type' in obj and isinstance(obj['type'], str)):
                        obj.setdefault('attributes', {})
                        objs.append(obj)
                except Exception as e:
                    logger.debug(f"多行正则JSON解析失败: {e}, 内容: {m[:100]}...")
                    continue
            if objs:
                return objs
                
            # 4. 按行提取每个对象
            lines = [l.strip().rstrip(',') for l in text.splitlines() if l.strip()]
            for line in lines:
                if not line or not line.startswith('{'):
                    continue
                try:
                    obj = json.loads(line)
                    # 严格验证必需字段
                    if (isinstance(obj, dict) and 
                        'text' in obj and isinstance(obj['text'], str) and
                        'type' in obj and isinstance(obj['type'], str)):
                        obj.setdefault('attributes', {})
                        objs.append(obj)
                except Exception as e:
                    logger.debug(f"单行JSON解析失败: {e}, 内容: {line[:100]}...")
                    continue
                    
            # 5. 全部失败，返回空并记录原始输出
            if not objs:
                logger.error(f"无法解析LLM输出:\n{llm_output[:500]}...")
            return objs

        try:
            llm_output_cleaned = llm_output.strip()
            if llm_output_cleaned.startswith('```json'):
                llm_output_cleaned = llm_output_cleaned.replace('```json', '').replace('```', '').strip()
            
            # 1. 尝试直接解析为JSON
            try:
                obj = json.loads(llm_output_cleaned)
                if isinstance(obj, dict) and "entities" in obj:
                    # 处理 {"entities": [...]} 格式
                    entities = obj["entities"]
                    if not isinstance(entities, list):
                        entities = [entities]
                elif isinstance(obj, list):
                    # 处理直接返回数组的情况
                    entities = obj
                elif isinstance(obj, dict):
                    # 处理单个实体对象
                    entities = [obj]
                else:
                    entities = []
            except Exception:
                # 2. 尝试健壮解析
                entities = robust_json_parse(llm_output_cleaned)
            
            # 验证和格式化结果
            result = []
            for entity in entities:
                if isinstance(entity, dict):
                    # 确保必需字段存在
                    if "text" not in entity or "type" not in entity:
                        continue
                    # 添加源文本信息
                    entity['source_text'] = source_text_segment
                    # 确保attributes存在
                    if "attributes" not in entity:
                        entity["attributes"] = {}
                    result.append(entity)
            
            if not result:
                logger.warning(f"NER解析返回空，原始LLM输出片段: {llm_output_cleaned[:200]}")
                # 明确返回空数组，保持格式一致
                return []
            
            return result
        except Exception as e:
            logger.error(f"解析 LLM 输出时发生异常: {e}\n原始LLM输出片段: {llm_output[:200]}")
            return []

    def extract_entities_from_text(self, text_data: Dict) -> List[Dict]:
        """
        从结构化文本数据中抽取实体，特别关注材料性能参数和热处理工艺
        
        Args:
            text_data: 包含sections和paragraphs的字典，或直接包含文本的分块
            
        Returns:
            实体列表
        """
        if not text_data or not isinstance(text_data, dict):
            return []
            
        # 处理两种格式：
        # 1. 完整结构: {'sections': [{'paragraphs': [{'text': ...}]}]}
        # 2. 分块结构: {'sections': [{'paragraphs': [{'text': ...}]}]} (由TextChunker生成)
        paragraphs = []
        if 'sections' in text_data:
            for section in text_data.get('sections', []):
                for para in section.get('paragraphs', []):
                    if isinstance(para, dict) and 'text' in para:
                        paragraphs.append(para['text'])
                    elif isinstance(para, str):
                        paragraphs.append(para)
        elif 'text' in text_data:
            paragraphs.append(text_data['text'])
            
        if not paragraphs:
            return []
            
        all_entities = []
        chunker = utils.TextChunker(self.config)
        
        for text in paragraphs:
            if not text.strip():
                continue
                
            try:
                # 使用TextChunker分块
                chunks = chunker.chunk_text(text)
                logger.debug(f"文本分块完成，共{len(chunks)}块")
                
                for chunk in chunks:
                    # 处理分块格式
                    chunk_text = ""
                    if isinstance(chunk, dict) and 'sections' in chunk:
                        for s in chunk['sections']:
                            for p in s.get('paragraphs', []):
                                if isinstance(p, dict) and 'text' in p:
                                    chunk_text += p['text'] + "\n"
                                elif isinstance(p, str):
                                    chunk_text += p + "\n"
                    elif isinstance(chunk, str):
                        chunk_text = chunk
                        
                    if not chunk_text.strip():
                        continue
                        
                    # 1. 首先尝试使用LLM提取实体
                    prompt = self.build_ner_prompt(chunk_text)
                    response = self.llm_client.complete(prompt)
                    logger.info(f"LLM原始输出:\n{response}")  # 记录原始输出
                    entities = self.parse_ner_response(response, chunk_text)
                    
                    # 2. 对性能参数和热处理工艺进行后处理增强
                    enhanced_entities = self.enhance_property_and_process_entities(entities, chunk_text)
                    all_entities.extend(enhanced_entities)
                    
                    logger.debug(f"分块文本({len(chunk_text)}字符): {chunk_text[:100]}...")
                    logger.debug(f"抽取到{len(enhanced_entities)}个实体(增强后)")
                    
            except Exception as e:
                logger.error(f"文本分块处理失败: {e}, 回退到整段处理")
                prompt = self.build_ner_prompt(text)
                response = self.llm_client.complete(prompt)
                logger.info(f"LLM原始输出(回退模式):\n{response}")  # 记录原始输出
                entities = self.parse_ner_response(response, text)
                enhanced_entities = self.enhance_property_and_process_entities(entities, text)
                all_entities.extend(enhanced_entities)
        
        return all_entities

    def enhance_property_and_process_entities(self, entities: List[Dict], source_text: str) -> List[Dict]:
        """
        增强性能参数和热处理工艺实体的提取
        
        Args:
            entities: 原始提取的实体列表
            source_text: 原始文本
            
        Returns:
            增强后的实体列表
        """
        import re
        
        # 1. 首先确保所有性能参数和热处理工艺实体都被正确识别
        property_patterns = [
            (r'(屈服强度|yield strength|YS)[^\d]{0,10}([0-9]+\.?[0-9]*)\s*MPa', 'MechanicalProperty', '屈服强度'),
            (r'(抗拉强度|tensile strength|UTS)[^\d]{0,10}([0-9]+\.?[0-9]*)\s*MPa', 'MechanicalProperty', '抗拉强度'),
            (r'(硬度|hardness|HV)[^\d]{0,10}([0-9]+\.?[0-9]*)\s*HV', 'MechanicalProperty', '硬度'),
            (r'(伸长率|elongation|延伸率)[^\d]{0,10}([0-9]+\.?[0-9]*)\s*%', 'MechanicalProperty', '伸长率')
        ]
        
        process_patterns = [
            (r'(T\d+|双级|单级)\s*(热处理|时效)', 'HeatTreatmentProcess'),
            (r'(固溶|时效|退火|淬火|回火)\s*处理', 'HeatTreatmentProcess'),
            (r'(\d+℃)[^\d]{0,5}(\d+[小时h])', 'HeatTreatmentProcess')
        ]
        
        enhanced_entities = entities.copy()
        
        # 2. 检查性能参数
        for pattern, ent_type, prop_name in property_patterns:
            matches = re.finditer(pattern, source_text, re.IGNORECASE)
            for match in matches:
                value = match.group(2)
                unit = match.group(3) if len(match.groups()) > 2 else None
                text = match.group(0)
                
                # 检查是否已存在该实体
                exists = any(e['text'] == text and e['type'] == ent_type for e in enhanced_entities)
                if not exists:
                    enhanced_entities.append({
                        'text': text,
                        'type': ent_type,
                        'attributes': {
                            'name': prop_name,
                            'value': value,
                            'unit': unit or 'MPa'
                        },
                        'source_text': source_text
                    })
        
        # 3. 检查热处理工艺
        for pattern, ent_type in process_patterns:
            matches = re.finditer(pattern, source_text, re.IGNORECASE)
            for match in matches:
                text = match.group(0)
                exists = any(e['text'] == text and e['type'] == ent_type for e in enhanced_entities)
                if not exists:
                    enhanced_entities.append({
                        'text': text,
                        'type': ent_type,
                        'attributes': {
                            'description': text
                        },
                        'source_text': source_text
                    })
        
        return enhanced_entities
