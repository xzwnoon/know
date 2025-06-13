# src/utils.py
import yaml
import os
import logging
import json
import requests
import time
from typing import List, Dict, Any, Optional
import re # 用于LLMClient的dummy模式模拟解析Prompt

# 配置 utils 模块的日志记录
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # INFO 级别会显示重要信息，DEBUG 会显示更详细过程


def load_config(config_path="config/settings.yaml"):
    """
    从 YAML 文件加载配置。

    该函数会尝试智能地找到项目根目录，并从那里加载配置文件。
    这使得即使脚本在子目录中运行，也能找到位于项目根目录的配置文件。

    Args:
        config_path: 相对于项目根目录的配置文件路径。默认为 'config/settings.yaml'。

    Returns:
        包含配置的字典，如果加载失败则返回 None。
    """
    try:
        # 尝试通过寻找特定文件（如 main_pipeline.py）来确定项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        # 向上遍历目录，直到找到一个包含 'main_pipeline.py' 的目录
        # 设置一个最大深度，防止无限向上遍历
        max_depth = 5
        for _ in range(max_depth):
            if os.path.exists(os.path.join(project_root, 'main_pipeline.py')):
                break
            parent_dir = os.path.dirname(project_root)
            if parent_dir == project_root: # 已经到达文件系统根目录
                break
            project_root = parent_dir

        # 构建配置文件的绝对路径
        absolute_config_path = os.path.join(project_root, config_path)
        # logger.debug(f"尝试从以下路径加载配置: {absolute_config_path}") # 调试信息

        if not os.path.exists(absolute_config_path):
            logger.error(f"错误: 配置文件未找到于 {absolute_config_path}")
            # 备选方案：如果绝对路径找不到，尝试相对于当前工作目录加载
            if os.path.exists(config_path):
                logger.warning(f"未在项目根目录路径找到配置，尝试相对于当前工作目录加载: {config_path}")
                absolute_config_path = config_path
            else:
                 return None # 彻底失败

        with open(absolute_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件从 {absolute_config_path} 加载成功。")
        return config
    except FileNotFoundError:
        logger.error(f"错误: 配置文件未找到。检查路径 {config_path}。")
        return None
    except yaml.YAMLError as e:
        logger.error(f"加载配置文件时发生YAML解析错误: {e}")
        return None
    except Exception as e:
        logger.error(f"加载配置文件时发生意外错误: {e}")
        return None

def get_path(config: dict, key: str, default: str = None) -> str:
    """
    从配置中获取路径型参数，支持多级（如 'paths.kg_data_dir'），并自动展开用户目录和绝对路径。
    """
    if not config or not key:
        return default
    # 支持多级 key，如 paths.kg_data_dir
    keys = key.split('.')
    val = config
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    # 展开 ~ 和绝对路径
    if isinstance(val, str):
        val = os.path.expanduser(val)
        val = os.path.abspath(val)
    return val

def get_param(config: dict, key: str, default: Any = None) -> Any:
    """
    从配置中获取任意参数，支持多级 key，如 'llm.temperature'。
    """
    if not config or not key:
        return default
    keys = key.split('.')
    val = config
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val

class LLMClient:
    """
    LLM API 调用包装类。
    负责根据配置向不同的LLM服务发起请求（如OpenAI API），并处理基本的响应。
    """
    def __init__(self, config: Dict):
        """
        使用LLM相关配置初始化客户端。

        Args:
            config: 配置字典，通常是 settings.yaml 中 'llm' 部分的内容。
        """
        self.provider = config.get('provider', 'dummy')
        self.model_name = config.get('model_name', 'dummy-model')
        self.api_key = config.get('api_key') # API Key 可以为 None
        self.base_url = config.get('base_url', None)
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 1024)
        self.request_timeout = config.get('request_timeout', 60) # 秒

        # 根据配置的 provider 初始化具体的客户端或逻辑
        if self.provider == 'openai':
            try:
                from openai import OpenAI
                # Explicitly set base_url if provided in config
                # Set timeout explicitly
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.request_timeout)
                logger.info(f"LLMClient: 初始化 OpenAI 客户端，模型: {self.model_name}, base_url: {self.base_url or '默认'}。")
            except ImportError:
                logger.error("LLMClient: 错误：未安装 'openai' 库。无法使用 'openai' 提供商。请安装：pip install openai")
                self.provider = 'dummy' # 回退到 dummy 提供商
            except Exception as e:
                 logger.error(f"LLMClient: 初始化 OpenAI 客户端时发生错误: {e}。回退到 dummy 提供商。")
                 self.provider = 'dummy' # 回退到 dummy 提供商

        # TODO: 未来可以添加对 Anthropic, Hugging Face 等其他提供商的支持
        # elif self.provider == 'anthropic':
        #     try:
        #         from anthropic import Anthropic
        #         self.client = Anthropic(api_key=self.api_key, base_url=self.base_url, timeout=self.request_timeout)
        #         logger.info(f"LLMClient: Initialized Anthropic client with model: {self.model_name}")
        #     except ImportError:
        #         logger.error("LLMClient: Error: 'anthropic' library not installed. Cannot use 'anthropic' provider.")
        #         self.provider = 'dummy'

        # 如果 provider 是 'dummy' 或者初始化失败
        if self.provider == 'dummy':
            logger.warning("LLMClient: 使用 Dummy LLM 客户端。不会进行实际API调用。")
        elif self.provider not in ['openai']: # 明确列出支持的提供商
             logger.warning(f"LLMClient: 警告：不支持的 LLM 提供商 '{self.provider}'。回退到 dummy 提供商。")
             self.provider = 'dummy'


    def _chat_complete(self, messages: List[Dict], response_format: Dict = {"type": "text"}, retries: int = 3, delay: int = 5) -> Optional[str]:
        """处理基于聊天的LLM API调用（如OpenAI ChatCompletion），包含重试逻辑。"""
        if self.provider != 'openai':
            # 如果提供商不支持聊天模式，或者客户端未成功初始化
            logger.error(f"LLMClient: 提供商 '{self.provider}' 不支持聊天完成模式，或客户端未就绪。")
            return None

        import httpx
        import openai
        for i in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format=response_format,
                    timeout=self.request_timeout
                )
                llm_output = response.choices[0].message.content.strip()
                logger.debug(f"LLMClient: API 调用成功 (尝试 {i+1}/{retries})。")
                return llm_output
            except (openai.error.OpenAIError, httpx.HTTPError, httpx.TimeoutException, Exception) as e:
                logger.error(f"LLMClient: API 调用失败 (尝试 {i+1}/{retries}): {e}")
                if i < retries - 1:
                    logger.info(f"LLMClient: 等待 {delay} 秒后重试...")
                    time.sleep(delay)
                else:
                    logger.error("LLMClient: 达到最大重试次数。放弃。")
                    return None # 重试失败后返回 None

    def complete(self, prompt: str, retries: int = 3, delay: int = 5) -> str:
        """
        向LLM发起请求并返回原始文本响应。

        Args:
            prompt: 发送给LLM的Prompt字符串。
            retries: 调用失败时的重试次数。
            delay: 每次重试之间的等待时间（秒）。

        Returns:
            LLM的原始字符串输出。如果调用失败（包括重试后），返回空字符串或代表失败的JSON格式（如 "[]"）。
        """
        logger.info(f"LLMClient: 调用 LLM ({self.provider}/{self.model_name})...")
        # logger.debug(f"LLMClient: Prompt 开头片段: '{prompt[:200]}...'") # 调试打印Prompt开头

        try:
            if self.provider == 'openai':
                # 对于 OpenAI，使用聊天完成API，Prompt 作为用户消息
                messages = [
                    {"role": "system", "content": "你是一个知识抽取助手，严格按照用户指令和格式输出JSON。"},
                    {"role": "user", "content": prompt}
                ]
                # 请求 JSON 对象输出格式（如果模型支持）
                response_format = {"type": "json_object"}
                llm_output = self._chat_complete(messages, response_format=response_format, retries=retries, delay=delay)
                # 如果 _chat_complete 返回 None (表示失败)，返回空 JSON 数组以避免下游模块解析出错
                return llm_output if llm_output is not None else "[]"

            elif self.provider == 'deepseek':
                # DeepSeek API调用实现
                if not self.base_url:
                    logger.error("LLMClient: DeepSeek base_url 未配置！")
                    return "[]"
                url = self.base_url.rstrip('/') + '/chat/completions'
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}'
                }
                data = {
                    'model': self.model_name,
                    'messages': [
                        {"role": "system", "content": "你是一个知识抽取助手，严格按照用户指令和格式输出JSON。"},
                        {"role": "user", "content": prompt}
                    ],
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                    'response_format': {"type": "json_object"}
                }
                
                logger.info(f"DeepSeek API请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
                
                for i in range(retries):
                    try:
                        logger.info(f"尝试DeepSeek API调用 (尝试 {i+1}/{retries})...")
                        resp = requests.post(url, headers=headers, json=data, timeout=self.request_timeout)
                        logger.info(f"DeepSeek API响应状态码: {resp.status_code}")
                        logger.info(f"DeepSeek API响应内容: {resp.text[:500]}...")
                        
                        resp.raise_for_status()
                        result = resp.json()
                        logger.info(f"DeepSeek API解析后的JSON结果: {json.dumps(result, ensure_ascii=False, indent=2)[:500]}...")
                        
                        if 'choices' in result and result['choices']:
                            content = result['choices'][0]['message']['content']
                            logger.info(f"DeepSeek API返回内容: {content}")
                            
                            # 确保返回的是有效的JSON字符串
                            try:
                                parsed = json.loads(content)
                                if not isinstance(parsed, (dict, list)):
                                    logger.error(f"LLMClient: DeepSeek返回的不是JSON对象或数组: {content}")
                                    return "[]"
                                logger.info(f"DeepSeek API返回有效JSON: {json.dumps(parsed, ensure_ascii=False, indent=2)[:500]}...")
                                return content
                            except json.JSONDecodeError as je:
                                logger.error(f"LLMClient: DeepSeek返回的不是有效JSON: {je}\n内容: {content}")
                                return "[]"
                        else:
                            logger.error(f"LLMClient: DeepSeek响应格式异常: {result}")
                            return "[]"
                    except requests.exceptions.RequestException as re:
                        logger.error(f"LLMClient: DeepSeek API请求异常 (尝试 {i+1}/{retries}): {re}")
                        if i < retries - 1:
                            logger.info(f"LLMClient: 等待 {delay} 秒后重试...")
                            time.sleep(delay)
                        else:
                            logger.error("LLMClient: 达到最大重试次数。放弃。")
                            return "[]"
                    except Exception as e:
                        logger.error(f"LLMClient: DeepSeek API处理异常 (尝试 {i+1}/{retries}): {e}")
                        if i < retries - 1:
                            logger.info(f"LLMClient: 等待 {delay} 秒后重试...")
                            time.sleep(delay)
                        else:
                            logger.error("LLMClient: 达到最大重试次数。放弃。")
                            return "[]"

            elif self.provider == 'dummy':
                # 模拟一个 JSON 格式的响应
                logger.info("LLMClient: ... 模拟 Dummy LLM 响应 ...")
                # Dummy 响应需要根据 Prompt 的类型（NER 或 RE）生成模拟数据
                # 这里我们做个简单的判断：如果 Prompt 中包含 "关系" 或 "实体"，则模拟一个NER或RE的响应
                if re.search(r'关系|实体', prompt):
                    # 模拟一个NER或RE的响应
                    dummy_response = {
                        "entities": [
                            {"text": "示例实体", "type": "示例类型", "start": 0, "end": 4}
                        ],
                        "relations": [
                            {"subject": "示例实体", "object": "另一个实体", "relation": "示例关系"}
                        ]
                    }
                    return json.dumps(dummy_response, ensure_ascii=False)
                else:
                    # 返回一个空 JSON 数组
                    return "[]"
        except Exception as e:
            logger.error(f"LLMClient: complete() 调用异常: {e}")
            return "[]"

def load_few_shot_examples(config: dict = None, file_path: str = None) -> list:
    """
    动态加载few-shot示例，优先从配置指定路径，其次默认kg_data/few_shot_examples.json。
    """
    # 优先从配置读取路径
    if config:
        from_path = get_path(config, 'paths.few_shot_examples', None)
        if from_path and os.path.exists(from_path):
            file_path = from_path
    if not file_path:
        # 默认路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.abspath(os.path.join(current_dir, '../kg_data/few_shot_examples.json'))
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                logger.info(f"成功加载few-shot示例: {file_path}，共{len(data)}条。")
                return data
        except Exception as e:
            logger.warning(f"加载few-shot示例失败: {e}")
    return []

def append_few_shot_example(example: dict, config: dict = None, file_path: str = None) -> bool:
    """
    追加一条few-shot示例到few_shot_examples.json，便于动态采样和持续优化。
    """
    # 路径同上
    if config:
        from_path = get_path(config, 'paths.few_shot_examples', None)
        if from_path:
            file_path = from_path
    if not file_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.abspath(os.path.join(current_dir, '../kg_data/few_shot_examples.json'))
    # 加载现有
    data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = []
    if not isinstance(data, list):
        data = []
    data.append(example)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"few-shot示例已追加到: {file_path}")
        return True
    except Exception as e:
        logger.error(f"写入few-shot示例失败: {e}")
        return False

def handle_multiple_relations(relations: List[Dict], strategy: str = "keep_all") -> List[Dict]:
    """
    处理同一对实体间的多种关系，根据策略保留或筛选
    
    Args:
        relations: 关系列表
        strategy: 处理策略 - "keep_all"(保留全部), "priority"(按优先级筛选)
    
    Returns:
        处理后的关系列表
    """
    if strategy == "keep_all":
        return relations
        
    # 按优先级筛选
    priority_order = {
        "HAS_COMPONENT": 1,
        "EXHIBITS_PROPERTY": 2,
        "UNDERGOES_TREATMENT": 3,
        "HAS_MICROSTRUCTURE": 4,
        "HAS_APPLICATION": 5,
        "INVESTIGATED_BY": 6
    }
    
    # 按实体对分组
    relation_groups = {}
    for rel in relations:
        key = (rel["head_entity_text"], rel["tail_entity_text"])
        if key not in relation_groups:
            relation_groups[key] = []
        relation_groups[key].append(rel)
    
    # 对每组应用策略
    processed = []
    for group in relation_groups.values():
        if len(group) == 1:
            processed.extend(group)
            continue
            
        if strategy == "priority":
            # 找出优先级最高的关系
            highest_priority = min(group, key=lambda x: priority_order.get(x["relation_type"], 99))
            processed.append(highest_priority)
    
    return processed

def track_relation_source(relations: List[Dict], source: str = "LLM") -> List[Dict]:
    """
    为关系添加来源信息
    
    Args:
        relations: 关系列表
        source: 来源 - "LLM", "rule", "manual"
    
    Returns:
        添加了来源信息的关系列表
    """
    for rel in relations:
        if "attributes" not in rel:
            rel["attributes"] = {}
        rel["attributes"]["source"] = source
    return relations


class TextChunker:
    """
    文本分块工具类，提供多种分块方法：
    - TextTiling: 基于语义的分块方法
    - BERT: 基于BERT嵌入相似度的分块方法
    - Fixed: 固定长度分块
    - PDF: 针对PDF文档的智能分块
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化分块器
        
        Args:
            config: 配置字典，包含分块相关参数
        """
        self.config = config or {}
        self.chunk_method = self.config.get("chunk_method", "pdf")  # 默认改为PDF分块
        self.max_chunk_size = self.config.get("max_chunk_size", 2000)  # 增大默认块大小
        self.min_chunk_size = self.config.get("min_chunk_size", 500)   # 增大最小块大小
        self.overlap_size = self.config.get("overlap_size", 200)       # 增大重叠大小
        self.pdf_structure = self.config.get("pdf_structure", True)    # 是否使用PDF结构
        
        # 初始化TextTiling
        try:
            from nltk.tokenize import TextTilingTokenizer
            self.tt = TextTilingTokenizer()
        except ImportError:
            logger.warning("TextTilingTokenizer不可用，请安装nltk")
            self.tt = None
            
        # 初始化BERT模型
        try:
            from sentence_transformers import SentenceTransformer
            self.bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except ImportError:
            logger.warning("SentenceTransformer不可用，请安装sentence-transformers")
            self.bert_model = None
    
    def chunk_text(self, text: str, method: str = None, is_pdf: bool = False) -> List[Dict]:
        """
        对文本进行分块，返回符合NER模块要求的格式
        
        Args:
            text: 输入文本
            method: 分块方法(texttiling/bert/fixed/pdf)
            is_pdf: 是否为PDF文档
            
        Returns:
            分块后的文本列表，每个元素是包含section和paragraphs的字典
            [{
                "sections": [{
                    "title": "章节标题",
                    "paragraphs": [{"text": "段落内容"}]
                }]
            }]
        """
        method = method or self.chunk_method
        if is_pdf and self.pdf_structure:
            return self._pdf_chunk(text)
        elif method == "texttiling" and self.tt:
            return self._texttiling_chunk(text)
        elif method == "bert" and self.bert_model:
            return self._bert_chunk(text)
        else:
            return self._fixed_chunk(text)
    
    def _texttiling_chunk(self, text: str) -> List[Dict]:
        """
        使用TextTiling算法进行语义分块，返回符合NER模块要求的格式
        
        Args:
            text: 输入文本
            
        Returns:
            分块后的文本列表，每个元素是包含section和paragraphs的字典
            [{
                "sections": [{
                    "title": "",
                    "paragraphs": [{"text": "分块内容"}]
                }]
            }]
        """
        try:
            # 预处理文本：确保段落之间有足够的换行符
            paragraphs = text.split('\n')
            processed_text = '\n\n'.join([p.strip() for p in paragraphs if p.strip()])
            chunks = self.tt.tokenize(processed_text)
            
            # 转换为统一格式
            formatted_chunks = []
            for chunk in chunks:
                formatted_chunks.append({
                    "sections": [{
                        "title": "",
                        "paragraphs": [{"text": chunk}]
                    }]
                })
            return formatted_chunks
        except Exception as e:
            logger.warning(f"TextTiling分块失败: {e}, 回退到固定分块")
            return self._fixed_chunk(text)
    
    def _bert_chunk(self, text: str) -> List[Dict]:
        """
        使用BERT嵌入相似度进行语义分块，返回符合NER模块要求的格式
        
        Args:
            text: 输入文本
            
        Returns:
            分块后的文本列表，每个元素是包含section和paragraphs的字典
            [{
                "sections": [{
                    "title": "",
                    "paragraphs": [{"text": "分块内容"}]
                }]
            }]
        """
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 分割句子
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                return []
                
            # 计算句子嵌入
            embeddings = self.bert_model.encode(sentences)
            
            # 计算相邻句子相似度
            similarities = []
            for i in range(1, len(embeddings)):
                sim = cosine_similarity(
                    embeddings[i-1].reshape(1, -1), 
                    embeddings[i].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
            
            # 基于相似度阈值分块
            threshold = np.mean(similarities) - np.std(similarities)
            chunks = []
            current_chunk = [sentences[0]]
            
            for i in range(1, len(sentences)):
                if similarities[i-1] > threshold:
                    current_chunk.append(sentences[i])
                else:
                    chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = [sentences[i]]
            
            if current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
            
            # 转换为统一格式
            formatted_chunks = []
            for chunk in chunks:
                formatted_chunks.append({
                    "sections": [{
                        "title": "",
                        "paragraphs": [{"text": chunk}]
                    }]
                })
            return formatted_chunks
        except Exception as e:
            logger.warning(f"BERT分块失败: {e}, 回退到固定分块")
            return self._fixed_chunk(text)
    
    def _pdf_chunk(self, text: str) -> List[Dict]:
        """
        增强版PDF智能分块，特别针对学术论文结构优化
        
        Args:
            text: 输入文本
            
        Returns:
            分块后的文本列表，保留PDF结构，格式符合NER模块要求
        """
        try:
            # 学术论文常见结构分割
            section_pattern = r'(\n\s*(?:ABSTRACT|摘要|INTRODUCTION|介绍|METHODS?|方法|RESULTS?|结果|DISCUSSION|讨论|CONCLUSION|结论|REFERENCES?|参考文献)\s*\n)'
            sections = re.split(section_pattern, text)
            
            # 第一个元素可能是空字符串或前言内容
            if sections and not sections[0].strip():
                sections = sections[1:]
            
            chunks = []
            current_title = ""
            
            for i in range(0, len(sections), 2):
                if i+1 < len(sections):
                    # 奇数索引是标题，偶数索引是内容
                    title = sections[i].strip()
                    content = sections[i+1].strip()
                else:
                    # 最后一个不完整部分
                    title = ""
                    content = sections[i].strip()
                
                if not content:
                    continue
                    
                # 处理子章节(如1.1, 1.2等)
                sub_sections = re.split(r'(\n\s*\d+\.\d+\s+.+?\n)', content)
                sub_content = []
                
                for j in range(0, len(sub_sections), 2):
                    if j+1 < len(sub_sections):
                        sub_title = sub_sections[j].strip() + sub_sections[j+1].strip()
                        sub_text = sub_sections[j+2].strip() if j+2 < len(sub_sections) else ""
                    else:
                        sub_title = ""
                        sub_text = sub_sections[j].strip()
                    
                    if sub_text:
                        # 按段落分割并合并
                        paragraphs = [p.strip() for p in sub_text.split('\n\n') if p.strip()]
                        merged_paragraphs = []
                        current_para = ""
                        
                        for para in paragraphs:
                            para_len = len(para)
                            if current_para and len(current_para) + para_len > self.max_chunk_size:
                                merged_paragraphs.append({"text": current_para})
                                current_para = para
                            else:
                                current_para += "\n\n" + para if current_para else para
                        
                        if current_para:
                            merged_paragraphs.append({"text": current_para})
                        
                        if merged_paragraphs:
                            full_title = (title + " " + sub_title).strip()
                            # 确保分块大小在合理范围内
                            if len(current_para) > self.max_chunk_size * 1.5:
                                # 对过大的段落进行二次分块
                                fixed_chunks = self._fixed_chunk(current_para)
                                for fc in fixed_chunks:
                                    chunks.append({
                                        "sections": [{
                                            "title": full_title,
                                            "paragraphs": fc["sections"][0]["paragraphs"]
                                        }]
                                    })
                            else:
                                chunks.append({
                                    "sections": [{
                                        "title": full_title,
                                        "paragraphs": merged_paragraphs
                                    }]
                                })
                
            return chunks if chunks else self._fixed_chunk(text)
            
        except Exception as e:
            logger.error(f"PDF分块失败: {str(e)[:200]}...", exc_info=True)
            logger.info("回退到固定分块方法")
            return self._fixed_chunk(text)

    def _fixed_chunk(self, text: str) -> List[Dict]:
        """
        优化后的固定长度分块，返回符合NER模块要求的格式
        
        Args:
            text: 输入文本
            
        Returns:
            分块后的文本列表，每个元素是包含section和paragraphs的字典
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        # 先尝试按自然段落分割
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_len = len(para)
            if current_size + para_len > self.max_chunk_size and current_chunk:
                # 当前段落会使块过大，先保存当前块
                chunks.append({
                    "sections": [{
                        "title": "",
                        "paragraphs": [{"text": "\n\n".join(current_chunk)}]
                    }]
                })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_len
            
            if current_size >= self.min_chunk_size:
                # 达到最小块大小，可以保存
                chunks.append({
                    "sections": [{
                        "title": "",
                        "paragraphs": [{"text": "\n\n".join(current_chunk)}]
                    }]
                })
                current_chunk = []
                current_size = 0
        
        # 处理剩余内容
        if current_chunk:
            chunks.append({
                "sections": [{
                    "title": "",
                    "paragraphs": [{"text": "\n\n".join(current_chunk)}]
                }]
            })
            
        return chunks
