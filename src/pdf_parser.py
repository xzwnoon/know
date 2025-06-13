# src/pdf_parser.py
import fitz  # PyMuPDF 库，用于处理 PDF
import os
import re
from typing import List, Dict, Any
# 导入同目录下的 utils 模块，用于加载配置
import utils

# 配置 pdf_parser 模块的日志记录
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # INFO 级别显示处理进度，DEBUG 更详细


class PDFParser:
    """
    处理PDF文档的加载、文本提取和基本清洗。

    该类负责从PDF中获取原始文本块，进行预处理，并尝试将其初步结构化（合并段落）。
    它不负责识别复杂的文档结构（如标题、摘要、章节），这部分可以在后续处理或这里更复杂的逻辑中实现。
    """

    def __init__(self, config: Dict):
        """
        使用配置初始化 PDFParser。

        Args:
            config: 项目的配置字典，从 settings.yaml 加载。
                    可能包含 PDF 解析相关的参数，例如页眉页脚检测的阈值。
        """
        self.config = config
        # 根据经验设置一些简单的阈值，用于判断页眉页脚和段落边界。
        # 这些是基于页面高度的比例，更具通用性，但可能需要根据实际文档调整。
        # 距离页面顶部或底部小于总高度的 5% 的块，可能是页眉/页脚
        self.header_footer_threshold_y_ratio = 0.05
        # 用于判断是否开始新段落的垂直距离阈值。如果块之间的垂直距离大于此比例*页面高度，可能是一个新段落。
        self.line_spacing_threshold_ratio = 0.01 # 1% of page height


    def load_pdf(self, file_path: str) -> fitz.Document:
        """
        加载一个PDF文件。

        Args:
            file_path: PDF文件的完整路径。

        Returns:
            一个 fitz.Document 对象，代表加载的PDF文档。

        Raises:
            FileNotFoundError: 如果指定路径的文件不存在。
            Exception: 对于 PyMuPDF 在加载文件时发生的其他任何错误。
        """
        if not os.path.exists(file_path):
            logger.error(f"PDF 文件未找到: {file_path}")
            raise FileNotFoundError(f"PDF 文件未找到: {file_path}")
        try:
            # fitz.open() 加载文档。调用者有责任在处理完成后关闭文档。
            doc = fitz.open(file_path)
            logger.info(f"成功加载 PDF: {file_path}")
            return doc
        except Exception as e:
            logger.error(f"加载 PDF 文件 {file_path} 时发生错误: {e}")
            raise # 重新抛出异常，让调用方知晓加载失败


    def extract_blocks_with_layout(self, doc: fitz.Document) -> List[Dict]:
        """
        从PDF文档中提取带有布局信息的文本块。

        使用 PyMuPDF 的 'dict' 输出格式，它包含了每个文本块的文本内容、页码和包围盒(bbox)等信息。
        保留布局信息对于后续的清洗和结构化（例如，识别页眉页脚、合并段落）非常重要。

        Args:
            doc: 一个已经通过 fitz.open() 打开的 fitz.Document 对象。
                 注意：此函数不关闭文档，由调用方负责关闭。

        Returns:
            一个字典列表，每个字典代表一个文本块。
            每个字典至少包含键："text", "page", "bbox", "block_type", "source_doc_id", "source_page_height"。
        """
        text_blocks = []
        # 使用文件名作为文档的基本 ID。更健壮的方法可以使用文件的哈希值或 UUID。
        doc_id = os.path.basename(doc.name)

        logger.debug(f"开始提取文档 {doc_id} 的文本块...")
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            # page.get_text("dict") 返回一个包含 blocks, images, drawings 的字典
            # blocks 列表中的每个元素又包含 lines, spans 等更细粒度的信息
            # block['type'] == 0 表示文本块
            # block['type'] == 1 表示图片块
            page_blocks = page.get_text("dict")["blocks"]
            page_height = page.rect.height # 获取当前页面的实际高度

            for block in page_blocks:
                # 只处理文本块 (type 0)
                if block['type'] == 0:
                    block_text = ""
                    # 拼接块中所有行和跨度的文本
                    for line in block.get("lines", []): # 确保 line 键存在
                        line_text = ""
                        for span in line.get("spans", []): # 确保 span 键存在
                            # 保留原始文本，包括潜在的连字符
                            line_text += span.get("text", "")
                        # 在同一块内的行之间添加一个空格，除非该行以连字符结尾
                        # 这有助于后续处理断字连写的情况
                        # 也要处理空行（剔除空白后为空）
                        if line_text.strip():
                             # 如果上一行以连字符结束，当前行直接连接；否则加空格连接
                             # 这是一个简单的断字处理启发式
                             if block_text and block_text.endswith('-'):
                                 block_text = block_text[:-1].rstrip() + line_text.lstrip()
                             else:
                                block_text += ((" " if block_text else "") + line_text) # 如果不是第一行，加空格

                    # 清理块文本开头/结尾可能存在的空白
                    block_text = block_text.strip()

                    # 跳过清理后为空的文本块
                    if not block_text:
                         continue

                    text_blocks.append({
                        "text": block_text,
                        "page": page_num + 1, # 使用 1-based 页码
                        "bbox": block.get("bbox", (0,0,0,0)), # 获取包围盒信息，提供默认值
                        "block_type": "text",
                        "source_doc_id": doc_id,
                        "source_page_height": page_height # 存储页面高度，用于后续相对位置计算
                    })
                # TODO: 如果需要处理图片标题、表格等，可以在这里添加 block['type'] == 1 等的处理逻辑

        logger.debug(f"文档 {doc_id} 提取到 {len(text_blocks)} 个文本块。")
        return text_blocks

    def _clean_and_structure_blocks(self, text_blocks: List[Dict]) -> Dict[str, Any]:
        """
        对提取的文本块进行清洗、页眉/页脚去除和初步结构化（合并为段落）。

        这是一个启发式过程，旨在将相邻且在垂直方向上距离较近的文本块合并成一个逻辑段落。
        它还尝试根据垂直位置去除可能的页眉和页脚。

        Args:
            text_blocks: 带有布局信息的原始文本块列表。

        Returns:
            一个包含文档初步结构化文本数据的字典。
            结构示例: {"doc_id": ..., "sections": [{"title": "正文", "paragraphs": [...]}]}
            每个段落字典包含 "text", "page", "source_doc_id", "block_bboxes", "source_blocks" 等。
        """
        cleaned_units = []
        current_paragraph = None # 用 None 表示文档的开始，还没有当前段落

        # 按页码和垂直位置对文本块进行排序，以确保按阅读顺序处理
        # 确保每个 block 都有 page 和 bbox 键
        valid_blocks = [b for b in text_blocks if 'page' in b and 'bbox' in b and 'source_page_height' in b]
        valid_blocks.sort(key=lambda x: (x['page'], x['bbox'][1]))

        if not valid_blocks:
            logger.warning("没有有效的文本块可供清洗和结构化。")
            return {"doc_id": text_blocks[0].get("source_doc_id", "unknown") if text_blocks else "unknown", "sections": []}


        for block in valid_blocks:
            text = block.get("text", "").strip()
            page = block.get("page")
            doc_id = block.get("source_doc_id")
            bbox = block.get("bbox") # (x0, y0, x1, y1)
            page_height = block.get("source_page_height") # 使用存储的页面高度

            # 再次检查基本数据完整性，尽管 sorted valid_blocks 应该已过滤一部分
            if not text or not bbox or page_height is None:
                continue # 跳过数据不完整的块

            # 基于页面高度比例的简单页眉/页脚去除启发式
            # 这个方法对不同 PDF 格式的鲁棒性有限
            header_threshold = page_height * self.header_footer_threshold_y_ratio
            footer_threshold = page_height * (1 - self.header_footer_threshold_y_ratio)
            # bbox[1] 是块的顶部 Y 坐标，bbox[3] 是块的底部 Y 坐标
            if bbox[1] < header_threshold or bbox[3] > footer_threshold:
                # logger.debug(f"跳过潜在的页眉/页脚 (页 {page}, bbox={bbox}): {text[:50]}...")
                continue # 跳过位于页眉或页脚区域的块

            # 判断当前块是否应该开始一个新的段落
            start_new_paragraph = False
            if current_paragraph is None:
                # 如果是处理的第一个有效块，则开始第一个段落
                start_new_paragraph = True
            else:
                # 如果当前块在新的页码，则开始一个新段落
                # 或者，如果当前块与当前段落最后一个块之间的垂直距离较大，也开始新段落
                last_block_bbox = current_paragraph["block_bboxes"][-1]
                vertical_distance = bbox[1] - last_block_bbox[3] # 当前块顶部Y - 上一个块底部Y

                # 基于页面高度比例计算垂直距离阈值
                vertical_gap_threshold = page_height * self.line_spacing_threshold_ratio

                # 启发式判断：新页面 或者 垂直距离超过阈值
                if page != current_paragraph["page"] or vertical_distance > vertical_gap_threshold:
                    start_new_paragraph = True
                # TODO: 可以添加更多判断，例如：
                # - 当前块的字体大小或样式与前一块不同
                # - 当前块有明显的缩进
                # - 当前块以标题关键词开头等

            if start_new_paragraph:
                # 如果存在当前段落且不为空，则将其添加到已清洗单元列表
                if current_paragraph and current_paragraph["text"].strip():
                    cleaned_units.append(current_paragraph)
                # 初始化一个新的段落字典
                current_paragraph = {
                    "text": text, # 新段落从当前块的文本开始
                    "page": page,
                    "source_doc_id": doc_id,
                    "block_bboxes": [bbox], # 存储组成此段落的块的包围盒
                    "source_blocks": [block] # 存储组成此段落的原始块数据（如果需要更详细信息）
                }
            else:
                # 将当前块的文本追加到当前段落
                # 再次处理可能的断字连写：如果当前段落的文本以连字符结尾，且当前块文本以小写字母开头
                last_char_prev_text = current_paragraph["text"][-1] if current_paragraph["text"] else ''
                first_char_curr_text = text[0] if text else ''

                # 简单的断字处理：移除连字符和其后的空格，然后连接当前文本
                # 注意：之前在 extract_blocks_with_layout 中处理了一部分，这里是对段落文本的最终拼接处理
                if last_char_prev_text == '-' and isinstance(first_char_curr_text, str) and len(first_char_curr_text) == 1 and first_char_curr_text.islower():
                    # 移除当前段落末尾的连字符（以及拼接时可能加上的空格），然后拼接当前块的文本（移除开头空格）
                    current_paragraph["text"] = current_paragraph["text"][:-1].rstrip() + text.lstrip()
                else:
                    # 否则，简单地在段落文本和当前块文本之间加一个空格进行拼接
                    current_paragraph["text"] += (" " + text) # 如果 current_paragraph["text"] 为空，则只添加 text

                # 记录当前块的包围盒和原始块数据
                if bbox:
                    current_paragraph["block_bboxes"].append(bbox)
                current_paragraph["source_blocks"].append(block)


        # 循环结束后，将最后一个构建中的段落添加到列表
        if current_paragraph and current_paragraph["text"].strip():
            cleaned_units.append(current_paragraph)

        # 对最终的段落文本进行后处理清洗：
        for paragraph in cleaned_units:
            # 使用正则表达式将多个连续的空白字符（空格、换行符等）替换为单个空格
            paragraph["text"] = re.sub(r'\s+', ' ', paragraph["text"]).strip()
            # TODO: 这里可以添加更多的文本清洗规则，例如去除特殊符号、标准化格式等。

        # 构建简化的输出结构。
        # 精确识别文档的标题、摘要、各章节边界等需要更复杂的布局分析（字体、字号、位置、关键词）和机器学习模型。
        # 目前，我们将所有提取并结构化后的文本单元都视为正文段落，放在一个名为“正文”的默认章节中。
        # doc_id 从文本块中获取，应该代表原始文档。
        doc_id = valid_blocks[0].get("source_doc_id", "unknown") if valid_blocks else "unknown"

        structured_data = {
            "doc_id": doc_id, # 文档ID
            "filename": None, # 文件名需要在 process_single_pdf 方法中添加，因为这里没有 doc 对象了
            "title": "", # 占位符：需要更复杂的分析来识别
            "abstract": "", # 占位符：需要更复杂的分析来识别
            "sections": [{"title": "正文", "paragraphs": cleaned_units}] # 简化结构，所有段落归入“正文”章节
        }

        # TODO: 如果需要更细粒度的实体定位，可以在这里或后续步骤将段落文本拆分成句子，
        # 并记录每个句子在原始文本块/段落中的偏移量或包围盒信息。目前简化为段落作为源句。

        logger.debug(f"清洗并结构化完成。得到 {len(cleaned_units)} 个段落。")
        return structured_data

    def process_single_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        处理一个PDF文件，执行完整的文本解析和初步结构化流程。

        这是推荐在主流程中调用的方法，它结合了文本块提取和清洗结构化步骤。

        Args:
            file_path: 待处理的PDF文件的完整路径。

        Returns:
            一个包含结构化文本数据的字典。如果处理失败则返回空字典 {}。
        """
        doc = None
        try:
            doc = self.load_pdf(file_path)
            # 1. 提取带有布局信息的原始文本块
            raw_blocks = self.extract_blocks_with_layout(doc)

            if not raw_blocks:
                 logger.warning(f"文件 {file_path} 未提取到任何原始文本块。")
                 return {}

            # 2. 清洗、去除页眉/页脚并初步结构化（合并段落）
            structured_data = self._clean_and_structure_blocks(raw_blocks)

            # 在这里添加文件名，因为 _clean_and_structure_blocks 内部无法获取原始 doc 对象
            structured_data["filename"] = os.path.basename(file_path)
            # structured_data["doc_id"] 已经在 _clean_and_structure_blocks 中从 blocks 派生并设置

            # TODO: 如果需要，可以在这里添加更高级的文档结构识别步骤。

            logger.info(f"成功处理并初步结构化 PDF: {file_path}")
            return structured_data
        except FileNotFoundError:
             # 文件未找到错误已经在 load_pdf 中处理并记录，这里只返回空结果
             return {}
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时发生意外错误: {e}")
            return {}
        finally:
            # 确保在任何情况下都关闭 PDF 文档，释放资源
            if doc:
                doc.close()
                # logger.debug(f"已关闭文档 {file_path}")

# Example usage (for standalone testing - this code will run only if the script is executed directly)
if __name__ == "__main__":
    print("正在运行 PDFParser 模块的独立测试...")
    # 为了独立测试，需要模拟配置加载
    # 实际在主流程中，config 是由 main_pipeline.py 调用 utils.load_config 加载并传递进来的
    dummy_config_for_test = {
        "paths": {
            "raw_pdfs_dir": "data/raw_pdfs",
            # "processed_text_dir": "data/processed_text" # 不再强制保存中间 processed_text
        },
        # 可以添加 PDF 解析相关的阈值配置到 settings.yaml 并在此处模拟加载
        # "pdf_parser": {
        #     "header_footer_threshold_y_ratio": 0.05,
        #     "line_spacing_threshold_ratio": 0.01
        # }
    }

    # 确保测试所需的目录存在
    os.makedirs(dummy_config_for_test["paths"]["raw_pdfs_dir"], exist_ok=True)
    # os.makedirs(dummy_config_for_test["paths"]["processed_text_dir"], exist_ok=True) # 不再需要 processed_text 目录

    parser = PDFParser(dummy_config_for_test)

    # 指定一个测试 PDF 文件路径
    # 你需要手动在 data/raw_pdfs/ 目录中放入一个 PDF 文件，或者让下面的代码创建一个简单的虚拟 PDF
    test_pdf_filename = "test_document_for_parser.pdf" # 替换为你实际的 PDF 文件名
    test_pdf_path = os.path.join(dummy_config_for_test["paths"]["raw_pdfs_dir"], test_pdf_filename)

    # --- 创建一个简单的虚拟 PDF 文件用于测试，如果文件不存在 ---
    if not os.path.exists(test_pdf_path):
        print(f"测试文件未找到于 {test_pdf_path}。正在创建一个简单的虚拟 PDF 文件用于测试。")
        try:
            doc = fitz.open() # 创建一个新的空 PDF 文档
            page1 = doc.new_page() # 添加第一页
            # 插入一些文本块，模拟段落和页眉页脚
            page1.insert_text((50, 50), "这是一个测试文档的开头。\n这里是第一段。", fontsize=12)
            page1.insert_text((50, 80), "第一段的第二句话。", fontsize=12)
            page1.insert_text((50, 110), "这是第二段的开始，\n模拟一个新段落。", fontsize=12)
            page1.insert_text((50, 750), "页脚信息。", fontsize=8) # 模拟页脚

            page2 = doc.new_page() # 添加第二页
            page2.insert_text((50, 50), "页眉信息。", fontsize=8) # 模拟页眉
            page2.insert_text((50, 100), "第二页的正文第一段。\n这段落可能包含关于 AA6061 铝合金的信息。", fontsize=12)
            page2.insert_text((50, 130), "它的主要成分是 Mg 和 Si。", fontsize=12)
            page2.insert_text((50, 160), "经过固溶处理后，\n性能显著提升，例如抗拉强度。", fontsize=12)

            doc.save(test_pdf_path) # 保存文档
            doc.close() # 关闭文档
            print("虚拟 PDF 文件创建成功。")
        except ImportError:
             print("错误：PyMuPDF (fitz) 库未安装。无法创建虚拟 PDF 文件。请安装：pip install PyMuPDF")
             test_pdf_path = None # 如果无法创建 PDF，跳过后续测试
        except Exception as e:
             print(f"创建虚拟 PDF 文件时发生错误: {e}")
             test_pdf_path = None # 如果创建失败，跳过后续测试
    # --- 虚拟 PDF 创建结束 ---


    if test_pdf_path and os.path.exists(test_pdf_path):
        print(f"\n开始处理测试文件: {test_pdf_path}")

        # 调用推荐的处理方法
        structured_text = parser.process_single_pdf(test_pdf_path)

        if structured_text:
            print("\n--- 处理结果 (结构化文本示例) ---")
            import json
            # 打印部分结果，方便查看结构和内容，避免输出过长
            print(json.dumps(structured_text, indent=2, ensure_ascii=False)[:2000] + "...") # 打印前2000个字符

            # 示例：打印所有提取的段落文本及其页码
            print("\n--- 提取的段落文本 ---")
            # 结构化数据包含 sections 列表，每个 section 包含 paragraphs 列表
            for section in structured_text.get("sections", []):
                 # 确保 section 是字典类型
                 if isinstance(section, dict):
                     print(f"章节: {section.get('title', '未知章节')}")
                     for i, paragraph in enumerate(section.get("paragraphs", [])):
                          # 确保 paragraph 是字典类型
                          if isinstance(paragraph, dict):
                               # 打印每段文本的前100个字符
                               print(f"  段落 {i+1} (页码: {paragraph.get('page', '未知')}, DocID: {paragraph.get('source_doc_id', '未知')}): {paragraph.get('text', '')[:100]}...")

            # 打印总段落数
            total_paragraphs = sum(len(section.get('paragraphs', [])) for section in structured_text.get('sections', []) if isinstance(section, dict))
            print(f"\n总共提取 {total_paragraphs} 个段落。")

        else:
            print("\n处理 PDF 失败或未提取到有效内容。")

    else:
         print("\n跳过 PDF 处理测试，因为测试文件不存在或创建失败。")

    # 注意：在独立测试中，如果创建了临时文件，你可能需要手动删除它们
    # 例如： os.remove(test_pdf_path) # 如果你确定这是临时创建的文件

    print("\nPDFParser 模块独立测试结束。")

