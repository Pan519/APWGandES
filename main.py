#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
真正全自动AI系统
用户只需输入需求，系统全自动完成所有工作
"""

import os
import json
import time
import re
import base64
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

class FullyAutoAISystem:
    """真正全自动AI系统"""
    
    def __init__(self):
        """初始化系统"""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.api_available = True
        else:
            self.client = None
            self.api_available = False
            print("警告: 未配置API密钥，将使用模拟模式")
        
        # 从环境变量获取模型配置，如果不存在则使用默认值
        self.models = {
            # 文本生成模型
            "qwen-turbo": {"description": os.environ.get("QWEN_TURBO_DESC", "超快推理速度的推理模型")},
            "qwen-turbo-2024-06-24": {"description": os.environ.get("QWEN_TURBO_2024_06_24_DESC", "Turbo模型快照版本")},
            "qwen-plus": {"description": os.environ.get("QWEN_PLUS_DESC", "平衡处理中等复杂任务")},
            "qwen-plus-2024-08-06": {"description": os.environ.get("QWEN_PLUS_2024_08_06_DESC", "Plus模型快照版本")},
            "qwen-max": {"description": os.environ.get("QWEN_MAX_DESC", "强大处理复杂任务")},
            "qwen-max-2024-04-28": {"description": os.environ.get("QWEN_MAX_2024_04_28_DESC", "Max模型快照版本")},
            "qwen-max-2024-09-19": {"description": os.environ.get("QWEN_MAX_2024_09_19_DESC", "Max模型快照版本")},
            "qwen3": {"description": os.environ.get("QWEN_3_DESC", "最新的Qwen3大语言模型")},
            "qwen3-235b-a22b": {"description": os.environ.get("QWEN_3_235B_A22B_DESC", "Qwen3旗舰模型")},
            "qwen2.5": {"description": os.environ.get("QWEN_2_5_DESC", "通义千问2.5版本")},
            "qwen2.5-max": {"description": os.environ.get("QWEN_2_5_MAX_DESC", "Qwen2.5 Max版本")},
            "qwen2.5-turbo": {"description": os.environ.get("QWEN_2_5_TURBO_DESC", "Qwen2.5 Turbo版本")},
            "qwen2": {"description": os.environ.get("QWEN_2_DESC", "通义千问2版本")},
            "qwen1.5": {"description": os.environ.get("QWEN_1_5_DESC", "通义千问1.5版本")},
            "qwen-long": {"description": os.environ.get("QWEN_LONG_DESC", "支持超长上下文的模型")},
            "qwen-omni-turbo-2025-01-19": {"description": os.environ.get("QWEN_OMNI_TURBO_2025_01_19_DESC", "Omni Turbo模型")},
            "qwen-coder-turbo-0919": {"description": os.environ.get("QWEN_CODER_TURBO_0919_DESC", "面向编程的Turbo模型")},
            
            # 多模态模型
            "qwen-vl-max": {"description": os.environ.get("QWEN_VL_MAX_DESC", "视觉理解旗舰模型")},
            "qwen-vl-plus": {"description": os.environ.get("QWEN_VL_PLUS_DESC", "视觉理解增强模型")},
            "qwen-vl-plus-2025-01-25": {"description": os.environ.get("QWEN_VL_PLUS_2025_01_25_DESC", "VL Plus模型快照版本")},
            
            # 音频模型
            "qwen-audio-turbo-1204": {"description": os.environ.get("QWEN_AUDIO_TURBO_1204_DESC", "音频处理Turbo模型")},
            "qwen-audio-turbo-0807": {"description": os.environ.get("QWEN_AUDIO_TURBO_0807_DESC", "音频处理Turbo模型")},
            
            # 数学和编程推理模型
            "qwen-omni": {"description": os.environ.get("QWEN_OMNI_DESC", "多模态推理模型")},
            "qwq-32b-preview": {"description": os.environ.get("QWQ_32B_PREVIEW_DESC", "专注于推理的模型")},
            
            # 开源模型
            "qwen2-72b-instruct": {"description": os.environ.get("QWEN2_72B_INSTRUCT_DESC", "Qwen2 72B指令模型")},
            "qwen2-57b-a14b-instruct": {"description": os.environ.get("QWEN2_57B_A14B_INSTRUCT_DESC", "Qwen2 57B+A14B指令模型")},
            "qwen2-7b-instruct": {"description": os.environ.get("QWEN2_7B_INSTRUCT_DESC", "Qwen2 7B指令模型")},
            "qwen1.5-110b-chat": {"description": os.environ.get("QWEN1_5_110B_CHAT_DESC", "Qwen1.5 110B对话模型")},
            "qwen1.5-72b-chat": {"description": os.environ.get("QWEN1_5_72B_CHAT_DESC", "Qwen1.5 72B对话模型")},
            "qwen1.5-32b-chat": {"description": os.environ.get("QWEN1_5_32B_CHAT_DESC", "Qwen1.5 32B对话模型")},
            "qwen1.5-14b-chat": {"description": os.environ.get("QWEN1_5_14B_CHAT_DESC", "Qwen1.5 14B对话模型")},
            "qwen1.5-7b-chat": {"description": os.environ.get("QWEN1_5_7B_CHAT_DESC", "Qwen1.5 7B对话模型")}
        }
        
        self.default_model = os.environ.get("DEFAULT_MODEL", "qwen-plus")
        # 会话级别的token使用统计（来自平台的真实数据）
        self.session_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def read_file(self, file_path: str) -> Optional[str]:
        """
        读取本地文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容，如果读取失败则返回None
        """
        # 规范化路径
        file_path = os.path.normpath(file_path.strip())
        
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在")
            return None
        
        # 检查是否为图像文件
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in image_extensions:
            try:
                # 对于图像文件，转换为base64编码
                with open(file_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    return f"[图像文件 (base64编码): {encoded_string}]"
            except Exception as e:
                print(f"错误: 无法读取图像文件 '{file_path}': {e}")
                return None
        
        # 检查文件大小，避免读取过大的文件
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB限制
                print(f"警告: 文件 '{file_path}' 过大 ({file_size} 字节)，可能影响处理效果")
        except OSError:
            pass  # 获取文件大小失败，继续处理
        
        try:
            # 首先尝试UTF-8编码
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # 如果UTF-8失败，尝试GBK编码
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    # 如果GBK也失败，尝试Latin-1编码（总是能读取成功）
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                        # 尝试检测是否是二进制文件
                        if '\x00' in content[:1024]:  # 检查前1024个字符是否有空字节
                            print(f"警告: 文件 '{file_path}' 可能是二进制文件")
                            return content
                        return content
                except Exception as e:
                    print(f"错误: 无法读取文件 '{file_path}': {e}")
                    return None
        except Exception as e:
            print(f"错误: 读取文件 '{file_path}' 失败: {e}")
            return None
    
    def parse_request(self, request: str) -> tuple:
        """
        解析请求中的文件引用
        
        Args:
            request: 用户请求字符串，可能包含[file:filepath]形式的文件引用
            
        Returns:
            tuple: (处理后的请求, 文件内容字典)
        """
        # 查找 [file:filepath] 模式
        file_pattern = r'\[file:(.*?)\]'
        file_paths = re.findall(file_pattern, request)
        
        file_contents = {}
        processed_request = request
        
        # 读取所有引用的文件
        for file_path in file_paths:
            content = self.read_file(file_path.strip())
            if content is not None:  # 明确检查是否为None
                file_contents[file_path] = content
                # 在请求中用文件内容替换引用，并添加标识以便AI理解
                processed_request = processed_request.replace(
                    f'[file:{file_path}]', 
                    f"[文件内容开始:{file_path}]\n{content}\n[文件内容结束:{file_path}]"
                )
            else:
                print(f"警告: 无法读取文件 '{file_path}'，将在请求中保留原始引用")
        
        return processed_request, file_contents
    
    def select_model(self, request: str, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """AI自动选择最适合的模型"""
        if not self.api_available:
            return {
                "model": self.default_model,
                "reason": "API不可用，使用默认模型",
                "task_type": "general",
                "complexity": "medium"
            }
        
        # 检查是否有图像文件，如果有则优先选择视觉理解模型
        has_image = any("[图像文件" in content for content in file_contents.values())
        if has_image:
            # 检查可用的视觉理解模型
            vl_models = ["qwen-vl-max", "qwen-vl-plus", "qwen-vl-plus-2025-01-25"]
            for model in vl_models:
                if model in self.models:
                    print(f"检测到图像文件，选择视觉理解模型: {model}")
                    return {
                        "model": model,
                        "reason": "检测到图像文件，选择视觉理解模型",
                        "task_type": "multimodal",
                        "complexity": "medium"
                    }
        
        try:
            # 构造模型选择提示
            file_info = f"文件数量: {len(file_contents)}" if file_contents else "无文件"
            
            # 构建模型列表字符串，只包含实际可用的模型
            available_models = {model: info for model, info in self.models.items() if info.get("description")}
            model_list = "\n".join([f"{i+1}. {model}: {info['description']}" for i, (model, info) in enumerate(available_models.items())])
            
            prompt = f"""
作为一个AI模型选择专家，请根据以下任务选择最适合的模型：

任务描述: {request}
{file_info}

可选模型:
{model_list}

请以JSON格式回复:
{{
    "model": "模型名称",
    "task_type": "任务类型（如分析、创作、代码等）",
    "complexity": "复杂度（simple/medium/complex）",
    "reason": "选择理由"
}}
"""
            
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "你是一个AI模型选择专家，请以JSON格式回复"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            # 获取平台返回的真实token使用数据
            # 根据阿里云百炼平台API文档，usage字段包含真实的token计量信息
            if hasattr(response, 'usage'):
                usage = response.usage
                self.session_token_usage["prompt_tokens"] += usage.prompt_tokens
                self.session_token_usage["completion_tokens"] += usage.completion_tokens
                self.session_token_usage["total_tokens"] += usage.total_tokens
            
            # 安全地解析JSON响应
            response_content = response.choices[0].message.content
            if not response_content or not response_content.strip():
                raise ValueError("模型返回空响应")
            
            # 清理响应内容，确保它是有效的JSON
            response_content = response_content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:]  # 移除 ```json
            if response_content.endswith("```"):
                response_content = response_content[:-3]  # 移除 ```
            
            try:
                selection_result = json.loads(response_content)
            except json.JSONDecodeError as je:
                print(f"JSON解析失败，响应内容: {response_content}")
                raise ValueError(f"无法解析模型响应为JSON格式: {str(je)}")
            
            # 验证模型是否有效
            if "model" in selection_result and selection_result["model"] in self.models:
                print(f"AI选择了模型: {selection_result['model']}")
                return selection_result
            else:
                print(f"AI选择了无效模型 '{selection_result.get('model', 'unknown')}'，使用默认模型")
                selection_result["model"] = self.default_model
                return selection_result
                
        except Exception as e:
            print(f"模型选择失败，使用默认模型: {e}")
            return {
                "model": self.default_model,
                "reason": f"分析失败，使用默认模型: {str(e)}",
                "task_type": "general",
                "complexity": "medium"
            }
    
    def execute_task(self, request: str, model: str) -> Dict[str, Any]:
        """执行AI任务"""
        if not self.api_available:
            return {
                "content": f"模拟结果:\n\n用户请求: {request}\n\n这是模拟内容，在实际使用中会调用真实的AI模型生成结果。",
                "model": model,
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "execution_time": 0
            }
        
        try:
            start_time = time.time()
            # 获取模型ID用于API调用
            model_id = self.models.get(model, {}).get("id", model)
            
            response = self.client.chat.completions.create(
                model=model_id,  # 使用模型ID而不是模型名称
                messages=[
                    {"role": "system", "content": "你是一个万能AI助手，请高质量完成用户任务。请特别注意用户请求中包含的文件内容，确保充分理解和利用这些内容来完成任务。"},
                    {"role": "user", "content": request}
                ],
                max_tokens=int(os.environ.get("MAX_TOKENS", 2000)),
                temperature=float(os.environ.get("TEMPERATURE", 0.7))
            )
            end_time = time.time()
            
            # 获取平台返回的真实token使用数据（来自阿里云百炼平台的真实计量数据）
            # 根据平台API文档：usage字段包含input_tokens和output_tokens
            tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if hasattr(response, 'usage'):
                usage = response.usage
                tokens["prompt_tokens"] = usage.prompt_tokens      # 平台真实输入token数
                tokens["completion_tokens"] = usage.completion_tokens  # 平台真实输出token数
                tokens["total_tokens"] = usage.total_tokens        # 平台真实总token数
                
                # 累加到会话统计中
                self.session_token_usage["prompt_tokens"] += usage.prompt_tokens
                self.session_token_usage["completion_tokens"] += usage.completion_tokens
                self.session_token_usage["total_tokens"] += usage.total_tokens
            
            return {
                "content": response.choices[0].message.content,
                "model": model,
                "tokens": tokens,  # 这是平台返回的真实token数据
                "execution_time": end_time - start_time
            }
            
        except Exception as e:
            error_msg = f"任务执行失败: {e}"
            print(error_msg)
            return {
                "content": error_msg,
                "model": model,
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "execution_time": 0
            }
    
    def generate_prompt(self, request: str, model: str) -> Dict[str, Any]:
        """生成用于执行任务的优化提示词"""
        if not self.api_available:
            return {
                "content": f"请根据以下请求生成内容：{request}",
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        
        try:
            # 获取模型ID用于API调用
            model_id = self.models.get(model, {}).get("id", model)
            
            prompt = f"""
请为以下用户请求生成一个优化的提示词，用于指导AI模型生成相关内容：

用户请求：{request}

请生成一个结构清晰、目标明确的提示词，能够指导AI模型准确地完成用户请求。
特别注意要充分利用请求中提供的文件内容。
"""

            response = self.client.chat.completions.create(
                model=model_id,  # 使用模型ID而不是模型名称
                messages=[
                    {"role": "system", "content": "你是一个专业的提示词工程师"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            # 获取平台返回的真实token使用数据
            tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if hasattr(response, 'usage'):
                usage = response.usage
                tokens["prompt_tokens"] = usage.prompt_tokens
                tokens["completion_tokens"] = usage.completion_tokens
                tokens["total_tokens"] = usage.total_tokens
                
                # 累加到会话统计中
                self.session_token_usage["prompt_tokens"] += usage.prompt_tokens
                self.session_token_usage["completion_tokens"] += usage.completion_tokens
                self.session_token_usage["total_tokens"] += usage.total_tokens
            
            # 安全地处理响应内容
            response_content = response.choices[0].message.content
            if not response_content:
                response_content = f"请根据以下请求生成内容：{request}"
            
            return {
                "content": response_content,
                "tokens": tokens  # 这是平台返回的真实token数据
            }
            
        except Exception as e:
            print(f"生成提示词失败: {e}")
            return {
                "content": f"请根据以下请求生成内容：{request}",
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    
    def save_result(self, content: str, original_request: str, file_type: str = "result") -> str:
        """保存结果到文件，支持多种文件类型"""
        # 从请求中提取关键词用于文件名
        keywords = re.sub(r'[^\w\u4e00-\u9fff]', '', original_request)[:10] or "结果"
        timestamp = int(time.time())
        
        # 根据内容类型确定文件扩展名
        if file_type == "result":
            if self._is_code_content(content):
                extension = self._detect_code_language(content)
                # 提取纯代码内容
                pure_code = self._extract_pure_code(content)
                filename = f"AI结果_代码_{keywords}_{timestamp}.{extension}"
                content_to_save = pure_code
            elif self._is_markdown_content(content):
                filename = f"AI结果_文档_{keywords}_{timestamp}.md"
                content_to_save = content
            elif self._is_image_description(content):
                filename = f"AI结果_图像_{keywords}_{timestamp}.md"
                content_to_save = content
            else:
                filename = f"AI结果_文本_{keywords}_{timestamp}.txt"
                content_to_save = content
        elif file_type == "prompt":
            filename = f"AI提示词_{keywords}_{timestamp}.md"
            content_to_save = content
        elif file_type == "report":
            filename = f"AI执行报告_{keywords}_{timestamp}.md"
            content_to_save = content
        else:
            filename = f"AI结果_{file_type}_{keywords}_{timestamp}.txt"
            content_to_save = content
        
        # 确保文件名合法
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content_to_save)
            print(f"文件已保存到: {filename}")
            return filename
        except Exception as e:
            error_msg = f"保存文件失败: {e}"
            print(error_msg)
            return error_msg
    
    def _is_code_content(self, content: str) -> bool:
        """判断内容是否为代码"""
        code_indicators = ['def ', 'class ', 'function ', 'import ', 'public ', 'private ', '#include', '<?php']
        return any(indicator in content for indicator in code_indicators)
    
    def _is_markdown_content(self, content: str) -> bool:
        """判断内容是否为Markdown"""
        md_indicators = ['## ', '### ', '- ', '```', '**', '*']
        return any(indicator in content for indicator in md_indicators)
    
    def _is_image_description(self, content: str) -> bool:
        """判断内容是否为图像描述"""
        image_indicators = ['图像', '图片', '照片', '视觉', '图示', '图表', 'image', 'picture', 'photo']
        return any(indicator in content.lower() for indicator in image_indicators)
    
    def _detect_code_language(self, content: str) -> str:
        """检测代码语言"""
        if 'def ' in content and ':' in content:
            return "py"
        elif 'public ' in content or 'private ' in content:
            return "java"
        elif '#include' in content:
            return "cpp"
        elif 'function ' in content or 'console.log' in content:
            return "js"
        elif '<?php' in content:
            return "php"
        else:
            return "txt"
    
    def _extract_pure_code(self, content: str) -> str:
        """从AI响应中提取纯代码，去除额外的说明文字"""
        # 查找代码块
        code_block_pattern = r"```(?:\w+)?\s*\n(.*?)\n\s*```"
        code_blocks = re.findall(code_block_pattern, content, re.DOTALL)
        
        if code_blocks:
            # 如果找到代码块，返回第一个代码块的内容
            return code_blocks[0].strip()
        else:
            # 如果没有找到代码块，直接返回原内容
            return content
    
    def evaluate_task_result(self, result: str, original_request: str) -> Dict[str, Any]:
        """评估任务执行结果"""
        if not self.api_available:
            return {
                "quality_score": 80,
                "completeness": "中等",
                "feedback": "模拟结果，质量评估仅供参考"
            }
        
        try:
            prompt = f"""
请评估以下AI任务执行结果的质量：

原始请求: {original_request}

执行结果: {result[:2000]}  # 限制长度以避免超出token限制

请以JSON格式回复:
{{
    "quality_score": 0-100的评分,
    "completeness": "完成度评估（高/中/低）",
    "strengths": "优势点",
    "weaknesses": "不足点",
    "feedback": "改进建议"
}}
"""
            
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "你是一个AI输出质量评估专家，请以JSON格式回复"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # 获取平台返回的真实token使用数据
            if hasattr(response, 'usage'):
                usage = response.usage
                self.session_token_usage["prompt_tokens"] += usage.prompt_tokens
                self.session_token_usage["completion_tokens"] += usage.completion_tokens
                self.session_token_usage["total_tokens"] += usage.total_tokens
            
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
            
        except Exception as e:
            print(f"结果评估失败: {e}")
            return {
                "quality_score": 75,
                "completeness": "中等",
                "strengths": "完成了基本任务要求",
                "weaknesses": "评估过程出现异常",
                "feedback": f"评估出现异常: {str(e)}"
            }
    
    def generate_report(self, original_request: str, model_selection: Dict[str, Any], 
                       execution_result: Dict[str, Any], prompt_result: Dict[str, Any],
                       evaluation_result: Dict[str, Any], result_file: str, prompt_file: str) -> str:
        """生成全面的执行报告"""
        timestamp = int(time.time())
        
        # 计算总执行时间
        total_execution_time = execution_result.get("execution_time", 0)
        
        # 获取各阶段的token使用情况（这些都是平台返回的真实数据）
        execution_tokens = execution_result.get("tokens", {})
        prompt_tokens = prompt_result.get("tokens", {})
        
        report_content = f"""# AI全自动执行综合报告

## 任务基本信息
- **原始请求**: {original_request}
- **执行时间**: {time.ctime(timestamp)}
- **总执行耗时**: {total_execution_time:.2f} 秒

## 模型选择分析
- **选择的模型**: {model_selection.get("model", "未知")}
- **任务类型**: {model_selection.get("task_type", "未知")}
- **任务复杂度**: {model_selection.get("complexity", "未知")}
- **选择理由**: {model_selection.get("reason", "无")}

## 执行详情
- **结果文件**: {result_file}
- **提示词文件**: {prompt_file}
- **使用模型**: {execution_result.get("model", "未知")}

## Token使用统计（平台真实数据）
- **模型选择阶段**:
  - 输入Token: {self.session_token_usage["prompt_tokens"] - execution_tokens.get("prompt_tokens", 0) - prompt_tokens.get("prompt_tokens", 0)}
  - 输出Token: {self.session_token_usage["completion_tokens"] - execution_tokens.get("completion_tokens", 0) - prompt_tokens.get("completion_tokens", 0)}
  - 总计Token: {self.session_token_usage["total_tokens"] - execution_tokens.get("total_tokens", 0) - prompt_tokens.get("total_tokens", 0)}

- **提示词生成阶段**:
  - 输入Token: {prompt_tokens.get("prompt_tokens", 0)} (平台真实数据)
  - 输出Token: {prompt_tokens.get("completion_tokens", 0)} (平台真实数据)
  - 总计Token: {prompt_tokens.get("total_tokens", 0)} (平台真实数据)

- **任务执行阶段**:
  - 输入Token: {execution_tokens.get("prompt_tokens", 0)} (平台真实数据)
  - 输出Token: {execution_tokens.get("completion_tokens", 0)} (平台真实数据)
  - 总计Token: {execution_tokens.get("total_tokens", 0)} (平台真实数据)

- **整体Token使用**:
  - 总输入Token: {self.session_token_usage["prompt_tokens"]} (平台真实累计数据)
  - 总输出Token: {self.session_token_usage["completion_tokens"]} (平台真实累计数据)
  - 总消耗Token: {self.session_token_usage["total_tokens"]} (平台真实累计数据)

## 结果质量评估
- **质量评分**: {evaluation_result.get("quality_score", "无")}
- **完成度**: {evaluation_result.get("completeness", "无")}
- **优势**: {evaluation_result.get("strengths", "无")}
- **不足**: {evaluation_result.get("weaknesses", "无")}
- **改进建议**: {evaluation_result.get("feedback", "无")}

## 系统说明
此报告由AI全自动系统生成，所有Token使用数据均为平台返回的真实计量数据。
根据阿里云百炼平台API文档，总Token数量 = 输入Token数(input_tokens) + 输出Token数(output_tokens)，
费用与总Token数量成正比。

---
*AI全自动系统 v1.0*
"""
        return self.save_result(report_content, original_request, "report")
    
    def run(self):
        """运行主程序"""
        print("=" * 60)
        print("真正全自动AI系统")
        print("=" * 60)
        print("只需输入您的需求，AI将全自动完成所有工作")
        print("示例: 请分析[file:test_document.txt]中的内容并生成总结")
        print("输入 'quit' 退出系统")
        print("=" * 60)
        
        while True:
            try:
                # 获取用户需求
                request = input("\n请输入您的需求: ").strip()
                
                # 检查退出命令
                if request.lower() in ['quit', 'exit', '退出']:
                    print("感谢使用，再见！")
                    break
                
                if not request:
                    print("请输入有效需求")
                    continue
                
                # 重置token使用统计
                self.session_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                
                # 1. 自动解析请求中的文件引用
                print("正在解析请求...")
                processed_request, file_contents = self.parse_request(request)
                
                # 显示解析到的文件信息
                if file_contents:
                    print(f"已解析到 {len(file_contents)} 个文件:")
                    for file_path in file_contents.keys():
                        print(f"  - {file_path}")
                else:
                    print("未发现引用的文件")
                
                # 2. AI自动选择最适合的模型
                print("AI正在分析任务并选择模型...")
                model_selection = self.select_model(processed_request, file_contents)
                selected_model = model_selection["model"]
                
                # 3. 生成优化提示词
                print("正在生成优化提示词...")
                prompt_result = self.generate_prompt(processed_request, selected_model)
                prompt_file = self.save_result(prompt_result["content"], request, "prompt")
                
                # 4. AI自动执行任务
                print("AI正在执行任务...")
                execution_result = self.execute_task(prompt_result["content"], selected_model)
                
                # 5. 评估任务结果
                print("正在评估执行结果...")
                evaluation_result = self.evaluate_task_result(execution_result["content"], request)
                
                # 6. 自动保存结果
                print("正在保存结果...")
                result_file = self.save_result(execution_result["content"], request, "result")
                
                # 7. 生成全面的执行报告
                print("正在生成全面的执行报告...")
                report_file = self.generate_report(
                    request, model_selection, execution_result, 
                    prompt_result, evaluation_result, result_file, prompt_file
                )
                
                # 8. 显示结果
                print("\n" + "=" * 60)
                print("任务已完成！")
                print("=" * 60)
                print(f"使用模型: {selected_model}")
                print(f"结果文件: {result_file}")
                print(f"提示词文件: {prompt_file}")
                print(f"报告文件: {report_file}")
                print(f"总消耗Token: {self.session_token_usage['total_tokens']} (平台真实数据)")
                print("AI已全自动完成您的任务")
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n程序被中断，正在退出...")
                break
            except Exception as e:
                print(f"发生错误: {e}")

def main():
    """主函数"""
    system = FullyAutoAISystem()
    system.run()

if __name__ == "__main__":
    main()