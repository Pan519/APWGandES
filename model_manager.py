import os
import json
import requests
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class ModelManager:
    """
    模型管理器：管理AI模型选择和token使用情况
    """
    
    def __init__(self):
        """
        初始化模型管理器
        """
        # 初始化模型配置
        self.models = {
            # 文本生成模型
            "qwen-turbo": {
                "id": os.environ.get("QWEN_TURBO", "qwen-turbo"),
                "max_tokens": 1000000,  # 假设的最大token数
                "used_tokens": 0,
                "type": "text"
            },
            "qwen-turbo-2024-06-24": {
                "id": os.environ.get("QWEN_TURBO_2024_06_24", "qwen-turbo-2024-06-24"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen-plus": {
                "id": os.environ.get("QWEN_PLUS", "qwen-plus"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen-plus-2024-08-06": {
                "id": os.environ.get("QWEN_PLUS_2024_08_06", "qwen-plus-2024-08-06"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen-max": {
                "id": os.environ.get("QWEN_MAX", "qwen-max"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen-max-2024-04-28": {
                "id": os.environ.get("QWEN_MAX_2024_04_28", "qwen-max-2024-04-28"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen-max-2024-09-19": {
                "id": os.environ.get("QWEN_MAX_2024_09_19", "qwen-max-2024-09-19"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen3": {
                "id": os.environ.get("QWEN_3", "qwen3"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen3-235b-a22b": {
                "id": os.environ.get("QWEN_3_235B_A22B", "qwen3-235b-a22b"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen2.5": {
                "id": os.environ.get("QWEN_2_5", "qwen2.5"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen2.5-max": {
                "id": os.environ.get("QWEN_2_5_MAX", "qwen2.5-max"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen2.5-turbo": {
                "id": os.environ.get("QWEN_2_5_TURBO", "qwen2.5-turbo"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen2": {
                "id": os.environ.get("QWEN_2", "qwen2"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen1.5": {
                "id": os.environ.get("QWEN_1_5", "qwen1.5"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen-long": {
                "id": os.environ.get("QWEN_LONG", "qwen-long"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen-omni-turbo-2025-01-19": {
                "id": os.environ.get("QWEN_OMNI_TURBO_2025_01_19", "qwen-omni-turbo-2025-01-19"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen-coder-turbo-0919": {
                "id": os.environ.get("QWEN_CODER_TURBO_0919", "qwen-coder-turbo-0919"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            
            # 多模态模型
            "qwen-vl-max": {
                "id": os.environ.get("QWEN_VL_MAX", "qwen-vl-max"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "multimodal"
            },
            "qwen-vl-plus": {
                "id": os.environ.get("QWEN_VL_PLUS", "qwen-vl-plus"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "multimodal"
            },
            "qwen-vl-plus-2025-01-25": {
                "id": os.environ.get("QWEN_VL_PLUS_2025_01_25", "qwen-vl-plus-2025-01-25"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "multimodal"
            },
            
            # 音频模型
            "qwen-audio-turbo-1204": {
                "id": os.environ.get("QWEN_AUDIO_TURBO_1204", "qwen-audio-turbo-1204"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "audio"
            },
            "qwen-audio-turbo-0807": {
                "id": os.environ.get("QWEN_AUDIO_TURBO_0807", "qwen-audio-turbo-0807"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "audio"
            },
            
            # 数学和编程推理模型
            "qwen-omni": {
                "id": os.environ.get("QWEN_OMNI", "qwen-omni"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "reasoning"
            },
            "qwq-32b-preview": {
                "id": os.environ.get("QWQ_32B_PREVIEW", "qwq-32b-preview"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "reasoning"
            },
            
            # 开源模型
            "qwen2-72b-instruct": {
                "id": os.environ.get("QWEN2_72B_INSTRUCT", "qwen2-72b-instruct"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen2-57b-a14b-instruct": {
                "id": os.environ.get("QWEN2_57B_A14B_INSTRUCT", "qwen2-57b-a14b-instruct"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen2-7b-instruct": {
                "id": os.environ.get("QWEN2_7B_INSTRUCT", "qwen2-7b-instruct"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen1.5-110b-chat": {
                "id": os.environ.get("QWEN1_5_110B_CHAT", "qwen1.5-110b-chat"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen1.5-72b-chat": {
                "id": os.environ.get("QWEN1_5_72B_CHAT", "qwen1.5-72b-chat"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen1.5-32b-chat": {
                "id": os.environ.get("QWEN1_5_32B_CHAT", "qwen1.5-32b-chat"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen1.5-14b-chat": {
                "id": os.environ.get("QWEN1_5_14B_CHAT", "qwen1.5-14b-chat"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            },
            "qwen1.5-7b-chat": {
                "id": os.environ.get("QWEN1_5_7B_CHAT", "qwen1.5-7b-chat"),
                "max_tokens": 1000000,
                "used_tokens": 0,
                "type": "text"
            }
        }
        
        # 默认模型
        self.default_model = os.environ.get("DEFAULT_MODEL", "qwen-plus")
        
        # Qwen API配置
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")
        
        # 从文件加载token使用情况（如果存在）
        self._load_token_usage()
    
    def _load_token_usage(self):
        """
        从文件加载token使用情况
        """
        try:
            if os.path.exists("token_usage.json"):
                with open("token_usage.json", "r", encoding="utf-8") as f:
                    usage_data = json.load(f)
                    for model_name, usage in usage_data.items():
                        if model_name in self.models:
                            self.models[model_name]["used_tokens"] = usage
        except Exception as e:
            # 确保异常信息正确显示中文
            error_msg = str(e).encode('utf-8').decode('utf-8')
            print(f"加载token使用情况失败: {error_msg}")
    
    def _save_token_usage(self):
        """
        保存token使用情况到文件
        """
        try:
            usage_data = {}
            for model_name, model_info in self.models.items():
                usage_data[model_name] = model_info["used_tokens"]
            
            with open("token_usage.json", "w", encoding="utf-8") as f:
                json.dump(usage_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 确保异常信息正确显示中文
            error_msg = str(e).encode('utf-8').decode('utf-8')
            print(f"保存token使用情况失败: {error_msg}")
    
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
    
    def parse_request_with_files(self, request: str) -> Tuple[str, Dict[str, str]]:
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
                # 在请求中用文件内容替换引用
                processed_request = processed_request.replace(f'[file:{file_path}]', f"[文件内容开始:{file_path}]\n{content}\n[文件内容结束:{file_path}]")
            else:
                print(f"警告: 无法读取文件 '{file_path}'，将在请求中保留原始引用")
        
        return processed_request, file_contents
    
    def fetch_token_usage_from_qwen(self) -> Dict[str, int]:
        """
        从Qwen平台获取实时token使用情况
        
        Returns:
            各模型的token使用情况
        """
        # 检查API密钥是否存在
        if not self.api_key:
            print("警告: 未配置API密钥，无法获取实时token使用情况")
            # 返回本地缓存的数据作为后备
            local_usage = {}
            for model_name, model_info in self.models.items():
                local_usage[model_name] = model_info["used_tokens"]
            return local_usage
        
        try:
            # 调用Qwen平台的API来获取实时token使用情况
            # 注意：这需要根据Qwen平台的实际API文档进行调整
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept-Language": "zh-CN,zh;q=0.9"
            }
            
            # 这里使用DashScope的通用API端点作为示例
            # 实际使用时需要根据Qwen平台的具体API文档进行调整
            response = requests.get(
                f"{self.base_url}/services",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                # 解析返回的数据以提取token使用情况
                # 这里的解析逻辑需要根据实际API返回的数据结构进行调整
                usage_data = {}
                if "services" in data:
                    for service in data["services"]:
                        if "name" in service and "usage" in service:
                            model_name = service["name"]
                            # 将DashScope服务名称映射到我们内部的模型名称
                            internal_model_name = self._map_service_name_to_model(model_name)
                            if internal_model_name:
                                usage_data[internal_model_name] = service["usage"].get("total_tokens", 0)
                
                return usage_data
            else:
                print(f"获取token使用情况失败: HTTP {response.status_code}")
                # 返回本地缓存的数据作为后备
                local_usage = {}
                for model_name, model_info in self.models.items():
                    local_usage[model_name] = model_info["used_tokens"]
                return local_usage
                
        except Exception as e:
            # 处理中文编码异常
            error_msg = str(e).encode('utf-8').decode('utf-8')
            print(f"从Qwen平台获取token使用情况时出错: {error_msg}")
            # 返回本地缓存的数据作为后备
            local_usage = {}
            for model_name, model_info in self.models.items():
                local_usage[model_name] = model_info["used_tokens"]
            return local_usage
    
    def _map_service_name_to_model(self, service_name: str) -> Optional[str]:
        """
        将DashScope服务名称映射到内部模型名称
        
        Args:
            service_name: DashScope服务名称
            
        Returns:
            内部模型名称，如果无法映射则返回None
        """
        # 这里需要根据实际的服务名称和我们内部模型名称的对应关系进行映射
        service_to_model_map = {
            "qwen-turbo": "qwen-turbo",
            "qwen-plus": "qwen-plus",
            "qwen-max": "qwen-max",
            "qwen-vl-max": "qwen-vl-max",
            "qwen-vl-plus": "qwen-vl-plus"
            # 根据实际需要添加更多映射
        }
        return service_to_model_map.get(service_name)
    
    def update_token_usage(self, model_name: str, tokens: int):
        """
        更新模型的token使用情况
        
        Args:
            model_name: 模型名称
            tokens: 使用的token数量
        """
        if model_name in self.models:
            # 累加token使用量
            self.models[model_name]["used_tokens"] += tokens
            # 保存到文件
            self._save_token_usage()
    
    def refresh_token_usage(self):
        """
        从Qwen平台刷新所有模型的token使用情况
        """
        print("正在从Qwen平台获取实时token使用情况，请稍候...")
        real_time_usage = self.fetch_token_usage_from_qwen()
        
        # 更新本地模型的token使用情况
        for model_name, used_tokens in real_time_usage.items():
            if model_name in self.models:
                self.models[model_name]["used_tokens"] = used_tokens
        
        # 保存更新后的数据
        self._save_token_usage()
        print("token使用情况已更新")
    
    def get_model_by_name(self, model_name: str) -> Optional[Dict]:
        """
        根据名称获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息，如果不存在则返回None
        """
        model_info = self.models.get(model_name)
        if model_info:
            # 确保模型信息包含所有必要的字段
            model_info.setdefault("id", model_name)
            model_info.setdefault("max_tokens", 1000000)
            model_info.setdefault("used_tokens", 0)
            model_info.setdefault("type", "text")
        return model_info
    
    def get_available_models(self) -> List[str]:
        """
        获取所有可用的模型列表（使用率低于80%）
        
        Returns:
            可用模型列表
        """
        # 首先刷新token使用情况
        self.refresh_token_usage()
        
        available_models = []
        for model_name, model_info in self.models.items():
            # 检查token使用是否超过80%
            usage_ratio = model_info["used_tokens"] / model_info["max_tokens"]
            if usage_ratio < 0.8:
                available_models.append(model_name)
        
        # 如果没有可用模型，返回所有模型（作为后备方案）
        if not available_models:
            available_models = list(self.models.keys())
            
        return available_models
    
    def select_model_for_task(self, task_type: str) -> str:
        """
        根据任务类型选择合适的模型
        
        Args:
            task_type: 任务类型
            
        Returns:
            选定的模型名称
        """
        available_models = self.get_available_models()
        
        # 如果没有可用模型，使用默认模型
        if not available_models:
            return self.default_model
        
        # 根据任务类型选择模型
        if task_type == "multimodal":
            # 多模态任务
            for model_name in ["qwen-vl-max", "qwen-vl-plus"]:
                if model_name in available_models:
                    return model_name
        elif task_type == "complex":
            # 复杂任务
            for model_name in ["qwen-max", "qwen3", "qwen-plus"]:
                if model_name in available_models:
                    return model_name
        elif task_type == "simple":
            # 简单任务
            for model_name in ["qwen-turbo", "qwen-plus", "qwen-max"]:
                if model_name in available_models:
                    return model_name
        else:
            # 默认任务
            for model_name in [self.default_model, "qwen-plus", "qwen-max"]:
                if model_name in available_models:
                    return model_name
        
        # 如果没有找到合适的模型，返回第一个可用模型
        return available_models[0]
    
    def get_model_usage_ratio(self, model_name: str) -> float:
        """
        获取模型的使用比例
        
        Args:
            model_name: 模型名称
            
        Returns:
            使用比例 (0-1)
        """
        if model_name in self.models:
            model_info = self.models[model_name]
            return model_info["used_tokens"] / model_info["max_tokens"]
        return 1.0  # 如果模型不存在，返回100%使用率
    
    def get_model_info(self) -> Dict[str, Dict]:
        """
        获取所有模型的详细信息
        
        Returns:
            所有模型的详细信息
        """
        # 刷新token使用情况以获取最新数据
        self.refresh_token_usage()
        # 确保所有模型信息完整
        for model_name, model_info in self.models.items():
            model_info.setdefault("id", model_name)
            model_info.setdefault("max_tokens", 1000000)
            model_info.setdefault("used_tokens", 0)
            model_info.setdefault("type", "text")
        return self.models