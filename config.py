"""
@Desc: 全局配置文件读取
"""
import argparse
import yaml
from typing import Dict, List
import os
import shutil
import sys


class Data_pack_config:
    def __init__(
        self,
        filelist_path: str = "file.list",
        config_path: str = "config.json",
        audio_path: str = "audios",
        data_path: str = "datas",  # data_pack输出
        num_process: int = 1,
    ):
        pass


class Train_ms_config:
    """训练配置"""

    def __init__(
        self,
        config_path: str,
        env: Dict[str, any],
        base: Dict[str, any],
        model: str,
    ):
        self.env = env  # 需要加载的环境变量
        self.base = base  # 底模配置
        self.model = model  # 训练模型存储目录，该路径为相对于dataset_path的路径，而非项目根目录
        self.config_path = config_path  # 配置文件路径

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        # data["model"] = os.path.join(dataset_path, data["model"])
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)


class Webui_config:
    """webui 配置"""

    def __init__(
        self,
        device: str,
        model: str,
        config_path: str,
        language_identification_library: str,
        port: int = 7860,
        share: bool = False,
        debug: bool = False,
    ):
        self.device: str = device
        self.model: str = model  # 端口号
        self.config_path: str = config_path  # 是否公开部署，对外网开放
        self.port: int = port  # 是否开启debug模式
        self.share: bool = share  # 模型路径
        self.debug: bool = debug  # 配置文件路径
        self.language_identification_library: str = (
            language_identification_library  # 语种识别库
        )

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        data["config_path"] = os.path.join(dataset_path, data["config_path"])
        data["model"] = os.path.join(dataset_path, data["model"])
        return cls(**data)


class Server_config:
    def __init__(
        self, models: List[Dict[str, any]], port: int = 5000, device: str = "cuda"
    ):
        self.models: List[Dict[str, any]] = models  # 需要加载的所有模型的配置
        self.port: int = port  # 端口号
        self.device: str = device  # 模型默认使用设备

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


class Translate_config:
    """翻译api配置"""

    def __init__(self, app_key: str, secret_key: str):
        self.app_key = app_key
        self.secret_key = secret_key

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


class Config:
    def __init__(self, config_path: str):
        if not os.path.isfile(config_path) and os.path.isfile("default_config.yml"):
            shutil.copy(src="default_config.yml", dst=config_path)
            print(
                f"已根据默认配置文件default_config.yml生成配置文件{config_path}。请按该配置文件的说明进行配置后重新运行。"
            )
            print("如无特殊需求，请勿修改default_config.yml或备份该文件。")
            sys.exit(0)
        with open(file=config_path, mode="r", encoding="utf-8") as file:
            yaml_config: Dict[str, any] = yaml.safe_load(file.read())
            dataset_path: str = yaml_config["dataset_path"]
            self.dataset_path: str = dataset_path
            self.train_ms_config: Train_ms_config = Train_ms_config.from_dict(
                dataset_path, yaml_config["train_ms"]
            )
            self.webui_config: Webui_config = Webui_config.from_dict(
                dataset_path, yaml_config["webui"]
            )
            self.server_config: Server_config = Server_config.from_dict(
                yaml_config["server"]
            )
            self.translate_config: Translate_config = Translate_config.from_dict(
                yaml_config["translate"]
            )


parser = argparse.ArgumentParser()
# 为避免与以前的config.json起冲突，将其更名如下
parser.add_argument("-y", "--yml_config", type=str, default="config.yml")
args, _ = parser.parse_known_args()
config = Config(args.yml_config)
