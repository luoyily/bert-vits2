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
        train_filelist: str,
        train_audios: str,
        train_datas: str,
        eval_filelist: str,
        eval_audios: str,
        eval_datas: str,  # data_pack输出
        num_process: int,
        add_blank: bool,
        config_in: str,
        config_out: str,
    ):
        self.train_filelist: str = train_filelist  # 训练集文件
        self.train_audios: str = train_audios  # 训练集音频目录
        self.train_datas: str = train_datas  # 训练集预处理输出目录
        self.eval_filelist: str = eval_filelist  # 验证集文件
        self.eval_audios: str = eval_audios  # 验证集音频目录
        self.eval_datas: str = eval_datas  # 验证集预处理输出目录
        self.num_process: int = num_process  # 处理并发数
        self.add_blank: bool = add_blank  # TODO
        self.config_in: str = config_in  # config.json 模板目录 唯独这个路径是相对于项目根的，请注意。
        self.config_out: str = config_out  # config.json 输出目录

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        # 反序列化
        data["train_filelist"] = os.path.join(dataset_path, data["train_filelist"])
        data["train_audios"] = os.path.join(dataset_path, data["train_audios"])
        data["train_datas"] = os.path.join(dataset_path, data["train_datas"])
        data["eval_filelist"] = os.path.join(dataset_path, data["eval_filelist"])
        data["eval_audios"] = os.path.join(dataset_path, data["eval_audios"])
        data["eval_datas"] = os.path.join(dataset_path, data["eval_datas"])
        data["config_out"] = os.path.join(dataset_path, data["config_out"])

        return cls(**data)


class Train_config:
    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        data["model_dir"] = os.path.join(dataset_path, data["model_dir"])

        return cls(**data)


class Train_ms_config:
    """训练配置"""

    def __init__(
        self,
        env: Dict[str, any],
    ):
        self.env = env  # 需要加载的环境变量

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
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


class Hiyori_UI_config:
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
            self.data_pack_config: Data_pack_config = Data_pack_config.from_dict(
                dataset_path, yaml_config["data_pack"]
            )
            self.train_ms_config: Train_ms_config = Train_ms_config.from_dict(
                yaml_config["train_ms"]
            )
            self.webui_config: Webui_config = Webui_config.from_dict(
                dataset_path, yaml_config["webui"]
            )
            self.hiyori_UI_config: Hiyori_UI_config = Hiyori_UI_config.from_dict(
                yaml_config["hiyori_UI"]
            )
            self.translate_config: Translate_config = Translate_config.from_dict(
                yaml_config["translate"]
            )
            self.train_config: Train_config = Train_config.from_dict(
                dataset_path, yaml_config["train"]
            )


parser = argparse.ArgumentParser()
# 为避免与以前的config.json起冲突，将其更名如下
parser.add_argument("-y", "--yml_config", type=str, default="config.yml")
args, _ = parser.parse_known_args()
config = Config(args.yml_config)
