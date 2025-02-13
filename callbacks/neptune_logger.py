import types
import argparse
from typing import Union
import neptune.new as neptune
from neptune_pytorch import NeptuneLogger


class CustomNeptuneLogger:
    def __init__(self, **kwargs):
        token = kwargs["token"]
        project = kwargs["project"]
        self.config = kwargs["config"]
        self.data_config_path = kwargs["data_config_path"]
        self.yolo_args = kwargs["yolo_args"]
        self.config_path = kwargs["config_path"]
        self.device = kwargs["device"]
        self.run = neptune.init_run(api_token=token, project=project)
        self.logger = None

    def log_config_to_neptune(self) -> None:
        with open(self.data_config_path, 'r') as file:
            self.run["data_config"].upload(self.data_config_path)
            
        with open(self.yolo_args, 'r') as file:
            self.run["yolo_args"].upload(self.yolo_args)

        with open(self.config_path, 'r') as file:
            self.run["config/config_file"].upload(self.config_path)

        self.log_nested_parameters("config", self.config)

    def log_nested_parameters(self, key_path, namespace) -> None:
        if isinstance(namespace, types.SimpleNamespace):
            for key, value in vars(namespace).items():
                new_key_path = f"{key_path}/{key}"
                if isinstance(value, types.SimpleNamespace):
                    # Recursive call for nested Namespace
                    self.log_nested_parameters(new_key_path, value)
                else:
                    # Convert to supported type and log
                    self.run[new_key_path] = self.convert_to_supported_type(value)
        elif isinstance(namespace, dict):
            for key, value in namespace.items():
                new_key_path = f"{key_path}/{key}"
                if isinstance(value, (types.SimpleNamespace, dict)):
                    # Recursive call for nested Namespace or dict
                    self.log_nested_parameters(new_key_path, value)
                else:
                    # Convert to supported type and log
                    self.run[new_key_path] = self.convert_to_supported_type(value)
        else:
            # If not a namespace or dict, simply log the value directly
            self.run[key_path] = self.convert_to_supported_type(namespace)

    def convert_to_supported_type(self, value) -> Union[list, dict, str]:
        if isinstance(value, argparse.Namespace):
            return vars(value)  # Convert Namespace to dict
        if isinstance(value, list) or isinstance(value, tuple):
            return str(value)  # Convert list or tuple to string
        if isinstance(value, dict):
            return {k: self.convert_to_supported_type(v) for k, v in value.items()}  # Convert each item in the dict
        return value

    def start_logging(self, model) -> None:
        dim = self.config.augmentation.resize_param
        self.log_config_to_neptune()  # Log the config before starting
        self.logger = NeptuneLogger(run=self.run, model=model)
        
    def log_metric(self, name: str, value: Union[int, float]) -> None:
        """
        Logs a single metric to Neptune.
        
        :param name: The name of the metric.
        :param value: The value of the metric.
        """
        self.run[name].append(value)

    def save_checkpoint(self, checkpoint_name) -> None:
        self.logger.save_checkpoint(checkpoint_name=checkpoint_name)

    def save_model(self) -> None:
        self.logger.save_model()

    def stop(self) -> None:
        self.run.stop()
