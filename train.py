import os
import torch
from trainer.trainer import CustomTrainer
from utils.custom_argument_parser import CustomArgParser


# Function to update the first two lines of data.yaml paths dynamically
def update_data_yaml(data_yaml_path, train_path, val_path):
    with open(data_yaml_path, 'r') as file:
        lines = file.readlines()

    lines[0] = f"train: {train_path}\n"
    lines[1] = f"val: {val_path}\n"

    with open(data_yaml_path, 'w') as file:
        file.writelines(lines)

    
def get_model_yaml(config):
    if config.model.backend == "yolo":
        return f'{config.model.model_type}.yaml'
    else:
        model_type = config.model.model_type[:-1]
        return f'models/{model_type}_{config.model.backend}.yaml'
    

# Callback to upload model weights
def log_model(trainer):
    last_weight_path = trainer.last
    print(f"Model weights saved at: {last_weight_path}")


def main():
    path = os.path.abspath('configuration.ini')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n\nTraining on {device}\n\n")

    arg_parser = CustomArgParser(config_path=path)
    parsed_config = arg_parser.get_parsed_config()

    # Determine the relative paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'data/' + str(parsed_config.data_loader.dataset_name) + '/images/train')
    val_path = os.path.join(base_dir, 'data/' + str(parsed_config.data_loader.dataset_name) + '/images/val')

    # Update the data.yaml file with the correct paths
    data_yaml_path = os.path.join(base_dir, 'data/' + str(parsed_config.data_loader.dataset_name) + '.yaml')
    update_data_yaml(data_yaml_path, train_path, val_path)

    model = get_model_yaml(parsed_config)

    overrides = {
        'data': data_yaml_path,
        'epochs': parsed_config.trainer.n_epochs,
        'batch': parsed_config.data_loader.batch_size_train,
        'imgsz': parsed_config.augmentation.resize_param,
        'device': device,
        'cfg': model,
        'workers': parsed_config.data_loader.num_workers,
        'seed': parsed_config.trainer.seed,
        'pretrained': parsed_config.model.pretrained,
        'resume': False,  # Ensure not resuming from any checkpoint
        'model': model, # Train model from scratch
        'conf': 0.3,
        'lr0': parsed_config.optim.learning_rate_initial,
        'lrf': parsed_config.optim.learning_rate_final,
        'momentum': parsed_config.optim.momentum,
        'weight_decay': parsed_config.optim.weight_decay,
        'warmup_epochs': parsed_config.optim.warmup_epochs,
        'optimizer': parsed_config.optim.optimizer, 
        'patience': parsed_config.trainer.n_epochs,  # No early stopping
        'mosaic': parsed_config.augmentation.mosaic,
        'cos_lr' : parsed_config.optim.cos_lr,
        'rect' : parsed_config.data_loader.rect, # Improve train images aspect ratio
        'translate' : parsed_config.augmentation.translate,
        'scale' : parsed_config.augmentation.scale
    }

    path = os.path.abspath('configuration.ini')

    trainer = CustomTrainer(config=parsed_config, overrides=overrides, config_path=path)
    trainer.train()


if __name__ == "__main__":
    main()
