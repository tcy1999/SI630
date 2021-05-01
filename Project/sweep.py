from preprocess import load_data
import logging
from simpletransformers.ner import NERModel, NERArgs
import wandb

train_df, trial_df, test = load_data()

sweep_config = {
    "method": "grid",  # bayes, random
    "metric": {"name": "train_loss", "goal": "minimize"}, # "metric": {"name": "accuracy", "goal": "maximize"},
    # "parameters":  {"num_train_epochs": {"min": 10, "max": 50}, "learning_rate": {"min": 0, "max": 4e-4}},
    "parameters": {"num_train_epochs": {"values": [10, 30, 50]}, "learning_rate": {"values": [3e-7, 3e-5, 3e-4]}, "train_batch_size":{"values": [32, 64]}},
    "early_terminate": {"type": "hyperband", "min_iter": 6},
}

sweep_id = wandb.sweep(sweep_config, project="Simple-sweep")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = NERArgs()
model_args.labels_list = ["T", "N"]
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.output_dir = "./output-ner"
model_args.eval_batch_size = 32
model_args.adam_epsilon = 1e-9
model_args.mlm = True
model_args.manual_seed = 2021
model_args.evaluate_during_training = True
model_args.wandb_project = "Simple-sweep"
model_args.no_cache = True
model_args.no_save = True


def training():
    wandb.init()
    
    model = NERModel("roberta", "roberta-base", use_cuda=True, args=model_args, sweep_config=wandb.config)
    # model = NERModel("distilbert", "distilbert-base-cased", use_cuda=True, args=model_args, sweep_config=wandb.config)
    model.train_model(train_df, eval_data=trial_df)
    
    wandb.join()


wandb.agent(sweep_id, training)
