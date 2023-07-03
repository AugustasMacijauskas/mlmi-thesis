from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser

from trl import PPOConfig


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    
    reward_model_output_path: Optional[str] = field(default="", metadata={"help": "the reward model name"})

    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    remove_unused_columns: Optional[bool] = field(default=True, metadata={"help": "whether to remove unused columns"})
    
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    logging_dir: Optional[str] = field(default=None, metadata={"help": "the logging directory"})
    
    learning_rate: Optional[float] = field(default=1.4e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    steps: Optional[int] = field(default=5000, metadata={"help": "number of steps"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})

    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="../runs_ppo/", metadata={"help": "the directory to save the model to"})


def get_script_args(local_args=None):
    parser = HfArgumentParser(ScriptArguments)
    # script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    script_args: ScriptArguments = parser.parse_args_into_dataclasses(local_args)[0]

    return script_args


def get_ppo_config(script_args: ScriptArguments):
    return PPOConfig(
        model_name=script_args.model_name,
        remove_unused_columns=script_args.remove_unused_columns,
        log_with=script_args.log_with,
        learning_rate=script_args.learning_rate,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        steps=script_args.steps,
        ppo_epochs=script_args.ppo_epochs,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
        init_kl_coef=script_args.init_kl_coef,
        adap_kl_ctrl=script_args.adap_kl_ctrl,
        seed=script_args.seed,
        optimize_cuda_cache=True,
        project_kwargs={ "logging_dir": script_args.logging_dir },
    )


def main():
    script_args = get_script_args()
    print(script_args)


if __name__ == "__main__":
    main()
