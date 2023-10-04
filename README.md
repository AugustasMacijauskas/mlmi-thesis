# Eliciting latent knowledge from language reward models

Code for my thesis titled "Eliciting latent knowledge from language reward models" for the MPhil in Machine Learning and Machine Intelligence at the University of Cambridge.


# Idea

![The architecture of the reward model](assets/reward_model.png)

Use methods that _discover latent knowledge_ (DLK), such as <a href="https://arxiv.org/abs/2212.03827" target="_blank">CCS</a>, to build reward models that promote truthfulness. Utilize these reward models to execute _reinforcement learning_ (RL) fine-tuning to improve the "truthfulness" of LLMs.

For more details, see the accompanying <a href="https://augustasmacijauskas.github.io/personal-website/posts/thesis/thesis.html" target="_blank">blog post</a> and the <a href="https://augustasmacijauskas.github.io/personal-website/posts/thesis/mlmi-thesis.pdf" target="_blank">full pdf of the thesis</a>.


# Installation and prerequisites

1. Clone the repository.
1. Create a new conda environment in which all the libraries will be downloaded. Note I have dumped the dependencies used in my environment at the end of the project into `environment.yml` file, but it is not always possible to easily install from it by simply using:
    ```bash
    conda env create -f environment.yml
    ```
    but do try referencing it and you may find it helpful.
1. Install the `EleutherAI/elk` library. The version from <a href="https://github.com/EleutherAI/elk/tree/a2904e62765fa311b1197505f78fab295e1c87fb" target="_blank">this</a> commit was used (though trying their newest techniques might be worth a try too). Installation instructions can be found in the README of the provided link. Install to a folder adjacent to directory containing the cloned repository.
    - Once installed copy the `custom-prompts/AugustasM/` folder into `elk/elk/promptsource/templates/`, i.e. you want a folder `elk/elk/promptsource/templates/AugustasM` to exist.
1. Install the _Language Model Evaluation Harness_ (<a href="https://github.com/EleutherAI/lm-evaluation-harness" target="_blank">EleutherAI/lm-evaluation-harness</a>). To make sure my results match with _Open LLM Leaderboard_ (<a href="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard" target="_blank">link</a>), <a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463" target="_blank">this</a> version of the harness was used. Installation instructions can be found in the README. Install to a folder adjacent to directory containing the cloned repository.
1. The version of the harness used by the Open LLM Leaderboard does not support distributed inference, so the version of the harness on the _big-refactor_ branch was also used. The version from <a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/2820042d05e91c87852c82293f8973dc841c1a25" target="_blank">this</a> commit was used, but again, checking the current state of the branch might be worth it. Installation instructions can be found in the README. Install to a folder adjacent to directory containing the cloned repository. To avoid clash with the original harness repository cloned above, you can wrap this version into another folder, e.g. I installed into `~/lm_evalution_harness_refactored/lm-evalution-harness`.
    - Once installed copy the files in `custom-prompts/qnli/*` into `~/<wrap-directory>/lm-evalution-harness/tasks/glue/qnli/`, e.g. in my case I copied to files to `~/lm_evalution_harness_refactored/lm-evalution-harness/tasks/glue/qnli/` folder. Make sure you copy the files and not the folder, i.e. you want to extend the contents of the existing `glue/qnli/` folder.

The following <a href="https://www.git-tower.com/learn/git/faq/git-checkout-commits" target="_blank">guide</a> might be useful to checkout to a desired commit, but the main command you want to use is this:
```bash
git checkout <commit-id>
```
and then you can use
```bash
git show HEAD
```
to see if you are using the right version of the code.


# Usage

There are four main steps to run the method on new data:
1. Split the dataset and prepare it for reward model training and RL fine-tuning.
1. Train a reward model.
1. Performing RL fine-tuning on some pre-trained LLM.
1. Evaluate the fine-tuned LLM on both target and general NLP tasks.

Not all steps are fully automated, so some manual work has to be done, as explained in more detail below.


## Dataset preparation

The first thing that one has to do is prepare the dataset for reward model training and RL fine-tuning. The notebooks under in the [src/dataset_formation/](https://github.com/AugustasMacijauskas/mlmi-thesis/tree/main/src/dataset_formation) folder can be used for that. Make sure you create a `src/dataset_formation/datasets/` folder which is ignored in the remote repository, but will contain the temporary files created in the process before the datasets are pushed to Hugging Face Hub. 


Most commonly, the workflow is as follows:
1. Use the `form_train_ppo_datasets.ipynb` notebook to first split the training data from a dataset into `train` and `ppo` splits. The former will be used for reward model training, and the latter will be used for RL fine-tuning (it is called _PPO_ in the PPO algorithm is used for fine-tuning). The notebook requires some manual work to be done, some cells are commented out when they should not be, but hopefully staring at the code enough will make it clear what is going on, and make sure to refer to my thesis blog post and the pdf (linked above). Roughly speaking:
    - Split the data.
    - Choose a template(s) to be applied for the `train` split and apply it.
1. Use the `format_ppo_training_dataset.ipynb` to apply a chosen template on the `ppo` split.
1. Use the `form_val_dataset.ipynb` to form the `val` split. Choosing and applying a template is required this time as well.
1. Use the `push_dataset_to_hub.ipynb` notebook to combine the formed temporary files and push them to Hugging Face Hub.

There are many datasets on my <a href="https://huggingface.co/AugustasM" target="_blank">HF profile</a> available to use for the later steps of the pipeline so that you don't have to do anything if you do not want to. In particular, the processed QNLI dataset might be of interest (<a href="https://huggingface.co/datasets/AugustasM/qnli-vicuna-v1" target="_blank">train/val</a> and <a href="https://huggingface.co/datasets/AugustasM/qnli-vicuna-ppo-training-v1" target="_blank">ppo</a>).

Note that there are more datasets under `src/dataset_formation/`, but they are the same thing, just specifically shaped to work only for a particular dataset.


## Reward model training

Reward model training involves getting a few prerequisites right and then editing and running a batch script that trains a probe on a given datasets and saves the trained weights. A few things to notice:
1. Make sure you have your conda environment with all of the required dependencies activated.
1. Create a folder called `logs_elk/` adjacent to the whatever you called the folder for the cloned repository (should be called `mlmi-thesis` by default). The logs about probe training will be saved here.
1. Create a folder called `elk-probes/` which will contain all of the trained probes that we will use to build reward models.

Note that I only provide scripts that can be executed on a computing cluster that uses SLURM to obtain the trained probes. The `elk` library also provides ways to do this right from command line, check their documentation if this is something that you need.

With the prerequisites out of the way, open the `scripts/elk.sh` file and edit it to your liking to train a probe that you need. Make sure to carefully look through the file to find all of the available options.

Finally, you are ready to execute the batch script and train a probe. Run the following commands:
```bash
cd mlmi-thesis/ # Important!
scripts/launchers/run_elk.sh
```


## RL fine-tuning

Before running the code, create a `ppo_logs/` folder adjacent to the cloned repository.

Edit the `scripts/ppo_vicuna.sh` file to your liking. The script was tested with the distributed data parallel training and using 8 bit quantization. However, other configurations may work as well. Once you are finished with editing the script, execute the following:
```bash
cd mlmi-thesis/ # Important!
scripts/launchers/run_ppo_vicuna.sh
```
Note that this will use `wandb` logging, you can edit the project name in the `src/ppo/configs.py` file in the `get_ppo_config()` function under the `tracker_project_name` attribute.

If you want to see the code itself, it is contained in the `src/ppo/` folder. For example, you might want to do this to change the quantization type (currently, 8 bit quantization is hard-coded).

You can use `src/utils/merge_lora_weights.ipynb` to merge the trained LoRA matrices into the model and push to hub if needed.


## Evaluate the fine-tuned LLM

Finally, you can evaluate the fine-tuned model on the target ask and general NLP tasks. The only target task used in the thesis was the <a href="https://huggingface.co/datasets/glue/viewer/qnli/train" target="_blank">QNLI dataset</a>, so you might have to play around a bit to implement your new custom task. The general NLP tasks are the ones from Open LLM Leaderboard.

As a prerequisite, create a `logs_eval/` folder adjacent to the cloned repository.

To evaluate on the QNLI task, edit the `scripts/eval_harness_qnli_vicuna.sh` file, then execute:
```bash
cd mlmi-thesis/ # Important!
scripts/launchers/run_eval_harness_qnli_vicuna.sh
```
This will run 8 bit inference using the new LoRA weights, or using the original models of an empty string is passed instead of LoRA weights.

To evaluate on the Open LLM Leaderboard datasets, edit the `scripts/eval_harness_qnli_vicuna_openllm.sh` file. It works very similarly (mostly indentically) to the script above.

After the execution finishes, the results will be available in the `logs_eval/` folder.

