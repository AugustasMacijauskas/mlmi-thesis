# Eliciting latent knowledge from language reward models

Code for my thesis titled "Eliciting latent knowledge from language reward models" for the MPhil in Machine Learning and Machine Intelligence at the University of Cambridge.


## Idea

![The architecture of the reward model](assets/reward_model.png)

Use methods that _discover latent knowledge_ (DLK), such as <a href="https://arxiv.org/abs/2212.03827" target="_blank">CCS</a>, to build reward models that promote truthfulness. Utilize these reward models to execute _reinforcement learning_ (RL) fine-tuning to improve the "truthfulness" of LLMs.

For more details, see the accompanying <a href="https://augustasmacijauskas.github.io/personal-website/posts/thesis/thesis.html" target="_blank">blog post</a> and the <a href="https://augustasmacijauskas.github.io/personal-website/posts/thesis/mlmi-thesis.pdf" target="_blank">full pdf of the thesis</a>.


## Installation and prerequisites

1. Clone the repository.
1. Install the `EleutherAI/elk` library. The version from <a href="https://github.com/EleutherAI/elk/tree/a2904e62765fa311b1197505f78fab295e1c87fb" target="_blank">this</a> commit was used (though trying their newest techniques might be worth a try too). Installation instructions can be found in the README of the provided link.
1. Install the _Language Model Evaluation Harness_ (<a href="https://github.com/EleutherAI/lm-evaluation-harness" target="_blank">EleutherAI/lm-evaluation-harness</a>). To make sure my results match with _Open LLM Leaderboard_ (<a href="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard" target="_blank">link</a>), <a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463" target="_blank">this</a> version of the harness was used.
1. The version of the harness used by the Open LLM Leaderboard does not support distributed inference, so the version of the harness on the _big-refactor_ branch was also used. The version from <a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/2820042d05e91c87852c82293f8973dc841c1a25" target="_blank">this</a> commit was used, but again, checking the current state of the branch might be worth it.

The following <a href="https://www.git-tower.com/learn/git/faq/git-checkout-commits" target="_blank">guide</a> might be useful to checkout to a desired commit, but the main command you want to use is this:
```bash
git checkout <commit-id>
```
and then you can use
```bash
git show HEAD
```
to see if you are using the right version of the code.


## Usage

There are four main steps to run the method on new data:
1. Split the dataset and prepare it for reward model training and RL fine-tuning.
1. Train a reward model.
1. Performing RL fine-tuning on some pre-trained LLM.
1. Evaluate the pre-trained LLM on both target and general NLP tasks.

Not all steps are fully automated, so some manual work has to be done, as explained in more detail below.


### Dataset preparation

The first thing that one has to do is prepare the dataset for reward model training and RL fine-tuning. The notebooks under in the [src/dataset_formation/](https://github.com/AugustasMacijauskas/mlmi-thesis/tree/main/src/dataset_formation) folder can be used for that. Make sure you create a `src/dataset_formation/datasets/` folder which is ignored in the remote repository, but will contain the temporary files created in the process before the datasets are pushed to Hugging Face Hub. 


Most commonly, the workflow is as follows:
1. Use the `form_train_ppo_datasets.ipynb` notebook to first split the training data from a dataset into `train` and `ppo` splits. The former will be used for reward model training, and the latter will be used for RL fine-tuning (it is called _PPO_ in the PPO algorithm is used for fine-tuning). The notebook requires some manual work to be done, some cells are commented out when they should not be, but hopefully staring at the code enough will make it clear what is going on, and make sure to refer to my thesis blog post and the pdf (linked above). Roughly speaking:
    - Split the data.
    - Choose a template(s) to be applied for the `train` split and apply it.
1. Use the `form_val_dataset.ipynb` to form the `val` split. Choosing and applying a template is required this time as well.
1. Use the `push_dataset_to_hub.ipynb` notebook to combine the formed temporary files and push them to Hugging Face Hub.

There are many datasets on my <a href="https://huggingface.co/AugustasM" target="_blank">HF profile</a> available to use for the later steps of the pipeline so that you don't have to do anything if you do not want to. In particular, the processed QNLI dataset might be of interest (<a href="https://huggingface.co/datasets/AugustasM/qnli-vicuna-v1" target="_blank">train/val</a> and <a href="https://huggingface.co/datasets/AugustasM/qnli-vicuna-ppo-training-v1" target="_blank">ppo</a>).

Note that there are more datasets under `src/dataset_formation/`, but they are the same thing, just specifically shaped to work only for a particular dataset.


## Reward model training
