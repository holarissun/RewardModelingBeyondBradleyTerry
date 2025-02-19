# Official Implementation for [ICLR'2025 Paper](https://openreview.net/forum?id=rfdblE10qm&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions))

## Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives

#### Authors: Hao Sun*, Yunyi Shen*, Jean-Francois Ton. The first two authors contribute equally.

[ [Website] ](https://sites.google.com/view/rewardmodels)        |      [ [Preprint] ](https://arxiv.org/pdf/2411.04991)       |       [Embeddings (To be released soon)]     |     [Code (To be released soon)]

_We have a series of work focusing on reward models in RLHF:_
- Part I. Reward Model Foundation (This Repo.)
- Part II. Active Reward Modeling (SOON)
- Part III. Accelerating Reward Model Research with our Infra. (SOON)

----
## Infra for Easy-Reproducible Reward Model Research
The reproduction for reward modeling research has long been a challenge, given its high demand for hardware and cost in training, evaluation, and inference. We propose to conduct easy-reproducible reward model research on the embedding space.

Details of the workflow are posited in this paper: [Part III. TO BE RELEASED SOON.]. Our motivation is to make every researcher with a single CPU can also conduct reward modeling (and RLHF) research.

## Reproducing the Results with a CPU
- Step 1 (optional, GPU required): SFT (you need to update the PATH to the models/open-sourced datasets. You may need to apply for licences to use those models/datasets first.) Note that
```python
python3 step1_sft.py --model_name gemma2b --dataset hh-rlhf-helpful-gpt4
```

- Step 2 (optional, GPU required): Generate samples on training (10 per prompt) and testing prompts (500 per prompt)
- Step 3 (optional, GPU required): annotating response qualities using golden reward models
- Step 4 (optional, GPU required): Generate and store embeddings of all prompt-response pairs

### The above 4 steps enable us to create an embedding-based dataset, then we can easily reproduce any research with such a dataset

To illustrate:
![example code](demo.png)

- Step 5



## Call for Contribution to the Infra (an Embedding-based Dataset for Reward Modeling Research)

`Call for contributors! --- Please contact me at sunhopht@gmail.com if your are interested in contributing your embedding / golden-reward annotations in your reward model research to the open-source RM community!`





