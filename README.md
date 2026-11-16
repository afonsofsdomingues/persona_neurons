# Neuron-level origins of misaligned personas

**Which neurons write a language model's persona, and how does training shape them?**

A mechanistic interpretability project on OLMo 2 1B. Work in progress.

## Background

Persona vectors are directions in a language model's activations that track character
traits such as sycophancy or "evil". They can be used to monitor and steer a model's
persona ([Chen et al., 2025](https://arxiv.org/abs/2507.21509)).

Persona vectors are usually studied as whole directions. For example,
[Moskvoretskii et al. (2026)](https://arxiv.org/abs/2605.13329) trace them through the
training of OLMo 3 and report that they form early in pretraining and are expressed
less after post-training. This project asks a finer question: which neurons write such
a direction, and what does each stage of training change in them?

The motivation is safety. A model's persona emerges during
training, as the model generalises from its data. If we want models whose character we
can trust, we need to understand how that happens, and correcting unwanted behaviour
after training is not enough.
[Biderman et al. (2026)](https://arxiv.org/abs/2606.06533) argue this in general
terms: a science of AI must study the training dynamics that produce a model's
behaviour. Following a persona's neurons from pretraining to the final model is one
concrete way to do that.

The project works at the level of single neurons. In
[*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/), Cammarata
(Goodfire) argues that most neurons have one main job, and that much apparent
polysemanticity is **recursive generalisation**: as a neuron develops during training
it branches into related concepts, the way a tree branch grows sub-branches. This
project treats that as a hypothesis to test on personas, not as a premise.

## Plan

1. **Find the vector.** Extract persona vectors for sycophancy and "evil" from OLMo 2 1B
   and check that they are real: they should read the trait in new responses and
   produce it when added to the model's activations.
2. **Take it apart.** Split each vector into the amount that every neuron and attention
   layer writes into it directly. Test whether a few neurons carry more of it than
   they would for other directions, whether those neurons respond to the model
   speaking *as* the persona and not just to the trait's vocabulary, and whether
   switching them off changes the behaviour.
3. **Follow it through training.** OLMo 2 publishes checkpoints from pretraining and
   from every post-training stage. Each stage continues training the same network, so
   a neuron keeps its index. Following the neurons from part 2 across those checkpoints
   shows when they take on their role, whether they get there by branching from
   related concepts, as recursive generalisation predicts, and what each post-training
   stage does to them.

OLMo 2 1B is used because its whole training pipeline is public and it is small enough
for the code to fit on a single 8 GB GPU, with open-weight models only.

**Status.** Part 1 is implemented as the pipeline below. Of part 2, the direct
attribution and its comparison with random directions are implemented; the vocabulary
control and the ablation tests are not. Part 3 is planned. No results are published in
this repository yet.

## Setup

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Usage

One command runs the whole pipeline for a trait, from the repository root:

```bash
bash scripts/run_all.sh configs/sycophantic_olmo2_1b.yaml
```

Add `--smoke` to try it in about a minute on CPU with a tiny random model and a fake
judge. Smoke numbers are meaningless and go to `results/smoke/`.

A real run needs an NVIDIA GPU with 8 GB of memory. It downloads OLMo 2 1B Instruct
(3 GB) and the two judge models named in the config (about 15 GB each) from Hugging
Face. Results are written to `results/<trait>/`. Files that already exist are reused;
add `--force` to recompute them.

The pipeline, in order. Each script can also be run alone, for example
`uv run python scripts/separation.py --config configs/sycophantic_olmo2_1b.yaml`.

| # | Script | What it does |
|---|---|---|
| 1 | `generate.py` | The model answers each question twice: once told to show the trait, once told to show the opposite. One set of questions is for building the vector, a second set is held out for testing it. |
| 2 | `judge.py --input extract eval` | Two other open models read every answer and score it from 0 to 100 for how strongly it shows the trait and for how coherent it is. Their scores are averaged. |
| 3 | `extract_vector.py` | Keeps the answer pairs where the instructions worked, takes the model's activations on them, and subtracts the average of one group from the other. The difference is the persona vector. |
| 4 | `reliability.py` | Builds the vector twice, from two separate halves of the questions, and measures how similar the two are. A real trait direction should not depend on which questions were asked. |
| 5 | `separation.py` | Reading test. Measures how well the vector tells trait answers from opposite answers on the held-out questions, compared with random directions. |
| 6 | `steer.py` | Writing test. The model answers the held-out questions with the vector added to its activations at increasing strengths. |
| 7 | `judge.py --input steering` | Scores the steered answers. |
| 8 | `steering_summary.py` | Averages those scores per strength, to show whether the trait rises with strength while the text stays coherent. |
| 9 | `attribute.py` | Splits the vector into the amount written directly by each neuron, each attention layer and the embeddings, checks that the parts add up to the whole, and compares how concentrated it is with random directions. |

The judges are small quantised models, so their scores are approximate. They should be
checked against human ratings before a result rests on them.

## Structure

```
├── persona_neurons/              the library
│   ├── loading.py                load an OLMo 2 checkpoint by training stage; tiny model for tests
│   ├── hooks.py                  read the model's activations and add a vector to them
│   ├── persona.py                prompts, generation, activations, and the vector itself
│   ├── analysis.py               per-layer separation and reliability; the answer-pair filter
│   ├── baselines.py              shuffled-label vectors and random directions to compare against
│   ├── attribution.py            split a vector into per-neuron and per-layer contributions
│   ├── judge.py                  the local judge models, their rubrics and score parsing
│   ├── stats.py                  AUROC, rank correlation, bootstrap confidence intervals
│   ├── pipeline.py               what the scripts share: config, model loading, saved files
│   └── utils.py                  config files, JSON output, provenance of every result
├── scripts/                      one script per pipeline stage (see Usage)
│   └── run_all.sh                runs every stage in order
├── configs/                      one config per trait: model, prompts, sample sizes, layers
├── data/                         trait prompts, questions and rubrics from Chen et al.
└── tests/
    ├── test_persona.py           hooks, generation, activations, statistics
    ├── test_attribution.py       the contributions add up to the vector
    ├── test_analysis_and_pipeline.py   separation and reliability tables, stale-file guards
    └── test_judge_and_filter.py  score parsing, rubrics, answer-pair filtering
```

The files in `data/` come from
[safety-research/persona_vectors](https://github.com/safety-research/persona_vectors)
at commit `b8e0f04`, under the Apache 2.0 licence in `data/LICENSE`. They are renamed
and otherwise unchanged.
