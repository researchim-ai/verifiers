# Verifiers-unsloth: Reinforcement Learning with LLMs in Verifiable Environments

This repo is a fork of https://github.com/willccbb/verifiers with added unsloth examples (for limited resources experiments)

This repository contains a set of tools for reinforcement learning with LLMs in verifiable environments. 

**Note:** This repository in its current state should be viewed as "research code", and is not guaranteed to yield optimal training results. RL is delicate, expect that experimentation will be required. The examples are intended for illustrative purposes of usage patterns rather than stable training recipes. You are encouraged to write your own standalone training scripts, modifying environments/datasets/rewards/configs as needed for your use case.


## Installation

PyPI [coming soon](https://pypi.org/project/verifiers/) once a couple more features are added, just clone it for now and run:
```
conda create -n vf_unsloth python=3.11
conda activate vf_unsloth
git clone https://github.com/researchim-ai/verifiers-unsloth
cd verifiers-unsloth
python verifiers/examples/gsm8k_unsloth.py
```


Ensure your `wandb` and `huggingface-cli` logins are set up (or set `report_to=None` in `training_args`).

## Usage

See `examples` for additional usage examples. 

To create your own multi-step environment, inherit from `MultiStepEnv` and implement:
```python
def get_dataset(self, **kwargs: Any) -> Dataset:
    pass

def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
    pass

def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
    pass

def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
    pass
```

### Launch Commands
Accelerate:
```bash
python verifiers/examples/gsm8k_unsloth.py
```

## Features
- [X] Environments: `SimpleEnv`, `MathEnv`, `DoubleCheckEnv`, `CodeEnv`, `ToolEnv`
- [X] Multi-step execution in `CodeEnv` and `ToolEnv`
- [X] Dataset formatting + XML parsers
- [X] Basic ubrics for math/code correctness + formatting
- [X] Defaults for GRPO, model, tokenizer, etc.

## Roadmap

There are a number of features we're planning to support in the near future:
- [ ] Integrated evals
- [ ] TextArena games
- [ ] LLM judges
- [ ] Claude-generated rubrics
- [ ] A range of other environments (suggestions welcome!)
- [ ] PPO
- [ ] Potential interoperability with other RL libraries (veRL, OpenRLHF, open-instruct, oat, etc.)

Community contributions are appreciated and encouraged!

## Citation

If you use this code in your research, please cite:

```bibtex
@article{brown2025verifiers,
  title={Verifiers: Reinforcement Learning with LLMs in Verifiable Environments},
  author={Brown, William},
  year={2025}
}
```
