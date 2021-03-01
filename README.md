# gypsum

## Data Link: https://drive.google.com/file/d/1hQWQE6qm-qNGYKEPMoVMepMEZ72nXJL3/view?usp=sharing

### Data Directory Structure:
├── data
│   ├── java
│   │   ├── dev
│   │   │   ├── code.original
│   │   │   ├── code.original_subtoken
│   │   │   ├── ggnn.json
│   │   │   └── javadoc.original
│   │   ├── dev_remove
│   │   │   ├── code.original
│   │   │   ├── code.original_subtoken
│   │   │   ├── ids.txt
│   │   │   └── javadoc.original
│   │   ├── othter_models
│   │   │   ├── code2seq
│   │   │   └── rl_cs
│   │   ├── test
│   │   │   ├── code.original
│   │   │   ├── code.original_subtoken
│   │   │   ├── ggnn.json
│   │   │   └── javadoc.original
│   │   ├── test_remove
│   │   │   ├── code.original
│   │   │   ├── code.original_subtoken
│   │   │   ├── ids.txt
│   │   │   └── javadoc.original
│   │   └── train
│   │       ├── code.original
│   │       ├── code.original_subtoken
│   │       ├── ggnn.json
│   │       └── javadoc.original
│   └── python
│       ├── code2seq
│       │   ├── dev.raw.txt.json
│       │   ├── python.dict.c2s
│       │   ├── python.test.c2s
│       │   ├── python.train.c2s
│       │   ├── python.val.c2s
│       │   ├── test.raw.txt.json
│       │   └── train.raw.txt.json
│       ├── dev
│       │   ├── ast.original
│       │   ├── code.original
│       │   ├── code.original_subtoken
│       │   ├── ggnn.json
│       │   ├── javadoc.new
│       │   └── javadoc.original
│       ├── dev_remove
│       │   ├── code.original
│       │   ├── code.original_subtoken
│       │   ├── ids.txt
│       │   └── javadoc.original
│       ├── test
│       │   ├── ast.original
│       │   ├── code.original
│       │   ├── code.original_subtoken
│       │   ├── ggnn.json
│       │   ├── javadoc.new
│       │   └── javadoc.original
│       ├── test_remove
│       │   ├── code.original
│       │   ├── code.original_subtoken
│       │   ├── ids.txt
│       │   └── javadoc.original
│       └── train
│           ├── ast.original
│           ├── code.original
│           ├── code.original_subtoken
│           ├── ggnn.json
│           ├── javadoc.new
│           └── javadoc.original
├──


## How to run: 
> python -u -u bert_nmt/train.py config/general_config.yml config/xxxx.yml
