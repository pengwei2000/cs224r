# DPPO: Direct Preference and Penalization Optimization

Code for the course project of Stanford CS224R.

## Repository organization
```
├── data/                                      # Data loading, preprocessing, and transforms
│   ├── dpo_ultrafeedback_data_pipeline.py     # Ultrafeedback datasets for DPO                 
│   └── sft_data_pipeline.py                   # SmolTalk datasets for SFT 
├── scripts/                                   # Training, evaluation, and utility scripts
│   ├── eval_model.py                          # Model evaluation using llama-3.1-nemotron-70b-reward
│   ├── train_sft.py                           # SFT training scripts
│   ├── train_dpo.py                           # DPO training scripts
│   ├── train_simpo.py                         # SIMPO training scripts
│   └── train_dpo_*.py                         # Training scripts for different penalization strategies
│
└── paper/
    ├── Paper.pdf
    └── poster.pdf
```