# SI630 Project: Toxic Spans Detection

**Reference: Data and 2 of the 3 baselines come from https://github.com/ipavlopoulos/toxic_spans. The offensive word list baseline uses the word list from https://github.com/RobertJGabriel/Google-profanity-words**

In addition, this project utilizes computing resources from the [Great Lakes Cluster][https://arc.umich.edu/greatlakes/] and the [Simple Transformers][https://simpletransformers.ai/] library.



## Main files

The `spacy_f1.csv` is the result of the spaCy tagging baseline, and is just for the ease to compare the performance of the baselines. The `preprocess.py` includes all the data processing steps. `sweep.py` contains a not so successful `wandb` sweeping approach. Other steps are included in `SI630_Project.ipynb`.



## Additional Files

`spacy_tagging_revised.py` is a slightly revised version for the original provided one, for the ease to submit jobs on Great Lakes. The original `spacy_tagging.py` is in the `baselines` folder, but the revised one's path is directly under the `toxic_spans` folder (the parent directory of `baselines`.  In addition, to run this file, please follow the tips in the comments in the file.

`run.sh` includes an example batch script used for running a Slurm job.