### Step1: acquire dataset
Modify the json_path in `split_data.py` and run:
```
cd step1
python split_data.py
```

### Step2: Generate preference data
Set the necessary parameters in `ga.sh`, then:
```
cd step2
bash ga.sh
```

### Step3: Calculate the likelihood of paris (to be used to calculate reward scores in the future)
We develop a method that allows full-param DPO training to run on a 40G machine. We find that the parameters of the reference model remain unchanged during the DPO optimization process, which means that the likelihood of the reference model for the samples is also constant. As a result, we pre-compute these likelihoods and save them in a JSON file. When we later use these samples, we simply call the pre-computed likelihoods to calculate the reward scores.

Based on the above two steps, we obtain the JSON file `/data/step2/dpo_llava6k_0_chatQuestion_corGen_improveAll_ourLoss_noScores.json`. At this point, we need to run `bash run/run.sh --scoring` to calculate the sample likelihoods. We use 8 GPUs, which generates 8 `seek.json` files in the project directory. The `tool.py` in step 3 serves as a tool to merge these seek files.

```
cd ..
bash run/run.sh --scoring
cd step3
python tool.py
```
Because we use deepseek, this `--scoring` needs to be written into run.sh. It cannot be passed in using the command line. It is written `bash run/run.sh --scoring` here for the convenience of explanation.

Finally you will get the file `./data/step3/dpo_llava6k_0_chatQuestion_corGen_improveAll_ourLoss-scores.json`, then we start DPO training.
