MODEL=path2LLaVa-v1.5-7b
IMG_PATH=path2llava-data
IMG_DATA=/step1/llava6k_0.json
QUES_DATA=chat_corGen_question_0.jsonl
OUTPUT_DATA=dpo_llava6k_0_chatQuestion_corGen_improveAll_ourLoss

torchrun --nproc_per_node 8 --master_port 29503 generate_question.py \
    --model-path $MODEL \
    --image_file_list $IMG_DATA \
    --image_path $IMG_PATH \
    --save_dir ./ \
    --res_file $QUES_DATA \

torchrun --nproc_per_node 8 --master_port 29503 generate_with_aug.py \
    --model-path $MODEL \
    --image_file_list $QUES_DATA \
    --image_path $IMG_PATH \
    --save_dir ./ \
    --res_file ${OUTPUT_DATA}_ans.jsonl \

torchrun --nproc_per_node 8 --master_port 29503 generate_with_aug.py \
    --model-path $MODEL \
    --image_file_list $QUES_DATA \
    --image_path $IMG_PATH \
    --save_dir ./ \
    --res_file ${OUTPUT_DATA}_dnAns.jsonl \
    --augmentation "diffusion" \
    --noise_step 600 \

torchrun --nproc_per_node 8 --master_port 29503 generate_improve.py \
    --model-path $MODEL \
    --image_file_list ${OUTPUT_DATA}_ans.jsonl \
    --image_path $IMG_PATH \
    --save_dir ./ \
    --res_file ${OUTPUT_DATA}_improve.jsonl \

python make_pair.py --filename $OUTPUT_DATA


