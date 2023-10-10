model_name_or_path=roberta-base

dataset_name=EUR-Lex
subset=80%*1

CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_name_or_path $model_name_or_path \
  --dataset_name $dataset_name-$subset \
  --stratified_sampling true \
  --with_example true \
  --momentum_rate 0.3 \
  --centroid true \
  --output_dir output/$dataset_name/$subset \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 24 \
  --per_device_eval_batch_size 24 \
  --num_train_epochs 10 \
  --weight_decay 0.01 \
  --save_total_limit 1 \
  --load_best_model_at_end true \
  --metric_for_best_model micro-F1@1 \
  --log_on_each_node false \
  --logging_steps 500 \
  --save_steps 500