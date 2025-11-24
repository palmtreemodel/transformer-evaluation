for VARIABLE in 0 1 2 3 4 5 6 7 8 9
do
    # python evaluator/extract_answers.py -c data/$VARIABLE/test.jsonl -o data/$VARIABLE/answers.jsonl
    python finetune.py --output_dir=./data/$VARIABLE \
    --model_name_or_path=jTrans --do_test_only --evaluate_during_training --seed 123456 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --epoch 2 \
    --train_data_file=./data/$VARIABLE/train.jsonl \
    --eval_data_file=./data/$VARIABLE/valid.jsonl \
    --test_data_file=./data/$VARIABLE/test.jsonl 
    python evaluator/evaluator.py -a data/$VARIABLE/answers.jsonl -p data/$VARIABLE/predictions.jsonl > results/Bert_xl/${VARIABLE}_score.log
done

