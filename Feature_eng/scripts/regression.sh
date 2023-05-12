FEATURE_NAME=my_gencode_db_utr5_only
FEATURE_PATH=./data/${FEATURE_NAME}/${FEATURE_NAME}_final.csv
MODEL=lgb

python Regression.py --te_df /home/ksuga/UTRBERT/data/df_counts_and_len.TE_sorted.pc3.with_annot.txt \
					   --feature ${FEATURE_PATH} \
					   --save_dir ./data/${FEATURE_NAME} \
					   --model ${MODEL}\
					   --cv 5 \
					   --res_file pc3_${MODEL}_cv5
