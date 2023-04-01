FEATURE_NAME=Fiveprime_only_restricted_cds15
FEATURE_PATH=./data/${FEATURE_NAME}/${FEATURE_NAME}_final.csv

python Regression.py --te_df /home/ksuga/UTRBERT/data/df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt \
					   --feature ${FEATURE_PATH} \
					   --save_dir ./data/${FEATURE_NAME} \
					   --model lgb \
					   --cv 5 \
					   --res_file results_lgb_5fold
