FEATURE_NAME=Fiveprime_only_restricted_cds15_energy
FEATURE_PATH=./data/${FEATURE_NAME}/${FEATURE_NAME}_final.csv

python randomforest.py --te_df /home/ksuga/UTRBERT/data/df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt \
					   --feature ${FEATURE_PATH} \
					   --save_dir ./data/${FEATURE_NAME} \
					   --cv 5
