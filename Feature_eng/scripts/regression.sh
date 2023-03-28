FEATURE_NAME=Featurematrix_restricted
FEATURE_PATH=./data/${FEATURE_NAME}/${FEATURE_NAME}_final.csv

python randomforest.py --te_df /home/ksuga/UTRBERT/data/df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt \
					   --feature ${FEATURE_PATH} \
					   --save_dir ./data/${FEARUE_NAME}
