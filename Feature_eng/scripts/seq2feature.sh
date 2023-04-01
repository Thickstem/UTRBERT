OUTPUT_NAME=Fiveprime_only_restricted

python seq2feature.py --df_path /home/ksuga/UTRBERT/data/ensembl_data/fiveprime_only_restricted_db.csv \
					  --out_dir /home/ksuga/UTRBERT/Feature_eng/data/${OUTPUT_NAME} \
					  --save ${OUTPUT_NAME} \


python feat_mat_python.py --file_prefix ./data/${OUTPUT_NAME}/${OUTPUT_NAME} \

