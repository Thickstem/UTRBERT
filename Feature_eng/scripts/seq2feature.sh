OUTPUT_NAME=Restricted_cds15


python seq2feature.py --df_path /home/ksuga/UTRBERT/data/ensembl_data/ensembl_restricted_db.csv \
					  --out_dir /home/ksuga/UTRBERT/Feature_eng/data/${OUTPUT_NAME} \
					  --cds_len 15 \
					  --energy \
					  --save ${OUTPUT_NAME} \


python feat_mat_python.py --file_prefix ./data/${OUTPUT_NAME}/${OUTPUT_NAME} \

