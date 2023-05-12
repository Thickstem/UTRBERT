OUTPUT_NAME=my_gencode_db_utr5_only

python seq2feature.py --df_path /home/ksuga/UTRBERT/data/gencode_v17_utr5_only.csv \
					  --out_dir /home/ksuga/UTRBERT/Feature_eng/data/${OUTPUT_NAME} \
					  --head_cds_len 15 \
					  --energy \
					  --save ${OUTPUT_NAME} \


python feat_mat_python.py --file_prefix ./data/${OUTPUT_NAME}/${OUTPUT_NAME} \

