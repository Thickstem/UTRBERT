IMAGE_NAME=utr_performer
NOW=`date +"%Y%m%d%I%M%S"`
SHM_SIZE=4g


build:
	docker build -t ${IMAGE_NAME} .

start:
	docker run --gpus all --shm-size=${SHM_SIZE} --rm -it --name ${IMAGE_NAME}-${USER}-${NOW} \
						-v ${PWD}:/opt/jlp_maker \
						-v ${DATA}:/data \
						${IMAGE_NAME} sh -c "cd /opt/jlp_maker && bash"



