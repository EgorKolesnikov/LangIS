# set -x

DATA_PATH="../../data/clean/"


#
#	Move wav files in one language directory into "./all/" folder
#	(just a single-run function, really)
#

function language_move_to_all() {
	echo "$1"
	mkdir "$1/all"
	find "$1" -type f -maxdepth 1 -exec mv {} "$1/all" \;
}

function move_to_all() {
	for folder in $(find $DATA_PATH -maxdepth 1 -mindepth 1 -type d)
	do
		folder=$(realpath $folder)
		move_to_all "$folder"
	done
}


#
#	Create links to files from "./all/" to "./train/" and "./test/" dirs
#

function cleanup_links() {
	if [ -d "$1" ]; then
		find "$1" -type l -delete
	else
		mkdir "$1"
	fi
}


function create_links() {
	cleanup_links "$1/train/"
	cleanup_links "$1/test/"
	cleanup_links "$1/dev/"

	files_list=$(ls $1/all/)
	count=$(ls $1/all/ | wc -l)
	train_border=$(python -c "print int($count * 0.7)")
	test_border=$(python -c "print int($count * 0.9)")
	echo $(($train_border)), $(($test_border-$train_border)), $(($count-$test_border))
	
	IDX=0

	for file in $files_list; do
		if (( $IDX < $train_border )); then
			ln -s "$1/all/$file" "$1/train/$file"
		elif (( $IDX < $test_border )); then
			ln -s "$1/all/$file" "$1/test/$file"
		else
			ln -s "$1/all/$file" "$1/dev/$file"
		fi
		IDX=$((IDX+1))
	done
}


function split_train_test_dev() {
	for folder in $(find $DATA_PATH -maxdepth 1 -mindepth 1 -type d)
	do
		echo "Processing $folder"
		folder=$(realpath $folder)
		create_links "$folder"
	done
}


#
#	Print data folder languages stat
#

function print_stat() {
	printf "\n%-70s %-10s %-10s %-10s %-10s\n\n" FOLDER ALL TRAIN TEST DEV

	TOTAL_ALL=0
	TOTAL_TRAIN=0
	TOTAL_TEST=0
	TOTAL_DEV=0

	for folder in $(find $DATA_PATH -maxdepth 1 -mindepth 1 -type d)
	do
		folder=$(realpath $folder)

		CUR_ALL=$(ls $folder/all/ | wc -l)
		CUR_TRAIN=$(ls $folder/train/ | wc -l)
		CUR_TEST=$(ls $folder/test/ | wc -l)
		CUR_DEV=$(ls $folder/dev/ | wc -l)

		printf "%-70s %-10s %-10s %-10s %-10s\n" $(realpath $folder) $CUR_ALL $CUR_TRAIN $CUR_TEST $CUR_DEV

		TOTAL_ALL=$(($TOTAL_ALL+$CUR_ALL))
		TOTAL_TRAIN=$(($TOTAL_TRAIN+$CUR_TRAIN))
		TOTAL_TEST=$(($TOTAL_TEST+$CUR_TEST))
		TOTAL_DEV=$(($TOTAL_DEV+$CUR_DEV))
	done

	printf "\n%-70s %-10s %-10s %-10s %-10s\n\n" TOTAL $TOTAL_ALL $TOTAL_TRAIN $TOTAL_TEST $TOTAL_DEV
}


#
#	Run
#

if [ "$#" -ne 1 ]; then
    echo "Expecting 1 arg: [move|split|stat]"
else
	if [ "$1" == "move" ]; then
		move_to_all
	elif [ "$1" == "split" ]; then
		split_train_test_dev
	elif [ "$1" == "stat" ]; then
		print_stat
	else
		echo "Unknown value $1. Expecting [move|split|stat]"
	fi
fi
