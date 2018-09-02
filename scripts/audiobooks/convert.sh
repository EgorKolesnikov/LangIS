#
#   Manage mp3-to-wav convertion
#   This script can do:
#    - convert single mp3
#    - convert all mp3s in given folders
#    - convert all mp3s in all folders inside base folder
#


info="
INFO: Manage mp3-to-wav convertion
USAGE: ./convert.sh [INPUT_PATH] [OUTPUT_PATH] [MODE]

Expecting to have different INPUT_PATH and OUTPUT_PATH logic, based on [MODE].
Modes:
 - 'single'.    INPUT_PATH, OUTPUT_PATH - full paths to input (.mp3) and output (.wav) files
 - 'language'.  INPUT_PATH, OUTPUT_PATH - full paths to base folder with .mp3 and .wav files
 - 'packs'.     INPUT_PATH - folder with folders with .mp3 files. OUTPUT_PATH - base target .wav files folder
"


function convert_single(){
    # Convert mp3 file to wav format
    ## $1 - full path to input mp3 file
    ## $2 - full path to output wav file

    ffmpeg -i "$1" "$2" > /dev/null 2> /dev/null
}


function convert_language(){
    # Convert each mp3 file in given directory to wav format
    # and save resut in given directory
    ## $1 - path to folder with mp3 files
    ## $2 - path to target folder (for wav files)

    IFS=$'\n'
    for i in $(find "$1" -maxdepth 1 -name "*.mp3" -type f); do
        filename=$(basename $i)
        target_path="$2/${filename}.wav"
        convert_single "$i" "$target_path"
    done
}

function convert_packs(){
    # Run convert_folder() for all folders in given base folder
    # Create same folders inside target folder
    ## $1 - path to folders, which contain only folders with mp3 files
    ## $2 - path to base folder for target folders and files
    
    IFS=$'\n'
    for i in $(find "$1" -maxdepth 1 -mindepth 1 -type d); do
        folder="$(basename "$i")"
        target_folder="$2/${folder}/"

        if [ ! -d "$target_folder" ]; then
            mkdir "$target_folder" -p

            echo "$i -> $target_folder"
            convert_language "$i" "${target_folder}"
        else
            echo "Skip $target_folder"
        fi
    done
}


if [ $# != 3 ]; then
    echo "${info}"
    exit 1
fi


source="$(realpath "$1")"
target="$(realpath "$2")"


if [ "$3" = "single" ]; then
    convert_single "$source" "$target"
elif [ "$3" = "language" ]; then
    convert_language "$source" "$target"
elif [ "$3" = "packs" ]; then
    convert_packs "$source" "$target"
else
    echo "${info}"
    exit 1
fi
