#
#	Splitting wav files on smaller chunks
#	This needs to be done for large audio books files
#


info="
INFO: Process large wav files and prepare for LIS model (trim and cut)
USAGE: ./split.sh [INPUT_PATH] [OUTPUT_PATH] [MODE] [TRIM_BEGIN] [TRIM_END] [CUT_SECONDS]

Expecting to have different INPUT_PATH and OUTPUT_PATH logic, based on [MODE].
Modes:
 - 'single'.    INPUT_PATH - path to wav. OUTPUT_PATH - path to folder to save chunks to
 - 'language'.  INPUT_PATH - path to folder with wavs inside. OUTPUT_PATH - path to folder to save chunks to
 - 'packs'.     INPUT_PATH - folder with folders with .wav files. OUTPUT_PATH - base folder for languages chunks

Other params:
 - TRIM_BEGIN (default 0): number of seconds to trim from wav files start
 - TRIM_END (default 0): number of seconds to trim from wav file end
 - CUT_SECONDS (default 10): number of seconds in one chunk of wav file
"


function trim(){
    # Cut begin and end of given wav file
    ## $1 - path to source wav file
    ## $2 - path to save trimmed wav file
    ## $3 - number of seconds to trim from begin
    ## $4 - number of seconds to trim from end

    sox "$1" "$2" trim "$3" -"4" 2> /dev/null
}


function split(){
    # Split given wav file on chunks of specified size
    ## $1 - path to source wav file
    ## $2 - path to folder to place chunks to
    ## $3 - number of seconds in one chunk

    sox "$1" "$2" trim 0 "$3" : newfile : restart 2> /dev/null
}


function process_single(){
    # Process one single wav file
    # First - trim(). Then - split()
    ## $1 - path to source wav file
    ## $2 - path to folder to place source wav chunks
    ## $3 - number of seconds to trim from begin
    ## $4 - number of seconds to trim from end
    ## $5 - number of seconds in one wav chunk

    filename="$(basename "$1")"

    trimmed_file="$2/${filename}.trimmed.wav"
    trim "$1" "${trimmed_file}" "$3" "$4"

    chunk_file="$2/${filename}.chunk.wav"
    split "${trimmed_file}" "${chunk_file}" "$5"

    rm "${trimmed_file}"
}


function process_language(){
    # Process one folder with wav files
    ## $1 - path to language folder with .wav files inside
    ## $2 - path to folder to save wav files
    ## $3 - number of seconds to trim from begin
    ## $4 - number of seconds to trim from end
    ## $5 - number of seconds in one wav chunk

    mkdir "$2" -p

    IFS=$'\n'
    for i in $(find "$1" -maxdepth 1 -name "*.wav" -type f); do
        process_single "$i" "$2" "${3-0}" "${4-0}" "${5-10}"
    done
}


function process_packs(){
    # Run process_language for each folder inside given base folder
    ## $1 - path to folder with languages folders
    ## $2 - path to base folder, where result langiages folders will be created
    ## $3 - number of seconds to trim from begin
    ## $4 - number of seconds to trim from end
    ## $5 - number of seconds in one wav chunk

    IFS=$'\n'
    for i in $(find "$1" -maxdepth 1 -mindepth 1 -type d); do
        folder="$(basename "$i")"
        target_folder="$2/${folder}/"
        mkdir "$target_folder" -p

        echo "$i -> $target_folder"
        process_language "$i" "${target_folder}"
    done
}


if [ $# -le 1 ]; then
    echo "Expected at least 2 args"
    echo "${info}"
    exit 1
fi


source="$(realpath "$1")"
target="$(realpath "$2")"
mode="$3"
trim_begin="${4-0}"
trim_end="${5-0}"
cut_seconds="${6-10}"


if [ "$mode" = "single" ]; then
    process_single "$source" "$target" "$trim_begin" "$trim_end" "$cut_seconds"
elif [ "$mode" = "language" ]; then
    process_language "$source" "$target" "$trim_begin" "$trim_end" "$cut_seconds"
elif [ "$mode" = "packs" ]; then
    process_packs "$source" "$target" "$trim_begin" "$trim_end" "$cut_seconds"
else
    echo "${info}"
    exit 1
fi
