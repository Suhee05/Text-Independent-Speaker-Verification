#!/bin/bash

if [ $# -ne 2 ]
then
    echo "usage :$0 [input : input directory(including .wav)]
$1 [outdir : output directory (excluding file name)]"
    exit 0;
fi

in_dir=$1
out_dir=$2
# (${STR//DELIMITER/ })
# in STR, substitute DELIMITER with SUBSTITUTE and ' ' (a single space)
# then interprets into the space-delimited string as an array
dir_arr=(${in_dir//// })

new_file=${dir_arr[-3]}"_"${dir_arr[-2]}"_"${dir_arr[-1]}

cp $in_dir $out_dir/$new_file
