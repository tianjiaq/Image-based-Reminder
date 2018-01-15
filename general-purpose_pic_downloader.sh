#!/bin/bash

 

URL_ARRAY=(
'https://www.colourbox.com/image/banana-image-1660845'
)

 

NAME_ARRAY=(
'test_banana'

)

 

ELEMENTS=${#URL_ARRAY[@]}

for (( i=0;i<ELEMENTS;i++)); do

 echo "${URL_ARRAY[${i}]}"

 echo "saved as ${NAME_ARRAY[${i}]}"$i".jpg"

 curl "${URL_ARRAY[${i}]}"$i".jpg" -o ./"${NAME_ARRAY[${i}]}$i.jpg"

 sleep 1

 

done
