ffprobe -v error -select_streams v:0 \
-print_format json -show_format -show_streams \
-show_entries stream=codec_name,width,height,bit_rate \
-show_entries format=duration,filename,nb_streams,size \
rain.mp4 > info.json