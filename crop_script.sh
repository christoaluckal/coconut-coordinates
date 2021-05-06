read -e -p "Original:" original
read -e -p "Width:" width
read -e -p "Height:" height
read -p "Output:" output
convert $original -gravity center -crop ${width}x$height+0+0 +repage $output