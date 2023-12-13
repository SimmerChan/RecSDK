file=`find $1 | grep "\.pbtxt$"`

for i in $file
do
python3 dump_tf.py --type=1 --pbtxt=$i
done


