file=`find $1 | grep "\.pb$"`

for i in $file
do
python3 dump_tf.py --pb=$i 
done


