file=`find /root/cuiguoliang/models/models_all/ | grep "\.prototxt$"`

for i in $file
do
python3 dump_caffe.py --prototxt=$i --optype=ALL

done

