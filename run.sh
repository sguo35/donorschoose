c=1
while true ; # 10000 loop
do
    echo "Python script called $c times"
    KERAS_BACKEND=tensorflow python train.py
    c=$(($c + 1))
done