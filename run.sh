c=1
while true ; # 10000 loop
do
    echo "Python script called $c times"
    KERAS_BACKEND=tensorflow python model.py
    c=$(($c + 1))
done