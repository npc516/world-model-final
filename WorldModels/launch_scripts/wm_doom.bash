CONFIG_PATH=configs/doom.config
for i in `seq 1 4`;
do
  echo worker $i
  CUDA_VISIBLE_DEVICES=-1 python3 extract.py -c $CONFIG_PATH &
  sleep 1.0
done
# can't just use 'wait' b/c doom env leaves processes open
N_RUNNING=100 # some number > 0
while [ $N_RUNNING -gt 1 ] # 1 b/c the grep search will also show up in the count
do
  N_RUNNING=$(ps aux | grep -c extract)
  sleep 60.0 # wait a minute
  echo $N_RUNNING
done
pkill -9 -f vizdoom
CUDA_VISIBLE_DEVICES=0 python3 vae_train.py -c $CONFIG_PATH
CUDA_VISIBLE_DEVICES=0 python3 series.py -c $CONFIG_PATH
CUDA_VISIBLE_DEVICES=0 python3 rnn_train.py -c $CONFIG_PATH
CUDA_VISIBLE_DEVICES=-1 python3 train.py -c $CONFIG_PATH
