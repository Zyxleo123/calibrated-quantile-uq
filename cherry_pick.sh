python -u main_cherry_pick.py --data protein --loss calipso --num_ep 100 --nl 8 --hs 256 --seed 1 --gpu 2
python -u main_cherry_pick.py --data protein --loss maqr --num_ep 25 --nl 8 --hs 256 --seed 1 --gpu 2
python -u main_cherry_pick.py --data protein --loss batch_qr --num_ep 1000 --nl 8 --hs 256 --seed 1 --gpu 2
python -u main_cherry_pick.py --data protein --loss batch_int --num_ep 1000 --nl 8 --hs 256 --seed 1 --gpu 2
python -u main_cherry_pick.py --data protein --loss batch_cal --num_ep 1000 --nl 8 --hs 256 --seed 1 --gpu 2
