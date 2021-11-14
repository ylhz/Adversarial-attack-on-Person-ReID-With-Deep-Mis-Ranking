
# merket1501
python train_grad.py   --targetmodel='cam'   --dataset='market1501'  --mode='test'   --loss='xent_htri'   --ak_type=-1   --temperature=-1   --use_SSIM=2   --epoch=40 --eps 10 --iter_num 50 --gpu 0
# dukemtmcreid
python train_grad.py   --targetmodel='cam'   --dataset='dukemtmcreid'   --mode='test'   --loss='xent_htri'   --ak_type=-1   --temperature=-1   --use_SSIM=2   --epoch=40 --eps 10 --iter_num 50 --gpu 0