<center><b>## Drop these scripts into L2CS_Net's folder before running ##</b></center>

**Train**

```bash
python _train.py \
--dataset gaze360 \
--gpu 0 \
--num_epochs 25 \
--batch_size 32 \
--alpha 1
```

**Test**

```bash
python test.py \
 --dataset gaze360 \
 --snapshot snapshots/L2CS-gaze360-_1695878254 \
 --evalpath evaluation/L2CS-gaze360  \
 --gpu 0 \
 --arch ResNet18
```

MAE will eventually reach to value around 11 to 12
