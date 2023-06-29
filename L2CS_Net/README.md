<center><b>## Drop these scripts into L2CS_Net's folder before running ##</b></center>

**Train**

```bash
python _train.py \
--dataset gaze360 \
--gpu 0 \
--num_epochs 50 \
--batch_size 16 \
--lr 0.00001 \
--alpha 1
```

MSE Loss will eventually reach to value around 0.05

**Test**

```bash
python _test.py \
 --dataset gaze360 \
 --snapshot snapshots/L2CS-gaze360-_230627032327 \
 --evalpath evaluation/L2CS-gaze360  \
 --gpu 0
```

MAE will eventually reach to value around 11

**Demo**

```bash
_demo_video.py --snapshot snapshots/gazeBackend.pkl --input examples/Videos/0.mp4 --output examples/Videos/0_res18_out.mp4
```

New resnet18 has similar performance as original model, but it still struggles in processing videos that are too blur.
