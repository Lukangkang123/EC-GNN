/usr/bin/python training.py --dataset Cornell --lr 0.05 --dprate 0.5 --alpha 0.9 --net GPRGNN --device 0 --beta 0.77
/usr/bin/python training.py --dataset Texas --lr 0.05 --dprate 0.5 --alpha 1.0 --net GPRGNN --device 0 --beta 0.91
/usr/bin/python training.py --dataset Actor --lr 0.01 --dprate 0.9 --alpha 1.0 --weight_decay 0.0 --net GPRGNN --device 0 --beta 0.90
/usr/bin/python training.py --dataset Chameleon --lr 0.05 --dprate 0.7 --alpha 1.0 --weight_decay 0.0 --net GPRGNN --device 0 --beta 0.09
/usr/bin/python training.py --dataset Squirrel --lr 0.05 --dprate 0.7 --alpha 0.0 --weight_decay 0.0 --net GPRGNN --device 0 --beta 0.23
/usr/bin/python training.py --dataset Cora --alpha 0.8 --dprate 0.8  --lr 0.05  --net GPRGNN  --device 0 --beta 0.9
/usr/bin/python training.py --dataset Citeseer --alpha 0 --dprate 0.3   --lr 0.01 --net GPRGNN --device 0 --beta 0.25
/usr/bin/python training.py --dataset Pubmed  --alpha 0.9 --dprate  0.7  --lr 0.05 --net GPRGNN --device 0 --beta 0.75
/usr/bin/python training.py --dataset Computers  --alpha 0.6 --dprate  0.4  --lr  0.05 --net GPRGNN --device 0  --beta 0.15
/usr/bin/python training.py --dataset Photo  --alpha 0.7 --dprate  0.2  --lr 0.05 --net GPRGNN --device 1 --beta 0.05