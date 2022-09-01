# CIFAR10
python train.py --arch resnet32 /
                --dataset cifar10 --data_path './dataset/data_img' /
                --gpu 3 /
                --loss_type 'FeaBal' --batch_size 64 --learning_rate 0.1 --lambda_ 60

# CIFAR100
python train.py --arch resnet32 /
                --dataset cifar100 --data_path './dataset/data_img' /
                --gpu 3 /
                --loss_type 'FeaBal' --batch_size 64 --learning_rate 0.1 --lambda_ 60

# ImageNet
python train.py --arch resnet50 / 
                --dataset imagenet --data_path './dataset/data_txt' --img_path '/home/datasets/imagenet/ILSVRC2012_dataset' / 
                --gpu 3 /
                --loss_type 'FeaBal' --batch_size 512 --learning_rate 0.2 --lambda_ 150


# iNaturalist
python train.py --arch resnet50 / 
                --dataset inat --data_path './dataset/data_txt' --img_path '/home/datasets/iNaturelist2018' / 
                --gpu 3 /
                --loss_type 'FeaBal' --batch_size 512  --learning_rate 0.2 --lambda_ 150

# Places
python train.py --arch resnet152_p / 
                --dataset place365 --data_path './dataset/data_txt' --img_path '/home/datasets/Places365' /
                --gpu 3 /
                --loss_type 'FeaBal' --batch_size 512  --learning_rate 0.2 --lambda_ 150