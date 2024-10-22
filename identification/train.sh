#CUDA_VISIBLE_DEVICES=0 python main.py --task toxin --model voxel-tr --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task toxin --model seq --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task toxin --model mm --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task toxin --model mm --loss mlce

#CUDA_VISIBLE_DEVICES=0 python main.py --task anti --model voxel-tr --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task anti --model seq --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task anti --model mm --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task anti --model mm --loss mlce
#CUDA_VISIBLE_DEVICES=0 python main.py --task anti --model mmf --loss mlce

#CUDA_VISIBLE_DEVICES=0 python main.py --task mechanism --model voxel-tr --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task mechanism --model seq --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task mechanism --model mm --loss ce
CUDA_VISIBLE_DEVICES=0 python main.py --task mechanism --model mm --loss mlce

#CUDA_VISIBLE_DEVICES=0 python main.py --task mic --model mm --loss mse
