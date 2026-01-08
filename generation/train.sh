# python main.py --epochs 30 --gen_model vae --model seq
# python main.py --epochs 30 --gen_model vae --model mm_mt
# python main.py --epochs 30 --gen_model vae --model mm_unet

# Ablation study on random_mask_and_shift for mm_mt model
# Model A (original): With random_mask_and_shift (10% substitution probability)
python main.py --epochs 30 --gen_model vae --model mm_mt --use_augmentation False

python main.py --epochs 30 --gen_model vae --model mm_mt --use_augmentation True --mask_prob 0.2

python main.py --epochs 30 --gen_model vae --model mm_mt --use_augmentation True --mask_prob 0.1

python main.py --epochs 30 --gen_model vae --model mm_mt --use_augmentation True --mask_prob 0.05

