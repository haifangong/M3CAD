# Ablation study on random_mask_and_shift for mm_mt model
# Each model is sampled 3 times

for i in {1..3}
do
    echo "Starting trial $i for Model B (No Augmentation)"
    python eval.py --weight_path './runs/vae/mm_mt_seed1_no_aug/weights/best.pth' --gen_model vae --model mm_mt --use_augmentation False --condition 1111 --gen_number 1000 --run_name "mm_mt_no_aug_trial$i"

    echo "Starting trial $i for Model A (10% Augmentation)"
    python eval.py --weight_path './runs/vae/mm_mt_seed1_augTrue_prob0.1/weights/best.pth' --gen_model vae --model mm_mt --use_augmentation True --mask_prob 0.1 --condition 1111 --gen_number 1000 --run_name "mm_mt_aug0.1_trial$i"

    echo "Starting trial $i for Model C (20% Augmentation)"
    python eval.py --weight_path './runs/vae/mm_mt_seed1_augTrue_prob0.2/weights/best.pth' --gen_model vae --model mm_mt --use_augmentation True --mask_prob 0.2 --condition 1111 --gen_number 1000 --run_name "mm_mt_aug0.2_trial$i"

    echo "Starting trial $i for Model D (5% Augmentation)"
    python eval.py --weight_path './runs/vae/mm_mt_seed1_augTrue_prob0.05/weights/best.pth' --gen_model vae --model mm_mt --use_augmentation True --mask_prob 0.05 --condition 1111 --gen_number 1000 --run_name "mm_mt_aug0.05_trial$i"
done
