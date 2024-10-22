python eval.py --weight_path './runs/vae/seq1/weights/best.pth' --gen_model vae --model seq --condition 1111
python eval.py --weight_path './runs/vae/mm_mt1/weights/best.pth' --gen_model vae --model mm_mt --condition 1111
python eval.py --weight_path './runs/vae/mm_unet1/weights/best.pth' --gen_model vae --model mm_unet --condition 1111

