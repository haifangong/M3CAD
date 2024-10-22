import argparse
import time
import torch

from dataset import idx2ama, calculate_property
from model import SEQVAE, MMVAE, SEQWAE, MMWAE


def calculate_mean_std(property_list, property_name):
    if not property_list:
        print(f'{property_name:<23s}: No data to calculate statistics.')
        return None, None
    mean = sum(property_list) / len(property_list)
    variance = sum((x - mean) ** 2 for x in property_list) / len(property_list)
    std = variance ** 0.5
    print(f'{property_name:<23s}: Mean = {mean:.4f}, Std = {std:.4f}')
    return mean, std


def generate(args):
    path = 'results/' + args.gen_model + '-' + args.model + str(args.condition)
    gen = open(path + '_seq.fasta', 'w')
    gen_info = open(path + '_properties.csv', 'w')

    if args.gen_model == 'vae':
        if args.model == 'seq':
            model = SEQVAE(args.classes, sequence_length=args.seq_length).cuda()
        elif args.model == 'mm_unet':
            model = MMVAE(args.classes, sequence_length=args.seq_length, use_encoder_feature=True).cuda()
        elif args.model == 'mm_mt':
            model = MMVAE(args.classes, sequence_length=args.seq_length, use_encoder_feature=False).cuda()
    elif args.gen_model == 'wae':
        if args.model == 'seq':
            model = SEQWAE(args.classes, sequence_length=args.seq_length).cuda()
        elif args.model == 'mm_unet':
            model = MMWAE(args.classes, sequence_length=args.seq_length, use_encoder_feature=True).cuda()
        elif args.model == 'mm_mt':
            model = MMWAE(args.classes, sequence_length=args.seq_length, use_encoder_feature=False).cuda()

    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    # Process condition condition from args
    condition_list = [float(bit) for bit in args.condition]
    condition = torch.tensor(condition_list).float().unsqueeze(0).cuda()

    property_names = [
        'Gravy Score',
        'Aliphatic Index',
        'Aromaticity',
        'Instability Index',
        'Alpha Helix Fraction',
        'Beta Helix Fraction',
        'Turn Fraction',
        'Charge at pH 7',
        'Isoelectric Point',
        'Charge Density',
        'Average Hydrophobicity',
        'Hydrophobic Moment'
    ]

    gen_info.write('Count,Sequence,' + ','.join(property_names) + '\n')

    # Initialize counters and caches
    count = 0
    seq_cache = set()  # Using a set for faster lookup

    # Initialize dictionaries to store properties
    property_lists = {prop_name: [] for prop_name in property_names}

    # Start timer
    start = time.time()

    while count < args.gen_number:
        if count % 1000 == 0 and count != 0:
            elapsed_time = time.time() - start
            print(f'Generated {count} sequences. Elapsed Time: {elapsed_time:.2f} seconds.')

        # Generate random latent vector z
        z = torch.randn([condition.size(0), 32]).cuda()

        # Perform inference to generate sequence indices and their string representations
        recon_seq_indices, random_gen, ori_seq = model.inference(z, condition)
        classify_result = model.cls_head(ori_seq.float())[0]
        if classify_result.any() < 0.75:
            continue

        # Iterate over each sequence in the batch
        for batch_idx in range(recon_seq_indices.size(0)):
            pred = recon_seq_indices[batch_idx]  # Shape: (sequence_length,)
            # Handle sequence truncation
            if 21 in pred:
                zero_indices = torch.where(pred == 21)[0][0].item()
            elif 0 in pred:
                zero_indices = torch.where(pred == 0)[0][0].item()
            else:
                zero_indices = -1

            valid_indices = pred[:zero_indices] if zero_indices != -1 else pred

            # Convert indices to amino acids
            pred_seq = [idx2ama[idx.item()] for idx in valid_indices]
            seq_str = ''.join(pred_seq)

            # Check if the sequence is unique
            if seq_str not in seq_cache:
                print(f'Count: {count} | Sequence: {seq_str}')

                # Calculate properties
                properties = calculate_property(seq_str)

                # Append properties to respective lists using a loop
                for prop_name, prop_value in zip(property_names, properties):
                    property_lists[prop_name].append(prop_value)

                # Add sequence to cache
                seq_cache.add(seq_str)

                # Write to FASTA file
                gen.write(f'>{count}\n{seq_str}\n')

                # Write to info CSV file
                properties_str = ','.join(map(str, properties))
                gen_info.write(f'{count},{seq_str},{properties_str}\n')

                # Increment count
                count += 1
                if count >= args.gen_number:
                    break  # Break out if we've reached the desired count

    # After generation loop, calculate mean and std for each property
    print('\n--- Property Statistics of Generated Sequences ---')

    with open(path+'_property_stats.txt', 'w') as stats_file:
        for prop_name in property_names:
            prop_values = property_lists[prop_name]
            mean, std = calculate_mean_std(prop_values, prop_name)
            if mean is not None and std is not None:
                stats_file.write(f'{prop_name}: Mean = {mean:.4f}, Std = {std:.4f}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GOOD')
    parser.add_argument('--gen_model', type=str, default='vae',
                        help='vae or wae')
    parser.add_argument('--seq_length', type=int, default=20,
                        help="Max length of peptides")
    parser.add_argument('--model', type=str, default='mm_unet',
                        help='seq, mm_unet, or mm_mt')
    parser.add_argument('--weight_path', type=str, default='/path/to/weights.pth',
                        help='Path to the model weights file')
    parser.add_argument('--classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--condition', type=str, default='1111',
                        help='Condition as a binary string, e.g., 1111, 1101')
    parser.add_argument('--gen_number', type=int, default=1000,
                        help='Generated sample count')

    args = parser.parse_args()
    generate(args)