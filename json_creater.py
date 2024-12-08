import os
import random
import json

def generate_data_json(label_dir, output_json, num_folds=5):

    data = { "training": [] }
    
    for d in os.listdir(label_dir):
        num = int(d.split('.')[0])
        images = []
        label = f'label/{num}.nii.gz'
        images.append(f't1/data/{num}.nii.gz')
        images.append(f't2/data/{num}.nii.gz')

        if images and label:
            data['training'].append({
                'fold': num % 5,
                'image': images,
                'label': label
            })
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)

# Usage
root_directory = "/mnt/brain_tumor/label"  # Set this to your root directory path
output_file = "brain_tumor_mri.json"
generate_data_json(root_directory, output_file)
