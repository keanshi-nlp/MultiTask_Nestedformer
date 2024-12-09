import os
import random
import json

cla_list = ["Grade_1", "Grade_2_noninvasion","Grade_2_invasion"]

def generate_data_json(label_dir, output_json, num_folds=5):

    num = 1
    data = { "training": [] }
    for cla in cla_list:
        patients = os.listdir(cla)
        for pat in patients:
            pat_path = os.path.join(cla, pat)
            images = []
            label = os.path.join(pat_path, "label_concat.nii.gz")
            images.append(os.path.join(pat_path,f't1.nii.gz'))
            images.append(os.path.join(pat_path,f't2.nii.gz'))
            adc = os.path.join(pat_path, "adc.nii.gz")
            if cla == "Grade_1":
                level = 0
            elif cla == "Grade_2_noninvasion":
                level = 1
            else:
                level = 2


            if images and label:
                data['training'].append({
                    'fold': num % 5,
                    'image': images,
                    'adc': adc,
                    'label': label,
                    'level':level
                })
            num += 1

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)


# Usage
root_directory = r"F:\dachuang\used796_bbox_n4bias\BUS"  # Set this to your root directory path
output_file = "brain_tumor_mri.json"
generate_data_json(root_directory, output_file)
