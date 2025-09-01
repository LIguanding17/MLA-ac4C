import csv
import random


def parse_txt_file(input_file):
    sequences = []
    with open(input_file, 'r') as file:
        lines = file.readlines()
        for i in range(2, len(lines), 2):
            sequence = lines[i + 1].strip()
            sequences.append(sequence)
    return sequences


def save_to_csv(sequences, output_file, threshold):
    labeled_data = []
    for idx, seq in enumerate(sequences):
        label = 1 if idx < threshold else 0
        labeled_data.append((seq, label))
    random.shuffle(labeled_data)
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Sequence', 'Label'])
        csvwriter.writerows(labeled_data)



# input_file = 'data/iRNA-ac4c-trainset.txt'
# output_file = 'data/train_data.csv'
input_file = 'data/iRNA-ac4c-testset.txt'
output_file = 'data/test_data.csv'


sequences = parse_txt_file(input_file)
# save_to_csv(sequences, output_file, 2206)
save_to_csv(sequences, output_file, 552)

print(f"已成功将数据保存到 {output_file}")
