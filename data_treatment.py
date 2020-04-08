import argparse
import random

import matplotlib.pyplot as plt


DATASET_DIR = "dataset/"

ORIGIN_FILES = [f"{DATASET_DIR}sign_mnist_test.csv", f"{DATASET_DIR}sign_mnist_train.csv"]

TRAIN_DEST_FILE = f"{DATASET_DIR}merged_train_set.csv"
CV_DEST_FILE = f"{DATASET_DIR}merged_cv_set.csv"
TEST_DEST_FILE = f"{DATASET_DIR}merged_test_set.csv"
SET_DISTRIBUTION_FILE = f"{DATASET_DIR}sets_distribution.png"
TRAIN_GRAPH_TITLE = f"Train Data Distribution"
CV_GRAPH_TITLE = f"Cross Validation Data Distribution"
TEST_GRAPH_TITLE = f"Test Data Distribution"

SPLIT_DIST = [60, 20, 20]  # distribution to each dataset: [train, cv, test]


def split_data():
	all_data = []
	for file_name in ORIGIN_FILES:
		with open(file_name, "r") as file:
			all_data += file.readlines()[1:]
	random.shuffle(all_data)

	all_data_size = len(all_data)
	end_id_each = []
	sum = 0
	for dist in SPLIT_DIST:
		sum = int(sum + all_data_size * dist / 100)
		end_id_each.append(min(sum, all_data_size))

	all_dest_files = [TRAIN_DEST_FILE, CV_DEST_FILE, TEST_DEST_FILE]
	start_id = 0
	for id in range(len(end_id_each)):
		with open(all_dest_files[id], "w") as file:
			for data in all_data[start_id:end_id_each[id]]:
				file.write(data)
		start_id = end_id_each[id]


def generate_graphs():
	fig, axs = plt.subplots(1, 3, figsize=(20, 6.25))

	current_ax_id = 0
	for files in [(TRAIN_DEST_FILE, TRAIN_GRAPH_TITLE),
	                  (CV_DEST_FILE, CV_GRAPH_TITLE),
	                  (TEST_DEST_FILE, TEST_GRAPH_TITLE)]:
		with open(files[0], "r") as file:
			labels = [int(line.split(",")[0]) for line in file.readlines()]

		current_ax: plt.axes.Axes = axs[current_ax_id]
		current_ax.set_title(files[1])
		current_ax.set_xlabel("Label")
		current_ax.set_ylabel("Number of Examples")
		current_ax.hist(labels, bins=range(0, 26), ec='black')

		current_ax_id += 1

	plt.savefig(SET_DISTRIBUTION_FILE)


def main(args):
	if args.split:
		split_data()
	if args.generate_graphs:
		generate_graphs()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--split", action="store_true")
	parser.add_argument("--generate_graphs", action="store_true")

	args = parser.parse_args()
	main(args)
