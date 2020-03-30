import random

DATASET_DIR = "dataset/"

ORIGIN_FILES = [f"{DATASET_DIR}sign_mnist_test.csv", f"{DATASET_DIR}sign_mnist_train.csv"]

TRAIN_DEST_FILE = f"{DATASET_DIR}merged_train_set.csv"
CV_DEST_FILE = f"{DATASET_DIR}merged_cv_set.csv"
TEST_DEST_FILE = f"{DATASET_DIR}merged_test_set.csv"

SPLIT_DIST = [60, 20, 20]        # distribution to each dataset: [train, cv, test]

all_data = []
for file_name in ORIGIN_FILES:
   with open(file_name, "r") as file:
      all_data += file.readlines()[1:]
random.shuffle(all_data)

all_data_size = len(all_data)
end_id_each = []
sum = 0
for dist in SPLIT_DIST:
   sum = int(sum + all_data_size*dist/100)
   end_id_each.append(min(sum, all_data_size))

all_dest_files = [TRAIN_DEST_FILE, CV_DEST_FILE, TEST_DEST_FILE]
start_id = 0
for id in range(len(end_id_each)):
   with open(all_dest_files[id], "w") as file:
      for data in all_data[start_id:end_id_each[id]]:
         file.write(data)
   start_id = end_id_each[id]
