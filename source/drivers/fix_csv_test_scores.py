import os
import os.path

def check_scores(fn):
    scores_file = open(fn, mode='r')
    lines = scores_file.readlines()[1:]
    lines_dict = {}

    for line in lines:
        cols = line.split()
        assert(len(cols) == 2)
        lines_dict[int(cols[0])] = float(cols[1])

    if 51 not in lines_dict:
        return

    print("Fixing test scores for file `{}`.".format(fn))
    scores_file.close()
    scores_file = open(fn, mode='w+')

    print("Epoch\tScore", file=scores_file)
    for epoch, acc in sorted(lines_dict.items()):
        print("{}\t{}".format(epoch - 1, acc), file=scores_file)

def main():
    test_scores_fn = "test_scores.csv"
    model_dir = "models/svhn_5x5_batch_100"

    for root, dirs, files in os.walk(model_dir):
        for d in dirs:
            test_scores_fp = os.path.join(model_dir, d, test_scores_fn)
            if os.path.isfile(test_scores_fp):
                check_scores(test_scores_fp)

if __name__ == "__main__":
    main()
