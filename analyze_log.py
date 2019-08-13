import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='saved_models')
parser.add_argument('--key', default='tacred')
args = parser.parse_args()

folders = os.listdir(args.folder)
if args.key != 'all':
    files =[os.path.join(args.folder, folder, 'log.txt') for folder in folders if args.key in folder ]
else:
    files =[os.path.join(args.folder, folder, 'log.txt') for folder in folders ]


def get_best_numbers(fname):
    print('Doing', fname)
    with open(fname) as lfh:
        best_dev_f1 = 0
        best_train_f1 = 0
        best_dev_epoch = 0
        best_train_epoch = 0
        epoch = -1
        for line in lfh:
            if 'training epoch' in line:
                epoch += 1
            if 'SELF EVAL' in line:
                if 'dev' in line:
                    dev_f1 = line.strip().split('f1 ')[1]
                    dev_f1 = float(dev_f1)
                    if dev_f1 > best_dev_f1:
                        best_dev_f1 = dev_f1
                        best_dev_epoch = epoch
                elif 'train' in line:
                    train_f1 = line.strip().split('f1 ')[1]
                    train_f1 = float(train_f1)
                    if train_f1 > best_train_f1:
                        best_train_f1 = train_f1
                        best_train_epoch = epoch
        print('Best train F1 {} at EPOCH {}!'.format(best_train_f1, best_train_epoch))
        print('Best dev F1 {} at EPOCH {}!'.format(best_dev_f1, best_dev_epoch))


for f in files:
    get_best_numbers(f)
