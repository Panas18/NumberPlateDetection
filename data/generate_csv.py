import os

label_files = os.listdir('labels')
train_files, test_files = label_files[:2450], label_files[2450:]

for i in range(len(train_files)):
    train_pre, train_ext = os.path.splitext(train_files[i])
    train_label = train_files[i]
    train_img = train_pre + '.jpg'
    with open('train.csv', 'a') as f:
        f.write(f"{train_img},{train_label}\n")

    try:
        test_pre, test_ext = os.path.splitext(test_files[i])
        test_label = test_files[i]
        test_img = test_pre + '.jpg'
        with open('test.csv', 'a') as f:
            f.write(f"{test_img},{test_label}\n")
    except IndexError:
        pass
