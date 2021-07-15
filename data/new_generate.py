import os
images_file = os.listdir('new_img')

for image in images_file:
    test_image = image
    pre, ext = os.path.splitext(test_image)
    test_label = pre + '.txt'
    with open('new_test.csv', 'a') as f:
        f.write(f"{test_image},{test_label}\n")
print('completed')





    
