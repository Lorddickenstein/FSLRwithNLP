# Extract all frames from Ella's video
import cv2
import numpy
import os

# path = 'D:\Documents\Thesis\\Upload\Ella'
# class_obj = ['Invite', 'No', 'Our', 'Sorry', 'Them', 'Welcome', 'When', 'Who', 'Why', 'Yes', 'Bye',
#              'Go', 'Introduce', 'Live', 'Nice', 'Now', 'Allow', 'Break', 'Bring', 'Buy', 'Come', 'From',
#              'Get', 'Happen', 'Lived', 'Make', 'Maybe', 'Because', 'Thank you', 'Sit', 'What', 'Which',
#              'Work', 'Where', 'Let', 'How']
# class_obj = ['Our']

path = 'D:\Pictures\Camera Roll\\New Dataset'
# class_obj = ['Ago', 'Ball', 'Banana', 'Bread', 'Coconut', 'Coffee', 'Eroplano', 'Late', 'Mango', 'Office',
#              'School', 'Store', 'Strawberry', 'That', 'This', 'Today', 'We', 'Year', 'Yesterday']
class_obj = ['Chair', ]

# path = 'D:\Pictures\Camera Roll\Joshua'
# class_obj = ['Allow', 'Break', 'Bring', 'Buy', 'Come', 'From', 'Get', 'Happen', 'Let', 'Make', 'Maybe', 'Thank you',
#              'Sorry', 'Sit', 'What', 'Which', 'Work', 'Their', 'Them', 'We', 'Welcome', 'When', 'Where', 'Who', 'Why',
#              'Yes', 'Yes']

# path = 'D:\Pictures\Camera Roll\Joshua\Static\Static Double'
# class_obj = ['Congratulations','Great', 'Meet', 'Name', 'Night', 'Occupation', 'Pray', 'Rest', 'Stand', 'Study', 'To']

# path = 'D:\Pictures\Camera Roll\Joshua\Static\Static Single'
# class_obj = ['Fine', 'Gabi', 'Good', 'Hapon', 'He-She', 'His-Her', 'I Love You', 'I-Me', 'Mine', 'Tanghali', 'Umaga', 'You', 'Your']

for obj in class_obj:
    path_class = os.path.join(path, obj)
    j = 0
    for item in os.listdir(path_class):
        video_path = os.path.join(path_class, item)
        cap = cv2.VideoCapture(video_path)
        i = 0
        k = 0
        j += 1
        dirs = os.path.join('D:\Pictures\Camera Roll\Temp2', obj)
        if os.path.isdir(dirs) is False:
            os.makedirs(dirs)
        while cap.isOpened():
            _, frame = cap.read()
            if not _:
                break
            if i % 5 == 0:
                name = obj + '_' + str(j) + '_' + str(k) + '.jpg'
                save_path = os.path.join(dirs, name)
                print(name)
                cv2.imwrite(save_path, frame)
                k += 1
            i += 1
        cap.release()
cv2.destroyAllWindows()

extracted_path = 'D:\Pictures\Camera Roll\Temp2'
for obj in class_obj:
    class_path = os.path.join(extracted_path, obj)
    print(obj, len(os.listdir(class_path)))