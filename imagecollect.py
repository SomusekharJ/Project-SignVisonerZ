import os
import cv2

# Directory to save the collected data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (letters of the alphabet)
number_of_classes = 26  # For all letters from A to Z
dataset_size = 100

cap = cv2.VideoCapture(0)#only for mac

for j in range(number_of_classes):
    # Create directories for each class if they don't exist
    class_dir = os.path.join(DATA_DIR, chr(65 + j))  # 'A' is 65 in ASCII
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(chr(65 + j)))  # Print class name

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        # Save the frame with the class label
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()