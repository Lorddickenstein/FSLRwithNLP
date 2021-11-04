import tensorflow as tf
from tensorflow import keras
import numpy as np


# Return the sign that corresponds to the index of the predicted class
# def find_match(x):
#     classes = {0: 'A', 1: 'B', 2: 'C', 52: 'Chair', 39: 'Congratulations', 3: 'D', 4: 'E', 5: 'F', 26: 'Fine', 6: 'G',
#                27: 'Gabi', 28: 'Good', 40: 'Great', 7: 'H', 29: 'Hapon', 30: 'He-She', 41: 'Help', 31: 'His-Her',
#                8: 'I', 32: 'I Love You', 33: 'I-Me', 53: 'Invite', 9: 'J', 10: 'K', 11: 'L', 54: 'Late', 55: 'Late-2',
#                12: 'M', 42: 'Meet', 34: 'Mine', 13: 'N', 44: 'Name', 56: 'No', 57: 'No-2', 14: 'O', 45: 'Occupation',
#                58: 'Our', 59: 'Our-2', 15: 'P', 48: 'Pen', 46: 'Pray', 16: 'Q', 17: 'R', 47: 'Rest', 18: 'S', 60: 'Sorry',
#                49: 'Stand', 50: 'Study', 19: 'T', 43: 'Table', 35: 'Tanghali', 61: 'That', 62: 'Them', 51: 'To', 20: 'U',
#                36: 'Umaga', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 37: 'You', 38: 'Your', 25: 'Z'}
#     return classes[x]
def find_match(x):
    classes = {0: 'A', 1: 'B', 74: 'Ball', 75: 'Banana', 76: 'Banana-2', 77: 'Bread',  78: 'Break', 80: 'Break-2',
               81: 'Bring', 82: 'Bring-2', 83: 'Buy', 84: 'Buy-2', 85: 'Bye', 2: 'C', 52: 'Chair', 86: 'Coconut',
               88: 'Coffee', 89: 'Come', 90: 'Come-2', 39: 'Congratulations', 91: 'Cook', 3: 'D', 4: 'E', 79: 'Egg',
               87: 'Egg-2', 96: 'Egg-3', 5: 'F', 26: 'Fine', 92: 'From', 93: 'From-2', 6: 'G', 27: 'Gabi', 94: 'Get',
               95: 'Get-2', 97: 'Go', 98: 'Go-2', 28: 'Good', 40: 'Great', 7: 'H', 29: 'Hapon', 99: 'Happen',
               100: 'Happen-2', 30: 'He-She', 41: 'Help', 31: 'His-Her', 101: 'How', 102: 'How-2', 8: 'I', 32: 'I Love You',
               33: 'I-Me', 103: 'Introduce', 104: 'Introduce-2', 53: 'Invite', 9: 'J', 10: 'K', 11: 'L', 54: 'Late',
               55: 'Late-2', 105: 'Let', 106: 'Let-2', 107: 'Live', 12: 'M', 108: 'Mango', 109: 'Maybe', 42: 'Meet',
               34: 'Mine', 13: 'N', 44: 'Name', 110: 'Nice', 56: 'No', 57: 'No-2', 111: 'Now', 14: 'O', 45: 'Occupation',
               112: 'Office', 113: 'Office-2', 58: 'Our', 59: 'Our-2', 15: 'P', 48: 'Pen', 46: 'Pray', 16: 'Q', 17: 'R',
               47: 'Rest', 18: 'S', 114: 'School', 115: 'Sit', 60: 'Sorry', 49: 'Stand', 116: 'Store', 117: 'Strawberry',
               50: 'Study', 19: 'T', 43: 'Table', 35: 'Tanghali', 118: 'Thank You', 119: 'Thank You-2', 61: 'That', 62: 'Them',
               63: 'This', 51: 'To', 120: 'Today', 121: 'Today-2', 20: 'U', 36: 'Umaga', 21: 'V', 22: 'W', 64: 'We', 65: 'Welcome',
               66: 'Welcome-2', 122: 'What', 67: 'When', 123: 'Where', 124: 'Which', 68: 'Who', 69: 'Who-2', 70: 'Why',
               71: 'Why-2', 125: 'Work', 23: 'X', 24: 'Y', 72: 'Yes', 73: 'Yesterday', 37: 'You', 38: 'Your', 25: 'Z'}
    return classes[x]



# Classify the image using the model
def classify_image(src_img, model):
    predictions = model.predict(src_img)
    top_prediction_indices = np.argsort(predictions)[0, -5:]
    top_predictions = []
    for index in top_prediction_indices:
        prediction = find_match(index)
        score = float("%.02f" % (predictions[0, index] * 100))
        top_predictions.append((prediction, score))
    return predictions, top_predictions
    # score = float("%0.2f" % (prediction[0, class_x] * 100))
    # sign = find_match(class_x)