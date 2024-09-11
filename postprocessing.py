def filter_predictions(predictions, threshold=0.8):
    filtered_predictions = []
    for prob, char in predictions:
        if prob >= threshold:
            filtered_predictions.append(char)
    return ''.join(filtered_predictions)
