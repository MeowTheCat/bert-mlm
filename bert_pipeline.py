def get_data_and_label_from_files(file):
    data_text = []
    label_text = []
    with open(file) as f:
        for line in f:
            line = line.replace('。', '，')
            segments = line.split('，')
            data_text.append(segments[0] + '，')
            label_text.append(segments[1])
            data_text.append(segments[0] + '，' + segments[1] + '，')
            label_text.append(segments[2])
            data_text.append(segments[0] + '，' + segments[1] + '，' + segments[2] + '，')
            label_text.append(segments[3])
    return data_text, label_text
