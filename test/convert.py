

encoding = 'big5hkscs'
lines = []
num_errors = 0
with open('training.txt', encoding=encoding, errors='ignore') as f:
    while 1:
        try:
            line = f.readline()
            if line == "":
                break
            lines.append(line)
        except UnicodeDecodeError as e:
            num_errors += 1
print('Encountered %d decoding errors.' % num_errors)
print(len(lines))