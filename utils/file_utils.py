def load_file(filepath):
  infile = open(filepath, 'r')
  data = infile.readlines()
  new_list = []
  for line in data:
    line = line.strip()
    new_list.append(line)
  infile.close()
  return new_list
