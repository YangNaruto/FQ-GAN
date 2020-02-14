filename = 'prepare_data.sh'
fileCont = open(filename, 'r').read()
f = open(filename, 'w', newline='\n')
f.write(fileCont)
f.close()