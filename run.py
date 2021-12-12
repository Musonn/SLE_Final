f = open('result_Q2.txt','r')
Lines = f.readlines()
while '\n' in Lines: Lines.remove('\n')
Lines2=list()
for line in Lines:
    #line = line.split()
    Lines2.append([int(i) for i in line.split()])

avg_score_list  = []    # avg score over 10 trails
for __ in range(len(Lines2[0])):
    _=0
    for i in range(10):
        _ += Lines2[i][__]
    avg_score = _/10
    avg_score_list.append(avg_score)
        
x=lambda a: sum(Lines2[0][:a+1])
cum_score_list = [x(i) for i in range(len(Lines2[0]))]  # cuminative reward of 1st trails

g = open('position and angle.txt','r')
Lines = g.readlines()
Lines2=list()
for line in Lines:
    line = line.split('\t')
    try: line.remove('\n') 
    except ValueError: line.remove('')
    Lines2.append([float(i) for i in line])
while '\n' in Lines: Lines.remove('\n')
print('s')