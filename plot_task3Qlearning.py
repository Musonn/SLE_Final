import matplotlib.pyplot as plt
import numpy as np

def get_avg(name):
    f = open(name,'r')
    Lines = f.readlines()
    while '\n' in Lines: Lines.remove('\n')
    Lines2=list()
    for line in Lines:
        #line = line.split()
        Lines2.append([int(i) for i in line.split()])

    # get avg score over 10 trails
    avg_score_list  = []   
    for __ in range(len(Lines2[0])):
        _=0
        for i in range(10):
            _ += Lines2[i][__]
        avg_score = _/10
        avg_score_list.append(avg_score)
    return avg_score_list

def preprocessing(name):
    f = open(name,'r')
    Lines = f.readlines()
    while '\n' in Lines: Lines.remove('\n')
    Lines2=list()
    for line in Lines:
        #line = line.split()
        Lines2.append([int(i) for i in line.split()])
    return Lines2

avg_score_list_099 = get_avg('result_Q_099.txt')
avg_score_list_100 = get_avg('result_Q_1.txt')
avg_score_list_096 = get_avg('result_Q_096.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(1, len(avg_score_list_099) + 1)
y = avg_score_list_099
plt.scatter(x, y, marker='x')
fit = np.polyfit(x, y, deg=4)
p = np.poly1d(fit)
plt.plot(x,p(x),"r--", label= 'discount = 0.99')

x = np.arange(1, len(avg_score_list_100) + 1)
y = avg_score_list_100
plt.scatter(x, y, marker='x')
fit = np.polyfit(x, y, deg=4)
p = np.poly1d(fit)
plt.plot(x,p(x),"b--", label= 'discount = 1')

x = np.arange(1, len(avg_score_list_096) + 1)
y = avg_score_list_096
plt.scatter(x, y, marker='x')
fit = np.polyfit(x, y, deg=4)
p = np.poly1d(fit)
plt.plot(x,p(x),"m--", label= 'discount = 0.96')

plt.ylabel('Reward')
plt.xlabel('Episode #')
plt.legend()
        




Lines2 = preprocessing('result_Q_099.txt')
x=lambda a: sum(Lines2[0][:a+1])
cum_score_list = [x(i) for i in range(len(Lines2[0]))]  # cuminative reward of 1st trails

Lines3 = preprocessing('result_Q_1.txt')
x=lambda a: sum(Lines2[0][:a+1])
cum_score_list2 = [x(i) for i in range(len(Lines3[1]))]  # cuminative reward of 2nd trails

Lines4 = preprocessing('result_Q_096.txt')
x=lambda a: sum(Lines2[0][:a+1])
cum_score_list3 = [x(i) for i in range(len(Lines4[8]))]  # cuminative reward of 9th trails

fig2 = plt.figure()
ax = fig2.add_subplot(111)
x = np.arange(1, len(cum_score_list) + 1)
y = cum_score_list
plt.scatter(x, y, marker='x', label= 'discount = 0.99')
# fit = np.polyfit(x, y, deg=4)
# p = np.poly1d(fit)
# plt.plot(x,p(x),"r--")

x = np.arange(1, len(cum_score_list2) + 1)
y = cum_score_list2
plt.scatter(x, y, marker='x', label= 'discount = 1')
# fit = np.polyfit(x, y, deg=4)
# p = np.poly1d(fit)
# plt.plot(x,p(x),"b--")

x = np.arange(1, len(cum_score_list3) + 1)
y = cum_score_list3
plt.scatter(x, y, marker='x', label= 'discount = 0.96')
# fit = np.polyfit(x, y, deg=4)
# p = np.poly1d(fit)
# plt.plot(x,p(x),"m--")

plt.ylabel('Cumulative Reward')
plt.xlabel('Episode #')
plt.legend()

g = open('position and angle.txt','r')
Lines = g.readlines()
Lines2=list()
for line in Lines:
    line = line.split('\t')
    try: line.remove('\n') 
    except ValueError: line.remove('')
    Lines2.append([float(i) for i in line])
# while '\n' in Lines: Lines.remove('\n')

fig3, ax1 = plt.subplots()
ax2 = ax1.twinx()
x = np.arange(1, len(Lines2[0]) + 1)
y = Lines2[0]
ax1.scatter(x, y, marker='x', label= 'x', c='g')
y = Lines2[1]
ax2.scatter(x, y, marker='x', label= 'theta', c='b')

ax1.set_xlabel('Episode #')
ax1.set_ylabel('x', color='g')
ax2.set_ylabel('theta', color='b')
ax1.legend()
ax2.legend()
plt.show()