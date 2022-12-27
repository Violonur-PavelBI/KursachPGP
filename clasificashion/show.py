import matplotlib.pyplot as plt

def show(mass,a,b):
    y = mass[1][a:b]
    x = mass[0][a:b]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x,y, color='green')
    ax1.set_xlabel(u'Эпоха')
    ax1.set_ylabel(u'Loss', color='green')
    ax1.grid(True, color='green')
    ax1.tick_params(axis='y', which='major', labelcolor='green')
    ax1.set_title(u'Динамика Loss')
    plt.show()
    y = mass[2][a:b]
    x = mass[0][a:b]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x,y, color='green')
    ax1.set_xlabel(u'Эпоха')
    ax1.set_ylabel(u'accuracy', color='green')
    ax1.grid(True, color='green')
    ax1.tick_params(axis='y', which='major', labelcolor='green')
    ax1.set_title(u'Динамика accuracy')
    plt.show()