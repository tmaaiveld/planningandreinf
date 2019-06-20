import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# k = range(1, 6)
k = [1, 2, 3, 5, 10, 20, 50, 100]
print k
ax = plt.figure().gca()
# plt.plot(k,[-90000,-91000,-92000,-93000,-94000])
plt.plot(k,[-94232.9930926, -84110.5686959, -75871.0164206, -74115.3134138, -74094.8492289,-73484.8020663, -73463.8295259, -73455.7603475], linestyle='--', marker='o')

plt.xlabel('k')
plt.ylabel('log likelihood')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()