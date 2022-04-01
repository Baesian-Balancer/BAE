import wandb
import matplotlib.pyplot as plt 
import numpy as np

api = wandb.Api()
run = api.run("/dawon-horvath/openSim2Real/runs/3s9qnkah")
tail = run.scan_history(keys=["training episode reward", "reset orientation"], page_size=10)

reward = [row["training episode reward"] for row in tail if not np.isnan(row["training episode reward"]) ]
reward = [row["reset orientation"] for row in tail if not np.isnan(row["training episode reward"]) ]

# tail = run.history()
# reward = tail["training episode reward"].dropna()
# reset = tail["reset orientation"].dropna()

labels, runs = np.unique(reset, return_counts=True)

print(labels, runs)

ticks = range(len(runs))

plt.figure(1)
plt.title('Number each reset orientation')
plt.bar(ticks,runs, align='center')
plt.xticks(ticks, labels)

# indexes of each reset orientation
rews = np.zeros(len(labels))
for i, label in enumerate(labels):
	mask = reset == label
	rews[i] = np.sum(reward[mask])
	
plt.figure(2)
plt.title('total rewards for each reset orientation')
plt.bar(ticks,rews, align='center')
plt.xticks(ticks, labels)


plt.figure(3)
plt.title('Rewards per training run for each reset poisiton')
plt.bar(ticks,rews/runs, align='center')
plt.xticks(ticks, labels)

# indexes of each reset orientation
rews = np.zeros(len(labels))
for i, label in enumerate(labels):
	mask = reset == label
	rews[i] = np.sum(reward[mask] == 0)

plt.figure(4)
plt.title('Number zeros rews for each reset orientation')
plt.bar(ticks,rews, align='center')
plt.xticks(ticks, labels)
plt.show()
