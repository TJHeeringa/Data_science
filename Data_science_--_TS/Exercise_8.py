from Exercise_5 import activities
from dtw_meuk import KnnDtw
from dtw import dtw
import matplotlib.pyplot as plt
import numpy as np

manhattan_norm = lambda x, y: np.abs(x - y)

# for activity in activities:
#     d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_norm)

m = KnnDtw()
for activity in activities:
    print(np.shape(activity["val"].values))
d, cost_matrix, acc_cost_matrix, path = dtw(activities[0]["val"].values, activities[0]["val"].values, dist=manhattan_norm)
print(d)
# split_activities = [[activity["val"].values] for activity in activities]
# split_activities = None
# for activity in activities:
#     if split_activities is None:
#         split_activities = [activity["val"].values]
#         print(np.shape(split_activities))
#     else:
#         print(np.shape(split_activities))
#         split_activities = np.append(split_activities, [activity["val"].values], axis=0)
# print(np.shape(split_activities))
# print(np.shape(split_activities)[0])

# distance_matrix = m._dist_matrix(split_activities, split_activities)
# plt.figure()
# plt.plot(distance_matrix)
#
# plt.figure()
# plt.imshow(distance_matrix, origin='lower', cmap='gray', interpolation='nearest')
# plt.show()

