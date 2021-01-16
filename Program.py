import matplotlib.pyplot as plt

from FirstData import FirstData
from SecondData import SecondData
from ThirdData import ThirdData


def find_every_thing(data):
    data.silhouette_all()
    optimal_fcm = data.get_optimal_number_of_clustering_fcm()
    optimal_kmeans = data.get_optimal_number_of_clustering_kmeans()
    optimal_hierarchical = data.get_optimal_number_of_clustering_hierarchical()
    optimal_spectral = data.get_optimal_number_of_clustering_spectral()
    optimal_gmm = data.get_optimal_number_of_clustering_gmm()
    print("FCM- optimal number of clustering" + str(optimal_fcm))
    print("kmeans- optimal number of clustering" + str(optimal_kmeans))
    print("hierarchial- optimal number of clustering" + str(optimal_hierarchical))
    print("spectral- optimal number of clustering" + str(optimal_spectral))
    print("gmm- optimal number of clustering" + str(optimal_gmm))
    data.k_means(optimal_kmeans)
    data.gmm(optimal_gmm)
    data.fcm(optimal_fcm)
    data.hierarchical(optimal_hierarchical)
    data.spectral(optimal_spectral)
    print("AMI FUZZY: " + str(data.ami_fcm(optimal_fcm)))
    print("AMI HIERARCHIAL: " + str(data.ami_hierarchial(optimal_hierarchical)))
    print("AMI GMM: " + str(data.ami_gmm(optimal_gmm)))
    print("AMI KMEANS: " + str(data.ami_kmeans(optimal_kmeans)))
    print("AMI SPECTRAL: " + str(data.ami_spectral(optimal_spectral)))


print('First Data------')
first_data = FirstData()
find_every_thing(first_data)
plt.title('First Data')
plt.show()
# print('Second Data------')
# second_data = SecondData()
# find_every_thing(second_data)
# plt.title('Second Data')
# plt.show()
# print('Third Data')
# third_data = ThirdData()
# find_every_thing(third_data)
# plt.title('Third Data')
# plt.show()
