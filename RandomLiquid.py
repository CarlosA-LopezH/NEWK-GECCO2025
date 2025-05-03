import pickle
from sklearn.linear_model import Perceptron
import csv

from lib.auxiliary import split_data
from lib.encoding import Encoding
from lib.operationsEvo import evaluation

if __name__ == "__main__":
    for ds_name in ["PR8", "PR12"]:
        c = 0
        if ds_name == "PR8":
            c = 8
        else:
            c = 12
        print(f"Data: {ds_name}")
        # Load Dataset
        with open(f"../Datasets/{ds_name}.data", "rb") as file:
            data = pickle.load(file)
        # Separation of data:
        # Train-Test = 70% | Validation = 30% (p_validation)
        # Train = 70% (p_train) | Test = 30%
        data_evolve, data_val = split_data(classes=data["Classes"], data=data, p_validation=0.3, p_train=0.7)
        labels: list = data["Labels Names"]  # List of name of labels.
        # Produce and test 30 random LSM
        accuracy_results = []
        for i in range(30):
            print(f"  - Random Liquid {i}")
            # Create encoding
            encoding = Encoding(20, 8)
            # Evaluate encoding
            acc, _ = evaluation(encoding, data_evolve, time_sim=110., classifier=Perceptron)
            # Store result
            accuracy_results.append(acc)
        # Save results
        with open(f"Results/Results-RandomLiquid_{ds_name}", 'w') as f:
            writer = csv.writer(f)
            for value in accuracy_results:
                writer.writerow(value)

        with open(f"Results/RandomLiquids_Perceptron-GECCO2025-{ds_name}_Summary.data", 'wb') as f:
            pickle.dump({"Values": accuracy_results}, f)






