from data import load_trajectory
import numpy as np
import numpy.matlib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import analysis, model, data

pos = data.load_trajectory("aligned_lattice.csv", "/Users/b/Downloads/Deploy_model/")
PI_nor = analysis.compute_PI(pos)

autoencoder, encoder = model.build_autoencoder(input_dim=PI_nor.shape[1], bottleneck=50)

# train model stack1
PI_nor = np.array(PI_nor)
x_train, x_test, = train_test_split(PI_nor, test_size=0.2, random_state=42)


# Train the model
history = autoencoder.fit(x_train,
                x_train,
                epochs= 10,
                steps_per_epoch = 100,
                batch_size = None,
                validation_data = (x_test, x_test),
                validation_steps = 10,
                shuffle= False)

#autoencoder.save(data_fol + "autoencoder_aligned_lattice")
model.save_autoencoder(autoencoder, "autoencoder_aligned_lattice")
