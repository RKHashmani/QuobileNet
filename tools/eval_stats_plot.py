import matplotlib.pyplot as plt
import csv
import numpy as np

with open('log_validation.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')  
    quantum = np.array(list(reader)).astype(float)

with open('log_validation_classic.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')  
    classical = np.array(list(reader)).astype(float)[:6,:]

plt.plot(np.arange(0, (classical.shape[0]+0.2-1)/5, 0.2), classical[:,1], label='classical')
plt.plot(np.arange(0, (quantum.shape[0]+0.2-1)/5, 0.2), quantum[:,1], color='purple', label='quantum')
plt.title('Validation Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('validation_loss.png', dpi=600)

plt.clf()
plt.plot(np.arange(0, (classical.shape[0]+0.2-1)/5, 0.2), classical[:,0], label='classical')
plt.plot(np.arange(0, (quantum.shape[0]+0.2-1)/5, 0.2), quantum[:,0], color='purple', label='quantum')
plt.title('Validation Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()
plt.savefig('validation_acc.png', dpi=600)