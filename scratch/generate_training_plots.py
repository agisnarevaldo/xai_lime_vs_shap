
import matplotlib.pyplot as plt

# Data from notebook 03_modeling_finetuning_bert.ipynb
epochs = [1, 2, 3, 4, 5]
train_loss = [0.772270, 0.305110, 0.245581, 0.025586, 0.040559]
val_loss = [1.093899, 0.661995, 0.824083, 1.173830, 1.163000]
accuracy = [0.932692, 0.939103, 0.945513, 0.948718, 0.939103]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b-o', label='Training Loss')
plt.plot(epochs, val_loss, 'r-o', label='Validation Loss')
plt.title('Gambar 4.5 Grafik Training dan Validation Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('d:/xai_lime_vs_shap/outputs/figures/fig4_5_training_loss.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy, 'g-o', label='Validation Accuracy')
plt.title('Grafik Akurasi Validasi Selama Pelatihan', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('d:/xai_lime_vs_shap/outputs/figures/fig4_accuracy_progress.png', bbox_inches='tight', dpi=300)
plt.close()

print("✅ Figures generated: fig4_5_training_loss.png and fig4_accuracy_progress.png")
