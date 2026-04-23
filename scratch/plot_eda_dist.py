import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_distribution():
    FILE_PATH = "data/processed/tokopedia_reviews_balanced_eda.csv"
    OUTPUT_FIG = "outputs/figures/fig4_4_distribution_after_eda.png"
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
    
    df = pd.read_csv(FILE_PATH)
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='sentiment_label', palette='viridis', hue='sentiment_label', legend=False)
    
    plt.title('Gambar 4.4 Distribusi Label Sentimen Setelah Penyeimbangan (EDA)', fontsize=14, fontweight='bold')
    plt.xlabel('Kelas Sentimen')
    plt.ylabel('Jumlah Data')
    
    # Add count labels on top of bars
    for p in plt.gca().patches:
        plt.gca().annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300)
    print(f"Plot saved to {OUTPUT_FIG}")

if __name__ == "__main__":
    plot_distribution()
