import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def plot_pipeline_history(history):
    """
    Plots the evolution of Balanced Accuracy for KNN and NB across pipeline steps.
    Enhanced with 4 decimal places and improvement arrows.
    Expects history to be a list of dicts: [{'Step': 'Baseline', 'KNN': 0.5, 'NB': 0.4}, ...]
    """
    steps = [h['Step'] for h in history]
    knn_scores = [h['KNN'] for h in history]
    nb_scores = [h['NB'] for h in history]
    
    # Calculate improvements
    knn_improvements = [0] + [knn_scores[i] - knn_scores[i-1] for i in range(1, len(knn_scores))]
    nb_improvements = [0] + [nb_scores[i] - nb_scores[i-1] for i in range(1, len(nb_scores))]
    
    # Define colors based on improvement
    def get_colors(improvements):
        return ['green' if imp > 0.001 else 'red' if imp < -0.001 else 'gray' for imp in improvements]
    
    knn_colors = get_colors(knn_improvements)
    nb_colors = get_colors(nb_improvements)
    
    # Setup plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. KNN Chart
    bars_knn = axes[0].bar(range(len(steps)), knn_scores, color=knn_colors, alpha=0.7, edgecolor='black')
    axes[0].set_title("KNN Performance Evolution", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Balanced Accuracy", fontweight='bold')
    axes[0].set_xticks(range(len(steps)))
    axes[0].set_xticklabels(steps, rotation=45, ha='right')
    axes[0].set_ylim([min(knn_scores) - 0.05, max(knn_scores) + 0.05])
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels and improvement arrows for KNN
    for i, (bar, score, improvement) in enumerate(zip(bars_knn, knn_scores, knn_improvements)):
        height = bar.get_height()
        
        # Score label with 4 decimals
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Improvement arrow and percentage only (skip first step)
        if i > 0:
            arrow_symbol = '↑' if improvement > 0.001 else '↓' if improvement < -0.001 else '→'
            color = 'green' if improvement > 0.001 else 'red' if improvement < -0.001 else 'gray'
            
            # Calculate percentage change relative to previous score
            pct_change = (improvement / knn_scores[i-1]) * 100 if knn_scores[i-1] != 0 else 0
            
            axes[0].text(bar.get_x() + bar.get_width()/2., height - 0.015,
                        f'{arrow_symbol} {pct_change:+.2f}%',
                        ha='center', va='top', fontsize=9, color=color, fontweight='bold')

    # 2. NB Chart
    bars_nb = axes[1].bar(range(len(steps)), nb_scores, color=nb_colors, alpha=0.7, edgecolor='black')
    axes[1].set_title("Naïve Bayes Performance Evolution", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Balanced Accuracy", fontweight='bold')
    axes[1].set_xticks(range(len(steps)))
    axes[1].set_xticklabels(steps, rotation=45, ha='right')
    axes[1].set_ylim([min(nb_scores) - 0.05, max(nb_scores) + 0.05])
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels and improvement arrows for NB
    for i, (bar, score, improvement) in enumerate(zip(bars_nb, nb_scores, nb_improvements)):
        height = bar.get_height()
        
        # Score label with 4 decimals
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Improvement arrow and percentage only (skip first step)
        if i > 0:
            arrow_symbol = '↑' if improvement > 0.001 else '↓' if improvement < -0.001 else '→'
            color = 'green' if improvement > 0.001 else 'red' if improvement < -0.001 else 'gray'
            
            # Calculate percentage change relative to previous score
            pct_change = (improvement / nb_scores[i-1]) * 100 if nb_scores[i-1] != 0 else 0
            
            axes[1].text(bar.get_x() + bar.get_width()/2., height - 0.015,
                        f'{arrow_symbol} {pct_change:+.2f}%',
                        ha='center', va='top', fontsize=9, color=color, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Improvement'),
        Patch(facecolor='red', alpha=0.7, label='Decline'),
        Patch(facecolor='gray', alpha=0.7, label='No Change')
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=10)
        
    plt.tight_layout()
    plt.savefig("pipeline_performance_evolution.png", dpi=300, bbox_inches='tight')
    print("   -> Saved plot: pipeline_performance_evolution.png")
    
    # Create summary table
    create_summary_table(steps, knn_scores, knn_improvements, nb_scores, nb_improvements)
    
    plt.show()

def create_summary_table(steps, knn_scores, knn_improvements, nb_scores, nb_improvements):
    """
    Creates a detailed summary table showing step-by-step improvements
    """
    fig, ax = plt.subplots(figsize=(14, len(steps) * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    table_data.append(['Step', 'KNN Score', 'KNN Δ', 'KNN %', 'NB Score', 'NB Δ', 'NB %', 'Avg Score', 'Avg Δ', 'Avg %'])
    
    avg_scores = [(knn + nb) / 2 for knn, nb in zip(knn_scores, nb_scores)]
    avg_improvements = [0] + [avg_scores[i] - avg_scores[i-1] for i in range(1, len(avg_scores))]
    
    for i, step in enumerate(steps):
        # KNN metrics
        knn_delta = f"{knn_improvements[i]:+.4f}" if i > 0 else "-"
        knn_pct = f"{(knn_improvements[i]/knn_scores[i-1]*100):+.2f}%" if i > 0 and knn_scores[i-1] != 0 else "-"
        
        # NB metrics
        nb_delta = f"{nb_improvements[i]:+.4f}" if i > 0 else "-"
        nb_pct = f"{(nb_improvements[i]/nb_scores[i-1]*100):+.2f}%" if i > 0 and nb_scores[i-1] != 0 else "-"
        
        # Average metrics
        avg_delta = f"{avg_improvements[i]:+.4f}" if i > 0 else "-"
        avg_pct = f"{(avg_improvements[i]/avg_scores[i-1]*100):+.2f}%" if i > 0 and avg_scores[i-1] != 0 else "-"
        
        table_data.append([
            step,
            f"{knn_scores[i]:.4f}",
            knn_delta,
            knn_pct,
            f"{nb_scores[i]:.4f}",
            nb_delta,
            nb_pct,
            f"{avg_scores[i]:.4f}",
            avg_delta,
            avg_pct
        ])
    
    # Add total improvement row
    total_knn = knn_scores[-1] - knn_scores[0]
    total_knn_pct = (total_knn / knn_scores[0] * 100) if knn_scores[0] != 0 else 0
    total_nb = nb_scores[-1] - nb_scores[0]
    total_nb_pct = (total_nb / nb_scores[0] * 100) if nb_scores[0] != 0 else 0
    total_avg = avg_scores[-1] - avg_scores[0]
    total_avg_pct = (total_avg / avg_scores[0] * 100) if avg_scores[0] != 0 else 0
    
    table_data.append([
        'TOTAL IMPROVEMENT',
        '',
        f"{total_knn:+.4f}",
        f"{total_knn_pct:+.2f}%",
        '',
        f"{total_nb:+.4f}",
        f"{total_nb_pct:+.2f}%",
        '',
        f"{total_avg:+.4f}",
        f"{total_avg_pct:+.2f}%"
    ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.18, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(10):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style total row
    for i in range(10):
        table[(len(table_data)-1, i)].set_facecolor('#FFD966')
        table[(len(table_data)-1, i)].set_text_props(weight='bold')
    
    # Color code delta and percentage columns
    for row in range(1, len(table_data)-1):
        for col in [2, 3, 5, 6, 8, 9]:  # Delta and % columns
            cell = table[(row, col)]
            text = cell.get_text().get_text()
            if text != '-':
                # Check if it's a percentage or delta
                if '%' in text:
                    value = float(text.replace('%', ''))
                else:
                    value = float(text)
                
                if value > 0.001:
                    cell.set_facecolor('#C6EFCE')  # Light green
                elif value < -0.001:
                    cell.set_facecolor('#FFC7CE')  # Light red
    
    plt.title('Pipeline Performance Summary (with Percentage Changes)', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('pipeline_summary_table.png', dpi=300, bbox_inches='tight')
    print("   -> Saved plot: pipeline_summary_table.png")

def plot_final_confusion_matrices(X_train, y_train, X_test, y_test):
    """
    Trains final KNN and NB models and plots their confusion matrices.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Final KNN
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, 
                                      display_labels=['No Injury', 'Injury/Tow'])
    disp_knn.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title("Final KNN Confusion Matrix", fontsize=13, fontweight='bold')
    
    # 2. Final Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, 
                                     display_labels=['No Injury', 'Injury/Tow'])
    disp_nb.plot(ax=axes[1], cmap='Greens', values_format='d')
    axes[1].set_title("Final Naïve Bayes Confusion Matrix", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("final_confusion_matrices.png", dpi=300, bbox_inches='tight')
    print("   -> Saved plot: final_confusion_matrices.png")
    plt.show()