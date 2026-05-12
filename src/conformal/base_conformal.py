import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

class BaseConformal:
    title_map = {
        'VanillaConformal': 'VCP',
        'FilteringConformal': 'RF/LF',
        'TwoWayFilteringConformal': 'TWF',
        'AdvancedFilteringConformal': 'ATWF',
        'TreeHierarchicalRestrictConformal': 'CRSVP'
    }
    
    def __init__(self):
        self.save_fig_path: str | None = None
        self._calibration_scores_history: list[list[float]] = []

    # ---- Score distribution helpers ----
    def _record_calibration_scores(self, scores: list[float]) -> None:
        """Called inside compute_tau_star to register the calibration scores for
        the current trial.  Raises ValueError when save_fig_path has not been set
        (i.e. compute_tau_star was called outside of compute_factual_results /
        compute_removal_results)."""
        if self.save_fig_path is None:
            raise ValueError(
                "save_fig_path is None. Set self.save_fig_path before calling "
                "compute_tau_star (it is set automatically by compute_results)."
            )
        self._calibration_scores_history.append(list(scores))

    def _get_method_name(self) -> str:
        """Get the display name for the current conformal method."""
        if type(self).__name__ == "FilteringConformal":
            return "RF" if getattr(self, "is_right_filter", True) else "LF"
        return self.title_map.get(type(self).__name__, type(self).__name__)

    
    def plot_score_distribution(self, n_bins: int = 50) -> None:
        """Plot a histogram of calibration scores (mean ± std across all recorded
        trials) and save to <save_fig_dir>/score_distribution.png."""
        if not self._calibration_scores_history:
            return

        save_dir = os.path.dirname(self.save_fig_path)
        save_path = os.path.join(save_dir, "score_distribution.png")

        all_scores_flat = [s for trial in self._calibration_scores_history for s in trial]
        lo, hi = min(all_scores_flat), max(all_scores_flat)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        trial_counts = np.array([
            np.histogram(trial, bins=bin_edges, density=True)[0]
            for trial in self._calibration_scores_history
        ])
        mean_counts = trial_counts.mean(axis=0)
        std_counts = trial_counts.std(axis=0)

        plt.close("all")
        plt.figure(figsize=(6, 4))
        plt.bar(bin_centers, mean_counts, width=bin_width * 0.9, alpha=0.7, label="Mean density")
        plt.errorbar(bin_centers, mean_counts, yerr=std_counts, fmt="none", color="black", capsize=2, label="±1 std")
        plt.xlabel("Calibration Score", fontsize=13)
        plt.ylabel("Density", fontsize=13)
        method_name = self._get_method_name()
        plt.title(f"Score Distribution: {method_name}", fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        self._calibration_scores_history.clear()

    # ---- (1) Initialization part ----
    def initialize(self, data, seed=None):
        """Split data into calibration/test sets and prepare result holders."""
        raise NotImplementedError

    # ---- (2) Threshold computation ----
    def compute_tau_star(self, alpha, calib_data):
        """Return tau_star from calibration data."""
        raise NotImplementedError

    # ---- (3) Evaluation ----
    def evaluate_all(self, tau_star, test_data):
        """Evaluate accuracy and removal rate in one pass given tau_star and test_data.

        Returns (accuracy, total, correct, removal_rate).
        """
        raise NotImplementedError

    # ---- Shared trial loop ----
    def _run_trials(self, data, alphas, n_trials, seed):
        """Run n_trials per alpha, returning accuracy and removal rate stats together."""
        trial_results = []
        for alpha in tqdm(alphas):
            accuracies, tau_stars, totals, corrects, removal_rates = [], [], [], [], []
            for _ in range(n_trials):
                calib_data, test_data = self.initialize(data, seed=seed)
                tau_star = self.compute_tau_star(alpha, calib_data)
                accuracy, total, correct, removal_rate = self.evaluate_all(tau_star, test_data)

                tau_stars.append(tau_star)
                accuracies.append(accuracy)
                totals.append(total)
                corrects.append(correct)
                removal_rates.append(removal_rate)

            trial_results.append({
                'alpha': alpha,
                'tau_stars': tau_stars,
                'accuracies': accuracies,
                'totals': totals,
                'corrects': corrects,
                'removal_rates': removal_rates,
                '_last_test_data': test_data,
            })
        return trial_results

    # ---- Unified results wrapper ----
    def compute_results(
        self, data, alphas, n_trials=1000, seed=None,
        save_accuracy_csv_path=None, save_accuracy_fig_path=None,
        save_removal_csv_path=None, save_removal_fig_path=None,
        updated_data_file=None,
    ):
        self.save_fig_path = save_accuracy_fig_path
        self._calibration_scores_history.clear()

        trial_results = self._run_trials(data, alphas, n_trials, seed)

        accuracy_rows = []
        removal_rows = []
        for r in trial_results:
            accuracy_rows.append({
                'alpha': r['alpha'],
                'tau_star_mean': np.mean(r['tau_stars']),
                'accuracy_mean': np.mean(r['accuracies']),
                'accuracy_std': np.std(r['accuracies']),
                'total_mean': np.mean(r['totals']),
                'correct_mean': np.mean(r['corrects']),
            })
            removal_rows.append({
                'alpha': r['alpha'],
                'tau_star_mean': np.mean(r['tau_stars']),
                'removal_rate_mean': np.mean(r['removal_rates']),
                'removal_rate_std': np.std(r['removal_rates']),
            })

            # Save test data (conformal_set already populated by evaluate_all)
            test_data = r['_last_test_data']
            if r['alpha'] == 0.2 and updated_data_file and len(test_data) > 0:
                with open(updated_data_file, 'w') as f:
                    for d in test_data:
                        f.write(json.dumps(d) + '\n')

        df_accuracy = pd.DataFrame(accuracy_rows)
        df_removal = pd.DataFrame(removal_rows)

        if save_accuracy_csv_path:
            df_accuracy.to_csv(save_accuracy_csv_path, index=False)
        if save_removal_csv_path:
            df_removal.to_csv(save_removal_csv_path, index=False)

        self.plot_results(df_accuracy, save_accuracy_fig_path, len(trial_results[-1]['_last_test_data']))
        if save_accuracy_fig_path:
            self.plot_score_distribution()
        self.plot_removal_results(df_removal, save_removal_fig_path)

        return df_accuracy, df_removal

    # Plot removal results
    def plot_removal_results(self, df_results, save_path):
        plt.close('all')  # Clear any existing figures
        plt.figure(figsize=(4, 3))
        
        plt.plot(1 - df_results['alpha'], df_results['removal_rate_mean'], marker='s', linestyle='-', color='red')
        plt.fill_between(1 - df_results['alpha'], df_results['removal_rate_mean'] - df_results['removal_rate_std'], 
                        df_results['removal_rate_mean'] + df_results['removal_rate_std'], alpha=0.2, color='red')
        plt.xlabel(r"Target coverage $(1-\alpha)$",fontsize=14)
        plt.ylabel('Empirical Removal Rate',fontsize=14)
        method_name = self._get_method_name()
        plt.title(f'Removal Rate Under Target Coverage: {method_name}')
        plt.grid()
        
        plt.tight_layout()

        # plot y = 1-x line between [0, 1]
        alphas = df_results['alpha']
        min_range = max(1 - alphas.max() - 0.1, 0)
        max_range = min(1 - alphas.min() + 0.1, 1)
        plt.plot([min_range, max_range], [max_range, min_range], 'k--', lw=1)
        plt.savefig(save_path)
        plt.close()  # Close the figure after saving

    #plot
    def plot_results(self, df_results, save_path, test_data_size):
        plt.close('all')  # Clear any existing figures
        plt.figure(figsize=(4, 3))  # Create new figure
        plt.plot(1 - df_results['alpha'], df_results['accuracy_mean'], marker='o', linestyle='-')
        plt.fill_between(1 - df_results['alpha'], df_results['accuracy_mean'] - df_results['accuracy_std'], df_results['accuracy_mean'] + df_results['accuracy_std'], alpha=0.2)
        plt.xlabel(r"Target coverage $(1-\alpha)$", fontsize=14)
        plt.ylabel('Empirical coverage',fontsize=14)
        method_name = self._get_method_name()
        plt.title(f'Empirical vs. Target Coverage: {method_name}')
        plt.grid()
        alphas = df_results['alpha']
        # Add y=x line among all alpha range within [0, 1]
        min_range = max(1 - alphas.max() - 0.1, 0)
        max_range = min(1 - alphas.min() + 0.1, 1)
        plt.plot([min_range, max_range], [min_range - 1/test_data_size, max_range - 1/test_data_size], 'k--', lw=1)  
        plt.plot([min_range, max_range], [min_range + 1/test_data_size, max_range + 1/test_data_size], 'k--', lw=1)

        plt.tight_layout()
        #save it to empirical_accuracy_plot.png
        plt.savefig(save_path)
        plt.close()  # Close the figure after saving
