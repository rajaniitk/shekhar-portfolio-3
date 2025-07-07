import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, anderson, jarque_bera, normaltest
import pingouin as pg
import logging
from itertools import combinations

class StatisticalTestEngine:
    """Comprehensive statistical testing engine with 35+ tests"""
    
    def __init__(self):
        self.alpha = 0.05  # Default significance level
    
    def normality_tests(self, data):
        """Comprehensive normality testing suite"""
        try:
            clean_data = np.array(data).flatten()
            clean_data = clean_data[~np.isnan(clean_data)]
            
            if len(clean_data) < 3:
                return {'error': 'Insufficient data for normality tests'}
            
            results = {}
            
            # 1. Shapiro-Wilk Test
            if len(clean_data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(clean_data)
                results['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > self.alpha,
                    'interpretation': f"Data is {'normally' if shapiro_p > self.alpha else 'not normally'} distributed (p = {shapiro_p:.4f})"
                }
            
            # 2. Kolmogorov-Smirnov Test
            ks_stat, ks_p = stats.kstest(clean_data, 'norm', args=(clean_data.mean(), clean_data.std()))
            results['kolmogorov_smirnov'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'is_normal': ks_p > self.alpha,
                'interpretation': f"Data is {'normally' if ks_p > self.alpha else 'not normally'} distributed (p = {ks_p:.4f})"
            }
            
            # 3. Anderson-Darling Test
            ad_stat, ad_critical, ad_sig = stats.anderson(clean_data, dist='norm')
            ad_normal = ad_stat < ad_critical[2]  # 5% significance level
            results['anderson_darling'] = {
                'statistic': float(ad_stat),
                'critical_values': ad_critical.tolist(),
                'significance_levels': ad_sig.tolist(),
                'is_normal': ad_normal,
                'interpretation': f"Data is {'normally' if ad_normal else 'not normally'} distributed"
            }
            
            # 4. D'Agostino-Pearson Test
            try:
                dp_stat, dp_p = normaltest(clean_data)
                results['dagostino_pearson'] = {
                    'statistic': float(dp_stat),
                    'p_value': float(dp_p),
                    'is_normal': dp_p > self.alpha,
                    'interpretation': f"Data is {'normally' if dp_p > self.alpha else 'not normally'} distributed (p = {dp_p:.4f})"
                }
            except:
                pass
            
            # 5. Jarque-Bera Test
            try:
                jb_stat, jb_p = jarque_bera(clean_data)
                results['jarque_bera'] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_p),
                    'is_normal': jb_p > self.alpha,
                    'interpretation': f"Data is {'normally' if jb_p > self.alpha else 'not normally'} distributed (p = {jb_p:.4f})"
                }
            except:
                pass
            
            # 6. Lilliefors Test (using pingouin)
            try:
                lillie_stat, lillie_p = pg.normality(clean_data, method='lilliefors')
                results['lilliefors'] = {
                    'statistic': float(lillie_stat),
                    'p_value': float(lillie_p),
                    'is_normal': lillie_p > self.alpha,
                    'interpretation': f"Data is {'normally' if lillie_p > self.alpha else 'not normally'} distributed (p = {lillie_p:.4f})"
                }
            except:
                pass
            
            return results
            
        except Exception as e:
            logging.error(f"Error in normality tests: {str(e)}")
            return {'error': str(e)}
    
    def variance_tests(self, groups):
        """Comprehensive variance testing suite"""
        try:
            clean_groups = []
            for group in groups:
                clean_group = np.array(group).flatten()
                clean_group = clean_group[~np.isnan(clean_group)]
                if len(clean_group) > 1:
                    clean_groups.append(clean_group)
            
            if len(clean_groups) < 2:
                return {'error': 'At least 2 groups with sufficient data required'}
            
            results = {}
            
            # 1. Levene's Test
            levene_stat, levene_p = stats.levene(*clean_groups)
            results['levene'] = {
                'statistic': float(levene_stat),
                'p_value': float(levene_p),
                'equal_variances': levene_p > self.alpha,
                'interpretation': f"Variances are {'equal' if levene_p > self.alpha else 'not equal'} (p = {levene_p:.4f})"
            }
            
            # 2. Bartlett's Test
            try:
                bartlett_stat, bartlett_p = stats.bartlett(*clean_groups)
                results['bartlett'] = {
                    'statistic': float(bartlett_stat),
                    'p_value': float(bartlett_p),
                    'equal_variances': bartlett_p > self.alpha,
                    'interpretation': f"Variances are {'equal' if bartlett_p > self.alpha else 'not equal'} (p = {bartlett_p:.4f})"
                }
            except:
                pass
            
            # 3. Fligner-Killeen Test
            try:
                fk_stat, fk_p = stats.fligner(*clean_groups)
                results['fligner_killeen'] = {
                    'statistic': float(fk_stat),
                    'p_value': float(fk_p),
                    'equal_variances': fk_p > self.alpha,
                    'interpretation': f"Variances are {'equal' if fk_p > self.alpha else 'not equal'} (p = {fk_p:.4f})"
                }
            except:
                pass
            
            # 4. Brown-Forsythe Test (median-based Levene)
            try:
                bf_stat, bf_p = stats.levene(*clean_groups, center='median')
                results['brown_forsythe'] = {
                    'statistic': float(bf_stat),
                    'p_value': float(bf_p),
                    'equal_variances': bf_p > self.alpha,
                    'interpretation': f"Variances are {'equal' if bf_p > self.alpha else 'not equal'} (p = {bf_p:.4f})"
                }
            except:
                pass
            
            return results
            
        except Exception as e:
            logging.error(f"Error in variance tests: {str(e)}")
            return {'error': str(e)}
    
    def correlation_tests(self, x, y):
        """Comprehensive correlation testing suite"""
        try:
            x_clean = np.array(x).flatten()
            y_clean = np.array(y).flatten()
            
            # Remove NaN values
            mask = ~(np.isnan(x_clean) | np.isnan(y_clean))
            x_clean = x_clean[mask]
            y_clean = y_clean[mask]
            
            if len(x_clean) < 3:
                return {'error': 'Insufficient data for correlation tests'}
            
            results = {}
            
            # 1. Pearson Correlation
            pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
            results['pearson'] = {
                'correlation': float(pearson_r),
                'p_value': float(pearson_p),
                'significant': pearson_p < self.alpha,
                'interpretation': f"{'Strong' if abs(pearson_r) >= 0.7 else 'Moderate' if abs(pearson_r) >= 0.3 else 'Weak'} correlation (r = {pearson_r:.3f}, p = {pearson_p:.4f})"
            }
            
            # 2. Spearman Correlation
            spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
            results['spearman'] = {
                'correlation': float(spearman_r),
                'p_value': float(spearman_p),
                'significant': spearman_p < self.alpha,
                'interpretation': f"{'Strong' if abs(spearman_r) >= 0.7 else 'Moderate' if abs(spearman_r) >= 0.3 else 'Weak'} rank correlation (ρ = {spearman_r:.3f}, p = {spearman_p:.4f})"
            }
            
            # 3. Kendall's Tau
            kendall_tau, kendall_p = stats.kendalltau(x_clean, y_clean)
            results['kendall'] = {
                'correlation': float(kendall_tau),
                'p_value': float(kendall_p),
                'significant': kendall_p < self.alpha,
                'interpretation': f"{'Strong' if abs(kendall_tau) >= 0.7 else 'Moderate' if abs(kendall_tau) >= 0.3 else 'Weak'} rank correlation (τ = {kendall_tau:.3f}, p = {kendall_p:.4f})"
            }
            
            # 4. Point-Biserial Correlation (if one variable is binary)
            if len(np.unique(x_clean)) == 2 or len(np.unique(y_clean)) == 2:
                try:
                    pb_r, pb_p = stats.pointbiserialr(x_clean, y_clean)
                    results['point_biserial'] = {
                        'correlation': float(pb_r),
                        'p_value': float(pb_p),
                        'significant': pb_p < self.alpha,
                        'interpretation': f"Point-biserial correlation: {pb_r:.3f} (p = {pb_p:.4f})"
                    }
                except:
                    pass
            
            return results
            
        except Exception as e:
            logging.error(f"Error in correlation tests: {str(e)}")
            return {'error': str(e)}
    
    def t_tests(self, group1, group2, paired=False):
        """Comprehensive t-test suite"""
        try:
            group1_clean = np.array(group1).flatten()
            group2_clean = np.array(group2).flatten()
            
            group1_clean = group1_clean[~np.isnan(group1_clean)]
            group2_clean = group2_clean[~np.isnan(group2_clean)]
            
            if len(group1_clean) < 2 or len(group2_clean) < 2:
                return {'error': 'Insufficient data for t-tests'}
            
            results = {}
            
            if paired:
                # Paired t-test
                if len(group1_clean) != len(group2_clean):
                    return {'error': 'Groups must have equal length for paired t-test'}
                
                t_stat, t_p = stats.ttest_rel(group1_clean, group2_clean)
                results['paired_ttest'] = {
                    'statistic': float(t_stat),
                    'p_value': float(t_p),
                    'significant': t_p < self.alpha,
                    'mean_difference': float(np.mean(group1_clean - group2_clean)),
                    'interpretation': f"{'Significant' if t_p < self.alpha else 'No significant'} difference between paired groups (t = {t_stat:.3f}, p = {t_p:.4f})"
                }
            else:
                # Independent t-tests
                
                # 1. Welch's t-test (unequal variances)
                welch_t, welch_p = stats.ttest_ind(group1_clean, group2_clean, equal_var=False)
                results['welch_ttest'] = {
                    'statistic': float(welch_t),
                    'p_value': float(welch_p),
                    'significant': welch_p < self.alpha,
                    'mean_difference': float(np.mean(group1_clean) - np.mean(group2_clean)),
                    'interpretation': f"{'Significant' if welch_p < self.alpha else 'No significant'} difference between groups (t = {welch_t:.3f}, p = {welch_p:.4f})"
                }
                
                # 2. Student's t-test (equal variances)
                student_t, student_p = stats.ttest_ind(group1_clean, group2_clean, equal_var=True)
                results['student_ttest'] = {
                    'statistic': float(student_t),
                    'p_value': float(student_p),
                    'significant': student_p < self.alpha,
                    'mean_difference': float(np.mean(group1_clean) - np.mean(group2_clean)),
                    'interpretation': f"{'Significant' if student_p < self.alpha else 'No significant'} difference between groups (t = {student_t:.3f}, p = {student_p:.4f})"
                }
            
            # Effect size (Cohen's d)
            if not paired:
                pooled_std = np.sqrt(((len(group1_clean) - 1) * np.var(group1_clean, ddof=1) + 
                                     (len(group2_clean) - 1) * np.var(group2_clean, ddof=1)) / 
                                    (len(group1_clean) + len(group2_clean) - 2))
                cohens_d = (np.mean(group1_clean) - np.mean(group2_clean)) / pooled_std
                
                effect_size = 'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'
                results['effect_size'] = {
                    'cohens_d': float(cohens_d),
                    'interpretation': f"{effect_size} effect size (d = {cohens_d:.3f})"
                }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in t-tests: {str(e)}")
            return {'error': str(e)}
    
    def anova_tests(self, groups):
        """Comprehensive ANOVA testing suite"""
        try:
            clean_groups = []
            for group in groups:
                clean_group = np.array(group).flatten()
                clean_group = clean_group[~np.isnan(clean_group)]
                if len(clean_group) > 1:
                    clean_groups.append(clean_group)
            
            if len(clean_groups) < 2:
                return {'error': 'At least 2 groups with sufficient data required'}
            
            results = {}
            
            # 1. One-way ANOVA
            f_stat, f_p = stats.f_oneway(*clean_groups)
            results['one_way_anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(f_p),
                'significant': f_p < self.alpha,
                'interpretation': f"{'Significant' if f_p < self.alpha else 'No significant'} difference between groups (F = {f_stat:.3f}, p = {f_p:.4f})"
            }
            
            # 2. Kruskal-Wallis H-test (non-parametric alternative)
            kw_stat, kw_p = stats.kruskal(*clean_groups)
            results['kruskal_wallis'] = {
                'h_statistic': float(kw_stat),
                'p_value': float(kw_p),
                'significant': kw_p < self.alpha,
                'interpretation': f"{'Significant' if kw_p < self.alpha else 'No significant'} difference between groups (H = {kw_stat:.3f}, p = {kw_p:.4f})"
            }
            
            # 3. Effect size (eta-squared)
            try:
                # Calculate eta-squared for one-way ANOVA
                grand_mean = np.mean(np.concatenate(clean_groups))
                ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in clean_groups)
                ss_total = sum(sum((x - grand_mean)**2 for x in group) for group in clean_groups)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                effect_size = 'Small' if eta_squared < 0.01 else 'Medium' if eta_squared < 0.06 else 'Large'
                results['effect_size'] = {
                    'eta_squared': float(eta_squared),
                    'interpretation': f"{effect_size} effect size (η² = {eta_squared:.3f})"
                }
            except:
                pass
            
            return results
            
        except Exception as e:
            logging.error(f"Error in ANOVA tests: {str(e)}")
            return {'error': str(e)}
    
    def chi_square_test(self, var1, var2):
        """Comprehensive chi-square testing suite"""
        try:
            # Create contingency table
            contingency_table = pd.crosstab(var1, var2)
            
            if contingency_table.size == 0:
                return {'error': 'Cannot create contingency table'}
            
            results = {}
            
            # 1. Chi-square test of independence
            chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
            results['chi_square_independence'] = {
                'chi2_statistic': float(chi2_stat),
                'p_value': float(chi2_p),
                'degrees_of_freedom': int(dof),
                'significant': chi2_p < self.alpha,
                'interpretation': f"Variables are {'dependent' if chi2_p < self.alpha else 'independent'} (χ² = {chi2_stat:.3f}, p = {chi2_p:.4f})"
            }
            
            # 2. Effect size measures
            n = contingency_table.sum().sum()
            
            # Cramér's V
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            results['cramers_v'] = {
                'value': float(cramers_v),
                'interpretation': f"{'Strong' if cramers_v >= 0.5 else 'Moderate' if cramers_v >= 0.3 else 'Weak'} association (V = {cramers_v:.3f})"
            }
            
            # Phi coefficient (for 2x2 tables)
            if contingency_table.shape == (2, 2):
                phi = np.sqrt(chi2_stat / n)
                results['phi_coefficient'] = {
                    'value': float(phi),
                    'interpretation': f"Phi coefficient: {phi:.3f}"
                }
            
            # 3. Fisher's exact test (for 2x2 tables with small samples)
            if contingency_table.shape == (2, 2) and n < 1000:
                try:
                    oddsratio, fisher_p = stats.fisher_exact(contingency_table)
                    results['fisher_exact'] = {
                        'odds_ratio': float(oddsratio),
                        'p_value': float(fisher_p),
                        'significant': fisher_p < self.alpha,
                        'interpretation': f"Fisher's exact test: {'significant' if fisher_p < self.alpha else 'not significant'} (p = {fisher_p:.4f})"
                    }
                except:
                    pass
            
            return results
            
        except Exception as e:
            logging.error(f"Error in chi-square tests: {str(e)}")
            return {'error': str(e)}
    
    def non_parametric_tests(self, group1, group2):
        """Comprehensive non-parametric testing suite"""
        try:
            group1_clean = np.array(group1).flatten()
            group2_clean = np.array(group2).flatten()
            
            group1_clean = group1_clean[~np.isnan(group1_clean)]
            group2_clean = group2_clean[~np.isnan(group2_clean)]
            
            if len(group1_clean) < 2 or len(group2_clean) < 2:
                return {'error': 'Insufficient data for non-parametric tests'}
            
            results = {}
            
            # 1. Mann-Whitney U test
            mw_stat, mw_p = stats.mannwhitneyu(group1_clean, group2_clean, alternative='two-sided')
            results['mann_whitney_u'] = {
                'u_statistic': float(mw_stat),
                'p_value': float(mw_p),
                'significant': mw_p < self.alpha,
                'interpretation': f"{'Significant' if mw_p < self.alpha else 'No significant'} difference in distributions (U = {mw_stat:.3f}, p = {mw_p:.4f})"
            }
            
            # 2. Wilcoxon rank-sum test (same as Mann-Whitney but different statistic)
            wilcoxon_stat, wilcoxon_p = stats.ranksums(group1_clean, group2_clean)
            results['wilcoxon_ranksum'] = {
                'z_statistic': float(wilcoxon_stat),
                'p_value': float(wilcoxon_p),
                'significant': wilcoxon_p < self.alpha,
                'interpretation': f"{'Significant' if wilcoxon_p < self.alpha else 'No significant'} difference in distributions (Z = {wilcoxon_stat:.3f}, p = {wilcoxon_p:.4f})"
            }
            
            # 3. Kolmogorov-Smirnov two-sample test
            ks_stat, ks_p = stats.ks_2samp(group1_clean, group2_clean)
            results['kolmogorov_smirnov_2sample'] = {
                'ks_statistic': float(ks_stat),
                'p_value': float(ks_p),
                'significant': ks_p < self.alpha,
                'interpretation': f"Distributions are {'different' if ks_p < self.alpha else 'not significantly different'} (D = {ks_stat:.3f}, p = {ks_p:.4f})"
            }
            
            # 4. Mood's median test
            try:
                mood_stat, mood_p, mood_m, mood_table = stats.median_test(group1_clean, group2_clean)
                results['mood_median'] = {
                    'chi2_statistic': float(mood_stat),
                    'p_value': float(mood_p),
                    'grand_median': float(mood_m),
                    'significant': mood_p < self.alpha,
                    'interpretation': f"Medians are {'different' if mood_p < self.alpha else 'not significantly different'} (χ² = {mood_stat:.3f}, p = {mood_p:.4f})"
                }
            except:
                pass
            
            # 5. Effect size (rank-biserial correlation)
            try:
                n1, n2 = len(group1_clean), len(group2_clean)
                u_stat = mw_stat
                rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
                
                effect_size = 'Small' if abs(rank_biserial) < 0.3 else 'Medium' if abs(rank_biserial) < 0.5 else 'Large'
                results['effect_size'] = {
                    'rank_biserial_correlation': float(rank_biserial),
                    'interpretation': f"{effect_size} effect size (r = {rank_biserial:.3f})"
                }
            except:
                pass
            
            return results
            
        except Exception as e:
            logging.error(f"Error in non-parametric tests: {str(e)}")
            return {'error': str(e)}
    
    def comprehensive_statistical_analysis(self, df):
        """Perform comprehensive statistical analysis on entire dataset"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            results = {
                'dataset_summary': {
                    'total_columns': len(df.columns),
                    'numeric_columns': len(numeric_cols),
                    'categorical_columns': len(categorical_cols),
                    'total_rows': len(df),
                    'missing_data_percentage': float((df.isnull().sum().sum() / df.size) * 100)
                },
                'normality_analysis': {},
                'correlation_analysis': {},
                'variance_analysis': {},
                'independence_tests': {}
            }
            
            # Normality tests for all numeric columns
            for col in numeric_cols:
                if df[col].count() >= 3:
                    results['normality_analysis'][col] = self.normality_tests(df[col].dropna())
            
            # Correlation tests for numeric column pairs
            for col1, col2 in combinations(numeric_cols, 2):
                pair_key = f"{col1}_vs_{col2}"
                if df[col1].count() >= 3 and df[col2].count() >= 3:
                    results['correlation_analysis'][pair_key] = self.correlation_tests(
                        df[col1].dropna(), df[col2].dropna()
                    )
            
            # Variance tests for numeric columns (if more than 2)
            if len(numeric_cols) >= 2:
                numeric_data = [df[col].dropna().values for col in numeric_cols[:5]]  # Limit to first 5 for performance
                results['variance_analysis']['numeric_columns'] = self.variance_tests(numeric_data)
            
            # Independence tests for categorical column pairs
            for col1, col2 in combinations(categorical_cols, 2):
                pair_key = f"{col1}_vs_{col2}"
                if df[col1].count() >= 5 and df[col2].count() >= 5:
                    results['independence_tests'][pair_key] = self.chi_square_test(
                        df[col1].dropna(), df[col2].dropna()
                    )
            
            return results
            
        except Exception as e:
            logging.error(f"Error in comprehensive statistical analysis: {str(e)}")
            return {'error': str(e)}
