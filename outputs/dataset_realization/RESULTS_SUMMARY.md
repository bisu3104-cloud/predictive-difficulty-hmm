# Dataset-realization robustness summary

Selected HMMs: 16
Dataset realizations: [0, 1, 2]
Completed run-level conditions: 1152

## HMM-level process ICC
| architecture   | outcome          |   n_hmms |   n_dataset_replicates |   ms_between |   ms_within |   process_variance |   dataset_realization_variance |   process_icc |   icc_ci_low |   icc_ci_high |
|:---------------|:-----------------|---------:|-----------------------:|-------------:|------------:|-------------------:|-------------------------------:|--------------:|-------------:|--------------:|
| GRU            | A_width_shape    |       16 |                      3 |  0.000350482 | 2.78856e-05 |        0.000107532 |                    2.78856e-05 |      0.794077 |     0.600097 |      0.890085 |
| GRU            | mean_excess_ce_H |       16 |                      3 |  0.000693303 | 0.000148586 |        0.000181572 |                    0.000148586 |      0.549954 |     0.211995 |      0.696346 |
| Transformer    | A_width_shape    |       16 |                      3 |  0.000382996 | 2.9696e-05  |        0.000117767 |                    2.9696e-05  |      0.79862  |     0.621389 |      0.871217 |
| Transformer    | mean_excess_ce_H |       16 |                      3 |  0.00158167  | 0.000116521 |        0.000488383 |                    0.000116521 |      0.807372 |     0.590824 |      0.871899 |

## Dataset-pair rank stability
| architecture   | outcome          |   replicate_1 |   replicate_2 |   n_hmms |   spearman_rho |   bootstrap_ci_low |   bootstrap_ci_high |
|:---------------|:-----------------|--------------:|--------------:|---------:|---------------:|-------------------:|--------------------:|
| GRU            | A_width_shape    |             0 |             1 |       16 |       0.844118 |         0.599387   |            0.951807 |
| GRU            | A_width_shape    |             0 |             2 |       16 |       0.764706 |         0.461989   |            0.886905 |
| GRU            | A_width_shape    |             1 |             2 |       16 |       0.873529 |         0.619888   |            0.969925 |
| GRU            | mean_excess_ce_H |             0 |             1 |       16 |       0.5      |        -0.0283521  |            0.862697 |
| GRU            | mean_excess_ce_H |             0 |             2 |       16 |       0.497059 |         0.00889054 |            0.707036 |
| GRU            | mean_excess_ce_H |             1 |             2 |       16 |       0.479412 |        -0.111119   |            0.836078 |
| Transformer    | A_width_shape    |             0 |             1 |       16 |       0.838235 |         0.503736   |            0.948797 |
| Transformer    | A_width_shape    |             0 |             2 |       16 |       0.75     |         0.376846   |            0.903763 |
| Transformer    | A_width_shape    |             1 |             2 |       16 |       0.770588 |         0.426008   |            0.919406 |
| Transformer    | mean_excess_ce_H |             0 |             1 |       16 |       0.755882 |         0.307326   |            0.955359 |
| Transformer    | mean_excess_ce_H |             0 |             2 |       16 |       0.614706 |         0.0872894  |            0.870511 |
| Transformer    | mean_excess_ce_H |             1 |             2 |       16 |       0.735294 |         0.248849   |            0.922507 |

## Frozen augmented profile versus K-only
|   dataset_replicate | architecture   | outcome          |   n_hmms |   K_only_rmse |   augmented_rmse |   augmented_rmse_reduction_percent |   mean_squared_error_advantage |   exact_two_sided_sign_flip_p |
|--------------------:|:---------------|:-----------------|---------:|--------------:|-----------------:|-----------------------------------:|-------------------------------:|------------------------------:|
|                   0 | GRU            | A_width_shape    |       16 |     0.0117544 |       0.00774257 |                            34.1304 |                    7.82184e-05 |                     0.0628052 |
|                   0 | GRU            | mean_excess_ce_H |       16 |     0.0214868 |       0.0176612  |                            17.8044 |                    0.000149765 |                     0.342529  |
|                   0 | Transformer    | A_width_shape    |       16 |     0.0122567 |       0.0084795  |                            30.8174 |                    7.83247e-05 |                     0.162781  |
|                   0 | Transformer    | mean_excess_ce_H |       16 |     0.0255086 |       0.0176989  |                            30.6162 |                    0.000337441 |                     0.215485  |
|                   1 | GRU            | A_width_shape    |       16 |     0.0115222 |       0.00640664 |                            44.3976 |                    9.17169e-05 |                     0.101959  |
|                   1 | GRU            | mean_excess_ce_H |       16 |     0.016975  |       0.0142469  |                            16.071  |                    8.51752e-05 |                     0.382843  |
|                   1 | Transformer    | A_width_shape    |       16 |     0.0131771 |       0.00859731 |                            34.7559 |                    9.97236e-05 |                     0.252014  |
|                   1 | Transformer    | mean_excess_ce_H |       16 |     0.0253751 |       0.0176298  |                            30.5232 |                    0.000333085 |                     0.0940552 |
|                   2 | GRU            | A_width_shape    |       16 |     0.0147126 |       0.00788901 |                            46.3791 |                    0.000154223 |                     0.0344543 |
|                   2 | GRU            | mean_excess_ce_H |       16 |     0.0189033 |       0.0112682  |                            40.3905 |                    0.000230363 |                     0.0621033 |
|                   2 | Transformer    | A_width_shape    |       16 |     0.0113504 |       0.00828257 |                            27.0285 |                    6.02309e-05 |                     0.293701  |
|                   2 | Transformer    | mean_excess_ce_H |       16 |     0.0268342 |       0.0213323  |                            20.5034 |                    0.000265009 |                     0.314819  |

## Full variance components
| outcome     |   P |   R |   C |   S |   MS_process |   MS_condition |   MS_dataset_within_process |   MS_process_by_condition |   MS_dataset_by_condition |   MS_seed_residual |   process_variance |   dataset_within_process_variance |   process_by_condition_variance |   dataset_by_condition_variance |   seed_residual_variance |   stable_process_share_vs_dataset |   dataset_latent_share |   process_latent_share |
|:------------|----:|----:|----:|----:|-------------:|---------------:|----------------------------:|--------------------------:|--------------------------:|-------------------:|-------------------:|----------------------------------:|--------------------------------:|--------------------------------:|-------------------------:|----------------------------------:|-----------------------:|-----------------------:|
| shape_rmse  |  16 |   3 |   8 |   3 |      19.3197 |    5.18121e-30 |                    0.727712 |                   0.79949 |                  0.829319 |           0.74115  |           0.258637 |                         0         |                       0         |                       0.0293898 |                 0.74115  |                          1        |              0.102038  |               0.897962 |
| excess_ce_H |  16 |   3 |   8 |   3 |      13.7368 |    8.43178e-31 |                    1.25913  |                   1.33909 |                  0.725286 |           0.784618 |           0.164776 |                         0.0222435 |                       0.0682007 |                       0         |                 0.784618 |                          0.881063 |              0.0871542 |               0.912846 |