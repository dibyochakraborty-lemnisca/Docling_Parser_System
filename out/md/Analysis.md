# Analysis

## D-A-0001 [spec_alignment]

Schema specifications for volume, weight, and substrate appear to represent initial setpoints rather than trajectory bounds, given the continuous deviations observed across hundreds of data points in a fed-batch process.

- cites: `fad84738-3e85-4f88-a113-b3724cccf80a:F-0078`, `fad84738-3e85-4f88-a113-b3724cccf80a:F-0080`, `fad84738-3e85-4f88-a113-b3724cccf80a:F-0081`
- variables: volume_l, weight_kg, substrate_g_l
- confidence: 0.80 (process_priors)
- domain_tags: data_quality, process_control
