# Failures

## D-F-0001 [critical]

Temperature consistently violated the nominal specification of 297 ± 0.5 K across 949 observations, reaching up to 299.65 K.

- cites: `cca8939e-bd44-47d4-8c49-2d10c39acd38:F-0078`
- variables: temperature_k
- confidence: 0.85 (schema_only)
- domain_tags: environmental, process_control
- time_window: —

## D-F-0002 [critical]

Dissolved oxygen deviated from the nominal specification of 15 ± 0.5 mg/L across 1861 observations, dropping as low as 9.22 mg/L.

- cites: `cca8939e-bd44-47d4-8c49-2d10c39acd38:F-0080`
- variables: dissolved_o2_mg_l
- confidence: 0.85 (schema_only)
- domain_tags: environmental, process_control
- time_window: —

## D-F-0003 [critical]

Substrate concentration showed extreme deviations from the nominal 1 ± 0.1 g/L across 2242 observations, reaching up to 43.81 g/L.

- cites: `cca8939e-bd44-47d4-8c49-2d10c39acd38:F-0079`
- variables: substrate_g_l
- confidence: 0.85 (schema_only)
- domain_tags: metabolism, process_control
- time_window: —

## D-F-0004 [critical]

Biomass concentration significantly exceeded the nominal specification of 0.5 ± 0.05 g/L, reaching over 24.72 g/L at 144.0h.

- cites: `cca8939e-bd44-47d4-8c49-2d10c39acd38:F-0001`, `cca8939e-bd44-47d4-8c49-2d10c39acd38:F-0002`
- variables: biomass_g_l
- confidence: 0.85 (schema_only)
- domain_tags: growth
- time_window: —

## D-F-0005 [critical]

Reactor volume and weight substantially exceeded their nominal specifications (58000 ± 500 L and 62000 ± 500 kg) across over 2000 observations, reaching up to 87141 L and 96184 kg respectively.

- cites: `cca8939e-bd44-47d4-8c49-2d10c39acd38:F-0081`, `cca8939e-bd44-47d4-8c49-2d10c39acd38:F-0082`
- variables: volume_l, weight_kg
- confidence: 0.85 (schema_only)
- domain_tags: process_control
- time_window: —
