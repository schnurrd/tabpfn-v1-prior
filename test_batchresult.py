#!/usr/bin/env python3
"""Demonstrate the clean BatchResult approach for categorical masks."""

import torch
from tabpfn_prior.tabpfn_prior import build_tabpfn_prior

print("BatchResult Approach Demo")
print("=" * 50)

# Create prior with categorical mask enabled
prior = build_tabpfn_prior(
    prior_type='mlp',
    num_steps=1,
    batch_size=2,
    num_datapoints_max=20,
    num_features=5,
    max_num_classes=2,
    device='cpu',
    flexible=True,
    return_categorical_mask=True,
    prior_config={'categorical_feature_p': 0.6}
)

for batch in prior:
    print("\n✓ Batch generated successfully")
    print(f"  Keys: {list(batch.keys())}")
    print(f"  x shape: {batch['x'].shape}")
    
    if 'categorical_mask' in batch:
        mask = batch['categorical_mask']
        print(f"  categorical_mask: {mask}")
        print(f"  Categorical features: {mask.sum().item()}/{len(mask)}")
    else:
        print("  No categorical_mask (not requested or flexible=False)")
    break

print("\n" + "=" * 50)
print("Benefits of BatchResult approach:")
print("  • Clean: No metadata dicts, no tuple length changes")  
print("  • Type-safe: BatchResult is a tuple subclass")
print("  • Backward compatible: x, y, y_ = result still works")
print("  • Simple: Just check hasattr(batch_data, 'categorical_mask')")
print("=" * 50)
