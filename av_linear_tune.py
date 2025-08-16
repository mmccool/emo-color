# Switch to linear vs. kernel model & tune regularization

# Linear, stronger smoothing
mapper_lin = EmotionColorMapper(model_type="ridge", alpha=10.0)
mapper_lin.fit(EMOTIONS, SEED_COLORS)

# Nonlinear RBF with heuristic gamma (good for sparse/irregular seeds)
mapper_rbf = EmotionColorMapper(model_type="kernel_rbf", alpha=1.0, rbf_gamma=None)
mapper_rbf.fit(EMOTIONS, SEED_COLORS)
