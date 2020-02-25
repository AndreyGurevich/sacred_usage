class SacredMetricsSender(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        weight_sum = 1.0
        rmse = 0.0
        ground_truth = np.zeros(len(approx))
        prediction = np.zeros(len(approx))

        for i in range(len(approx)):
            ground_truth[i] = target[i]
            prediction[i] = approx[i]
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            rmse += w * ((approx[i] - target[i]) ** 2)

        error_sum = round(np.quantile(np.abs(ground_truth - approx), 0.9), 2)
        ex.log_scalar("90p", error_sum)
        ex.log_scalar("RMSE", rmse)
        return error_sum, weight_sum


model = CatBoostRegressor(
    loss_function=metric_name,
    eval_metric=SacredMetricsSender(),
    **catboost_parameters,
)

p_full = Pool(X, y)
parameters = {
    'iterations': iterations,
    'metric_period': metric_period,
    'loss_function': loss_name,
    'random_seed': seed,
    'custom_metric': ['F1', 'Recall']
}
scores = cv(p_full,
            params=parameters,
            stratified=True,
            fold_count=3,
            as_pandas=False
            )
for key, values_list in scores.items():
    for i, value in enumerate(values_list):
        ex.log_scalar(key, value, i * metric_period)