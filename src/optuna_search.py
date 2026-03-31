import optuna
from train_lora import train

def objective(trial):
    rank = trial.suggest_categorical("rank", [2, 4, 8])
    alpha = trial.suggest_categorical("alpha", [2, 4, 8])

    # run very short training
    _, history = train(rank, alpha, num_epochs=2, save_checkpoint=False)

    val_acc = history["val_accs"][-1]
    return val_acc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

print("Best config:")
print(study.best_params)