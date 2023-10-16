import pandas as pd
import numpy as np
from time import time
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error as sk_mean_absolute_percentage_error
from sklearn.metrics import r2_score as sk_r2_score
from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error
from sklearn.metrics import mean_squared_error as sk_mean_squared_error
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()


def evaluate(model, X, y, cv, kind, alg):
    start = time()
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["r2",
                 "neg_mean_absolute_error",
                 "neg_mean_absolute_percentage_error",
                 "neg_mean_squared_error",
                 "neg_root_mean_squared_error",
                 "max_error",
                 "explained_variance"],
    )
    end = time()
    
    r2 = cv_results["test_r2"]
    mae = -cv_results["test_neg_mean_absolute_error"]
    mape = -cv_results["test_neg_mean_absolute_percentage_error"]
    mse = -cv_results["test_neg_mean_squared_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    me = cv_results["test_max_error"]
    ev = cv_results["test_explained_variance"]
    
    print(
        f"R2:                             {r2.mean():.3f} +/- {r2.std():.3f}\n"
        f"Mean Absolute Error:            {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Mean Absolute Percentage Error: {mape.mean():.3f} +/- {mape.std():.3f}\n"
        f"Mean Squared Error:             {mse.mean():.3f} +/- {mse.std():.3f}\n"
        f"Root Mean Squared Error:        {rmse.mean():.3f} +/- {rmse.std():.3f}\n"
        f"Max Error:                      {me.mean():.3f} +/- {me.std():.3f}\n"
        f"Explained Variance:             {ev.mean():.3f} +/- {ev.std():.3f}"
    )
    
    return {"kind":kind,
            "alg":alg,
            "r2_mean":r2.mean(),
            "r2_std":r2.std(),
            "mae_mean":mae.mean(),
            "mae_std":mae.std(),
            "mape_mean":mape.mean(),
            "mape_std":mape.std(),
            "mse_mean":mse.mean(),
            "mse_std":mse.std(),
            "rmse_mean":rmse.mean(),
            "rmse_std":rmse.std(),
            "max_err_mean":me.mean(),
            "max_err_std":me.std(),
            "explained_variance_mean":ev.mean(),
            "explained_variance_std":ev.std(),
            "time_to_run_cv":end-start
    }


def run_cv(model,X,y,alg_name,k=10):
    start = time()
    cv_results = cross_validate(model,X,y,cv=k,scoring=["r2",
                 "neg_mean_absolute_error",
                 "neg_mean_absolute_percentage_error",
                 "neg_mean_squared_error",
                 "neg_root_mean_squared_error"])
    end = time()
    cv_ttf = end-start
    
    r2 = cv_results["test_r2"]
    mae = -cv_results["test_neg_mean_absolute_error"]
    mape = -cv_results["test_neg_mean_absolute_percentage_error"]
    mse = -cv_results["test_neg_mean_squared_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    
    print(
        f"R2:                             {r2.mean():.3f} +/- {r2.std():.3f}\n"
        f"Mean Absolute Error:            {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Mean Absolute Percentage Error: {mape.mean():.3f} +/- {mape.std():.3f}\n"
        f"Mean Squared Error:             {mse.mean():.3f} +/- {mse.std():.3f}\n"
        f"Root Mean Squared Error:        {rmse.mean():.3f} +/- {rmse.std():.3f}\n"
        f"Time to run CV:                 {cv_ttf} seconds\n"
    )
    
    return {"alg_name":alg_name,
            "r2_mean":r2.mean(),
            "r2_std":r2.std(),
            "mae_mean":mae.mean(),
            "mae_std":mae.std(),
            "mape_mean":mape.mean(),
            "mape_std":mape.std(),
            "mse_mean":mse.mean(),
            "mse_std":mse.std(),
            "rmse_mean":rmse.mean(),
            "rmse_std":rmse.std(),
            "time_to_finish":cv_ttf
    }


def get_metrics_cuda_reg(model_name,y_true,y_pred):
    r2 = r2_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred).tolist()
    mse = mean_squared_error(y_true,y_pred).tolist()
    rmse = mean_squared_error(y_true,y_pred,squared=False).tolist()
    mape = sk_mean_absolute_percentage_error(y_true,y_pred)

    return {"model":model_name,
            "r2":r2,
            "mae":mae,
            "mse":mse,
            "rmse":rmse,
            "mape":mape
    }


def get_metrics_clf(model_name,y_true,y_pred,y_probas):
    acc = accuracy_score(y_true,y_pred)
    prec,rec,fbeta,supp = precision_recall_fscore_support(y_true,y_pred,average="weighted")
    if len(y_probas) > 0:
        roc_auc = roc_auc_score(y_true,y_probas,multi_class="ovr")

    return {"model":model_name,
            "accuracy":acc,
            "precision":prec,
            "recall":rec,
            "fbeta":fbeta,
            "support":supp,
            "roc_auc":roc_auc if len(y_probas) > 0 else None
    }
        
    
def get_metrics_sk_reg(model_name,y_true,y_pred):
    r2 = sk_r2_score(y_true,y_pred)
    mae = sk_mean_absolute_error(y_true,y_pred)
    mse = sk_mean_squared_error(y_true,y_pred)
    rmse = sk_mean_squared_error(y_true,y_pred,squared=False)
    mape = sk_mean_absolute_percentage_error(y_true,y_pred)

    return {"model":model_name,
            "r2":r2,
            "mae":mae,
            "mse":mse,
            "rmse":rmse,
            "mape":mape
    }


def get_metrics_reg(model_name,y_true,y_pred):
    r2 = sk_r2_score(y_true,y_pred)
    mae = sk_mean_absolute_error(y_true,y_pred)
    mse = sk_mean_squared_error(y_true,y_pred)
    rmse = sk_mean_squared_error(y_true,y_pred,squared=False)
    mape = sk_mean_absolute_percentage_error(y_true,y_pred)

    return {"model":model_name,
            "r2":r2,
            "mae":mae,
            "mse":mse,
            "rmse":rmse,
            "mape":mape
    }


def train_and_test_cuda(file_name, model_name, model, X_train, X_test, y_train, y_test, plot_label, print_stats=True, dump_model=True, show_plot=True):
    pipeline = make_pipeline(CuStandardScaler(),model)

    print("Fitting model...")
    fit_start = time()
    pipe = pipeline.fit(X_train,y_train)
    fit_end = time()
    print(f"Fitting model... OK! Took {fit_end-fit_start} seconds\n")

    pred_start = time()
    y_pred = pipe.predict(X_test)
    pred_end = time()

    model_metrics = get_metrics_cuda_reg(model_name,y_test.to_numpy(),y_pred.to_numpy())
    model_metrics["time_to_fit"] = fit_end-fit_start
    model_metrics["time_to_predict"] = pred_end-pred_start

    if print_stats:
        for key in model_metrics.keys():
            print(key,model_metrics[key])
    
    if dump_model:
        print("\nDumping model...")
        dump_start = time()
        joblib.dump(pipe,f"./models/{file_name}.joblib",compress=9)
        dump_end = time()
        print(f"Dumping model... OK! Took {dump_end-dump_start} seconds")

    model_metrics_df = pd.DataFrame(model_metrics,index=[0])
    model_metrics_df.to_csv(f"./metrics/{file_name}.csv.zip",index=False,compression="zip")
    
    if show_plot:
        ax = sns.regplot(x=y_test.to_numpy(),y=y_pred.to_numpy(),line_kws={"color": "y"})
        ax.set(xlabel=f"Actual {plot_label}",ylabel=f"Predicted {plot_label}")

        plt.savefig(f"./plots/{file_name}.eps",format="eps",bbox_inches="tight")
        plt.savefig(f"./plots/{file_name}.png",bbox_inches="tight")
    
    return (model,model_metrics)
    

def train_and_test_sk(file_name, model_name, model, X_train, X_test, y_train, y_test, plot_label, print_stats=True, dump_model=True, show_plot=True):
    pipeline = make_pipeline(StandardScaler(), model)

    print("Fitting model...")
    fit_start = time()
    pipe = pipeline.fit(X_train.to_numpy(),y_train.to_numpy())
    fit_end = time()
    print(f"Fitting model... OK! Took {fit_end-fit_start} seconds\n")

    pred_start = time()
    y_pred = pipe.predict(X_test.to_numpy())
    pred_end = time()

    model_metrics = get_metrics_sk_reg(model_name,y_test.to_numpy(),y_pred)
    model_metrics["time_to_fit"] = fit_end-fit_start
    model_metrics["time_to_predict"] = pred_end-pred_start

    if print_stats:
        for key in model_metrics.keys():
            print(key,model_metrics[key])
    
    if dump_model:
        print("\nDumping model...")
        dump_start = time()
        joblib.dump(pipe,f"./models/{file_name}.joblib",compress=9)
        dump_end = time()
        print(f"Dumping model... OK! Took {dump_end-dump_start} seconds")

    model_metrics_df = pd.DataFrame(model_metrics,index=[0])
    model_metrics_df.to_csv(f"./metrics/{file_name}.csv.zip",index=False,compression="zip")
    
    if show_plot:
        ax = sns.regplot(x=y_test.to_numpy(),y=y_pred,line_kws={"color": "y"})
        ax.set(xlabel=f"Actual {plot_label}",ylabel=f"Predicted {plot_label}")

        plt.savefig(f"./plots/{file_name}.eps",format="eps",bbox_inches="tight")
        plt.savefig(f"./plots/{file_name}.png",bbox_inches="tight")
    
    return (model,model_metrics)


def train_and_test(file_name, model_name, model, X_train, X_test, y_train, y_test, plot_label, print_stats=True, dump_model=True, show_plot=True):
    try:
        pipe = joblib.load(f"./models/{file_name}.joblib")
        
        print("Model found! Predicting...")
        y_pred = pipe.predict(X_test)
        print("Model found! Predicting... OK")

        model_metrics = pd.read_csv(f"./metrics/{file_name}.csv.zip")
    except:
        print("Model NOT found!")
        pipeline = make_pipeline(StandardScaler(), model)
    
        print("Fitting model...")
        fit_start = time()
        pipe = pipeline.fit(X_train,y_train)
        fit_end = time()
        print(f"Fitting model... OK! Took {fit_end-fit_start} seconds\n")
    
        pred_start = time()
        y_pred = pipe.predict(X_test)
        pred_end = time()
    
        model_metrics = get_metrics_reg(model_name,y_test,y_pred)
        model_metrics["time_to_fit"] = fit_end-fit_start
        model_metrics["time_to_predict"] = pred_end-pred_start
    
        if print_stats:
            for key in model_metrics.keys():
                print(key,model_metrics[key])
        
        if dump_model:
            print("\nDumping model...")
            dump_start = time()
            joblib.dump(pipe,f"./models/{file_name}.joblib",compress=9)
            dump_end = time()
            print(f"Dumping model... OK! Took {dump_end-dump_start} seconds")
    
        model_metrics_df = pd.DataFrame(model_metrics,index=[0])
        model_metrics_df.to_csv(f"./metrics/{file_name}.csv.zip",index=False,compression="zip")
    
    if show_plot:
        ax = sns.regplot(x=y_test,y=y_pred,line_kws={"color": "y"})
        ax.set(xlabel=f"Actual {plot_label}",ylabel=f"Predicted {plot_label}")

        plt.savefig(f"./plots/{file_name}.eps",format="eps",bbox_inches="tight")
        plt.savefig(f"./plots/{file_name}.png",bbox_inches="tight")
        plt.savefig(f"./plots/{file_name}.pdf",bbox_inches="tight")
    
    return (model,model_metrics)


def run_clf_cuda(file_name, model_name, model, binner, X_train, X_test, y_train_binned, y_test_binned, predict_probas=True):
    pipeline = make_pipeline(StandardScaler(),model)

    print("Fitting model...")
    fit_start = time()
    pipe = pipeline.fit(X_train,y_train_binned.flatten())
    fit_end = time()
    print(f"Fitting model... OK! Took {fit_end-fit_start} seconds")

    pred_start = time()
    y_pred = pipe.predict(X_test)
    if predict_probas:
        y_probas = pipe.predict_proba(X_test)
    else:
        y_probas = []
    pred_end = time()

    y_pred_orig = binner.inverse_transform(y_pred.values.reshape(-1,1)).flatten()
    
    if predict_probas:
        model_metrics = get_metrics_clf(model_name,y_test_binned.get(),y_pred.to_numpy().astype("int32"),y_probas.to_numpy())
    else:
        model_metrics = get_metrics_clf(model_name,y_test_binned.get(),y_pred.to_numpy().astype("int32"),y_probas)
    model_metrics["time_to_fit"] = fit_end-fit_start
    model_metrics["time_to_predict"] = pred_end-pred_start

    for key in model_metrics.keys():
        print(key,model_metrics[key])

    print("\nDumping model...")
    dump_start = time()
    joblib.dump(pipe,f"./models/{file_name}.joblib",compress=9)
    dump_end = time()
    print(f"Dumping model... OK! Took {dump_end-dump_start} seconds")

    model_metrics_df = pd.DataFrame(model_metrics,index=[0])
    model_metrics_df.to_csv(f"./metrics/{file_name}.csv.zip",index=False,compression="zip")
    
    
def run_clf_sk(file_name, model_name, model, binner, X_train, X_test, y_train_binned, y_test_binned, predict_probas=True):
    pipeline = make_pipeline(StandardScaler(),model)

    print("Fitting model...")
    fit_start = time()
    pipe = pipeline.fit(X_train.to_numpy(),y_train_binned.flatten().get())
    fit_end = time()
    print(f"Fitting model... OK! Took {fit_end-fit_start} seconds")

    pred_start = time()
    y_pred = pipe.predict(X_test.to_numpy())
    if predict_probas:
        y_probas = pipe.predict_proba(X_test.to_numpy())
    else:
        y_probas = []
    pred_end = time()

    y_pred_orig = binner.inverse_transform(y_pred.reshape(-1,1)).flatten()
        
    model_metrics = get_metrics_clf(model_name,y_test_binned.get(),y_pred,y_probas)
    model_metrics["time_to_fit"] = fit_end-fit_start
    model_metrics["time_to_predict"] = pred_end-pred_start

    for key in model_metrics.keys():
        print(key,model_metrics[key])

    print("\nDumping model...")
    dump_start = time()
    joblib.dump(pipe,f"./models/{file_name}.joblib",compress=9)
    dump_end = time()
    print(f"Dumping model... OK! Took {dump_end-dump_start} seconds")

    model_metrics_df = pd.DataFrame(model_metrics,index=[0])
    model_metrics_df.to_csv(f"./metrics/{file_name}.csv.zip",index=False,compression="zip")


def run_stratified_kfold(model_name,model,X,y):
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)

    r2 = []
    mae = []
    mse = []
    rmse = []
    mape = []
    ttf = []
    
    splits = skf.split(X,y)

    for i, (train_index,val_index) in enumerate(splits):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        print(f"Running fold {i}...")
        start = time()
        model.fit(X_train,y_train)
        end = time()
        print(f"Fold {i} took {end-start} seconds to fit!")

        y_pred = model.predict(X_val)
        
        fold_metrics = get_metrics_sk_reg(model_name,y_val,y_pred)

        r2.append(fold_metrics["r2"])
        mae.append(fold_metrics["mae"])
        mse.append(fold_metrics["mse"])
        rmse.append(fold_metrics["rmse"])
        mape.append(fold_metrics["mape"])
        ttf.append(end-start)

    r2_np = np.array(r2)
    mae_np = np.array(mae)
    mse_np = np.array(mse)
    rmse_np = np.array(rmse)
    mape_np = np.array(mape)
    ttf_np = np.array(ttf)

    model_cv_metrics = {"model":model_name,"r2":r2_np.mean(),"mae":mae_np.mean(),"mape":mape_np.mean(),"mse":mse_np.mean(),"rmse":rmse_np.mean(),"time_to_fit":ttf_np.mean()}
    
    return model_cv_metrics


def run_param_search(estimator, param_grid, file_name, model_name, X_train, y_train, verbose=0):
    import os
    
    if f"{file_name}.csv" not in os.listdir("./params"):
        pg = ParameterGrid(param_grid)
        print(f"Testing {len(pg)} param combinations for {model_name}. CV=5. Total fits: {len(pg)*5}\n")

        reg = GridSearchCV(estimator, param_grid, scoring="r2", verbose=verbose, n_jobs=-1)

        print(f"Searching space...")
        fit_start = time()
        fitted_reg = reg.fit(X_train, y_train)
        fit_end = time()
        print(f"Searching space... OK! Took {fit_end - fit_start} seconds")

        best_params = fitted_reg.best_params_
        best_score = fitted_reg.best_score_

        best_params["score"] = best_score
        best_params["model"] = model_name
        
        for key in best_params:
            best_params[key] = str(best_params[key])

        best_params_df = pd.DataFrame(best_params, index=[0])

        print(f"Storing best params...")
        best_params_df.to_csv(f"./params/{file_name}.csv",index=False)
        print(f"Storing best params... OK")
        
    else:
        print(f"Best params for {model_name} already found!")
        best_params_df = pd.read_csv(f"./params/{file_name}.csv")
    
    return best_params_df


def params_to_dict(file_name):
    import ast

    best_params = pd.read_csv(f"./params/{file_name}.csv")
    best_params.drop(["score","model"],inplace=True,axis=1)

    best_params = best_params.to_dict(orient='list')

    params = dict()
    for key in best_params.keys():
        try:
            params[key] = ast.literal_eval(best_params[key][0])
        except:
            params[key] = best_params[key][0]

    return params


def plot_prediction(actual,pred,plot_title,ylabel,file_name):
    df = pd.DataFrame({"Actual":actual,"Pred.":pred})
    sample = df.sample(100).reset_index()[["Actual","Pred."]]

    ax = sample.plot()
    ax.set(title=plot_title, ylabel=ylabel, xlabel="Observation no.")

    plt.savefig(f"./plots/{file_name}.eps",format="eps",bbox_inches="tight")
    plt.savefig(f"./plots/{file_name}.png",bbox_inches="tight")
    plt.savefig(f"./plots/{file_name}.pdf",bbox_inches="tight")